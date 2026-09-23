"""Run Phases 2-4 on downloaded annual reports in a session without PyMuPDF.

    python scripts/process_reports.py --data-dir D text      # page text (pypdfium2 + OCR), parallel
    python scripts/process_reports.py --data-dir D cincheck  # CIN printed in each insolvent firm's report
    python scripts/process_reports.py --data-dir D phases    # sections, labels, financials, language

``text`` writes ``interim/pages/<doc_id>.json`` in exactly the layout of
``bpp extract-text``; everything after it is the project's own code, unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bpp.config import Paths, load_config  # noqa: E402

log = logging.getLogger("process_reports")


def _extract_one(args: tuple[str, str, str, int, str, dict]) -> dict:
    pdf, out, firm_id, fy, doc_id, cfg = args
    from bpp.extract.pdf_text_pdfium import extract_pdf_pdfium, write_pages_json
    t0 = time.time()
    try:
        res = extract_pdf_pdfium(Path(pdf), cfg)
    except Exception as exc:  # noqa: BLE001 - one broken PDF must not stop the batch
        return {"doc_id": doc_id, "error": f"{type(exc).__name__}: {exc}"}
    write_pages_json(Path(out), doc_id, firm_id, fy, pdf, res)
    return {"doc_id": doc_id, **{k: v for k, v in res.items() if k != "pages"}, "wall_s": round(time.time() - t0, 1)}


def step_text(paths: Paths, cfg: dict, workers: int, force: bool) -> None:
    man = pd.read_csv(paths.documents, dtype=str)
    todo = []
    for _, r in man.iterrows():
        out = paths.pages / f"{r['doc_id']}.json"
        if out.exists() and not force:
            continue
        todo.append((r["local_path"], str(out), r["firm_id"], int(r["fy"]), r["doc_id"], cfg))
    log.info("extracting text from %d PDFs with %d workers", len(todo), workers)
    paths.pages.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_extract_one, a) for a in todo]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows.append(r)
            log.info("[%d/%d] %s %s", i, len(todo), r["doc_id"],
                     r.get("error") or f"{r['n_pages']} pages, {r['n_ocr_pages']} OCR, {r['wall_s']}s")
    summ = paths.interim / "text_extraction_summary.csv"
    old = pd.read_csv(summ) if summ.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(old) and len(new):
        old = old[~old["doc_id"].isin(new["doc_id"])]
    pd.concat([old, new], ignore_index=True).to_csv(summ, index=False)


def step_cincheck(paths: Paths, n_pages: int = 12) -> None:
    """Confirm each insolvent firm's listing with the CIN printed in its own report."""
    from bpp.cohort.cin_match import confirm_with_report_cin
    cohort = pd.read_csv(paths.processed / "cohort.csv", dtype=str)
    rows = []
    for _, c in cohort[cohort["role"] == "distressed"].iterrows():
        checks = []
        for f in sorted(paths.pages.glob(f"{c['firm_id']}_FY*.json")):
            pj = json.loads(f.read_text(encoding="utf-8"))
            text = "\n".join(p["text"] for p in pj["pages"][:n_pages])
            status, found = confirm_with_report_cin(c.get("cin"), text)
            checks.append((pj["fy"], status, found))
        best = next((x for x in checks if x[1].startswith("confirmed")), checks[0] if checks else (None, "no_report", ""))
        rows.append({"pair_id": c["pair_id"], "firm_id": c["firm_id"], "company_name": c["company_name"],
                     "ibbi_cin": c.get("cin"), "report_cin_check": best[1], "report_cin": best[2],
                     "checked_fy": best[0], "n_reports_checked": len(checks),
                     "all_checks": "; ".join(f"FY{a}:{b}" for a, b, _ in checks)})
    out = pd.DataFrame(rows)
    out.to_csv(paths.interim / "report_cin_check.csv", index=False)
    log.info("report CIN check: %s", out["report_cin_check"].value_counts().to_dict())


def step_phases(paths: Paths, cfg: dict) -> None:
    from bpp.cohort.labels import run_assign_labels
    from bpp.extract.sections import run_extract_sections
    from bpp.features.financials import run_financials
    from bpp.nlp.features import run_language_features
    run_extract_sections(cfg, paths)
    run_assign_labels(cfg, paths)
    run_financials(cfg, paths)
    run_language_features(cfg, paths)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("step", choices=["text", "cincheck", "phases"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config()
    paths = Paths(a.data_dir)
    if a.step == "text":
        step_text(paths, cfg, a.workers, a.force)
    elif a.step == "cincheck":
        step_cincheck(paths)
    else:
        step_phases(paths, cfg)


if __name__ == "__main__":
    main()
