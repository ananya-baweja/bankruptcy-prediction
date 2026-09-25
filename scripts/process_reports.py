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
import re
import sys
import time
from collections import Counter
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


def _write_summary(summ: Path, rows: list[dict]) -> None:
    old = pd.read_csv(summ) if summ.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    if len(old) and len(new):
        old = old[~old["doc_id"].isin(new["doc_id"])]
    pd.concat([old, new], ignore_index=True).to_csv(summ, index=False)


def step_text(paths: Paths, cfg: dict, workers: int, force: bool, cohort_only: bool = False) -> None:
    man = pd.read_csv(paths.documents, dtype=str)
    # The cohort's reports first: OCR takes hours, and reports of firms that found
    # no peer are only needed if they are matched later.
    cohort = paths.processed / "cohort.csv"
    if cohort.exists():
        in_cohort = set(pd.read_csv(cohort, dtype=str)["firm_id"])
        man = man.assign(_first=~man["firm_id"].isin(in_cohort)).sort_values("_first", kind="stable")
        if cohort_only:
            man = man[man["firm_id"].isin(in_cohort)]
    todo = []
    for _, r in man.iterrows():
        out = paths.pages / f"{r['doc_id']}.json"
        if out.exists() and not force:
            continue
        todo.append((r["local_path"], str(out), r["firm_id"], int(r["fy"]), r["doc_id"], cfg))
    log.info("extracting text from %d PDFs with %d workers", len(todo), workers)
    paths.pages.mkdir(parents=True, exist_ok=True)
    summ = paths.interim / "text_extraction_summary.csv"
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_extract_one, a) for a in todo]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows.append(r)
            log.info("[%d/%d] %s %s", i, len(todo), r["doc_id"],
                     r.get("error") or f"{r['n_pages']} pages, {r['n_ocr_pages']} OCR, {r['wall_s']}s")
            if i % 10 == 0:            # a stopped run keeps the summary of what it finished
                _write_summary(summ, rows)
    _write_summary(summ, rows)


#: a CIN as printed or OCR'd: "L 141030R2000PLC006230" is L14103OR2000PLC006230
_CIN_LOOSE = re.compile(r"\b([LU])\s?([\dO]{5})\s?([A-Z0]{2})\s?([\dO]{4})\s?([A-Z]{3})\s?([\dO]{6})\b")


def _cins(text: str) -> Counter:
    """Every CIN in a report, OCR's O-for-0 slips undone, with how often it is printed."""
    out: Counter = Counter()
    for m in _CIN_LOOSE.finditer(text.upper()):
        lu, nic, state, year, kind, reg = m.groups()
        digits = str.maketrans("O", "0")
        out[lu + nic.translate(digits) + state.replace("0", "O") + year.translate(digits) + kind
            + reg.translate(digits)] += 1
    return out


def report_cin_status(expected: str | None, text: str) -> tuple[str, str]:
    """(status, CIN) for one report: confirmed | confirmed_state_changed | confirmed_reg_no |
    mismatch | cin_not_found | no_expected_cin.

    The whole report is read: the own CIN is not always within the first pages, and
    those pages can carry a subsidiary's CIN instead (both seen on the full cohort).
    On a mismatch the CIN printed most often is reported, not the first one.
    """
    found = _cins(text)
    if not expected:
        return "no_expected_cin", found.most_common(1)[0][0] if found else ""
    if not found:
        return "cin_not_found", ""
    if expected in found:
        return "confirmed", expected
    # Andhra Pradesh companies became Telangana ones in 2014: the CIN's state changes,
    # its industry code, year, type and registration number do not
    moved = [c for c in found if c[6:8] != expected[6:8] and c[1:6] == expected[1:6] and c[8:] == expected[8:]]
    if moved:
        return "confirmed_state_changed", moved[0]
    # a change of listing status or company type keeps the registration number and year
    same_reg = [c for c in found if c[-6:] == expected[-6:] and c[8:12] == expected[8:12]]
    if same_reg:
        return "confirmed_reg_no", same_reg[0]
    return "mismatch", found.most_common(1)[0][0]


def step_cincheck(paths: Paths) -> None:
    """Confirm each insolvent firm's listing with the CIN printed in its own report."""
    cohort = pd.read_csv(paths.processed / "cohort.csv", dtype=str)
    rows = []
    for _, c in cohort[cohort["role"] == "distressed"].iterrows():
        checks = []
        for f in sorted(paths.pages.glob(f"{c['firm_id']}_FY*.json")):
            pj = json.loads(f.read_text(encoding="utf-8"))
            text = "\n".join(p["text"] for p in pj["pages"])
            status, found = report_cin_status(c.get("cin") if isinstance(c.get("cin"), str) else None, text)
            checks.append((pj["fy"], status, found))
        best = next((x for x in checks if x[1].startswith("confirmed")), checks[0] if checks else (None, "no_report", ""))
        rows.append({"pair_id": c["pair_id"], "firm_id": c["firm_id"], "company_name": c["company_name"],
                     "ibbi_cin": c.get("cin"), "report_cin_check": best[1], "report_cin": best[2],
                     "checked_fy": best[0], "n_reports_checked": len(checks),
                     "all_checks": "; ".join(f"FY{a}:{b}" for a, b, _ in checks)})
    out = pd.DataFrame(rows)
    out.to_csv(paths.interim / "report_cin_check.csv", index=False)
    log.info("report CIN check: %s", out["report_cin_check"].value_counts().to_dict())


#: the report sections the narrative stream reads, as extracted in Phase 2
EXPORT_SECTIONS = ["mdna", "directors_report", "auditor_report", "caro_annexure",
                   "basis_for_modified_opinion", "going_concern", "emphasis_of_matter"]


def step_export_text(paths: Paths, n_parts: int = 2) -> None:
    """The cohort's report text in ``processed/`` so the dataset travels as one folder.

    One JSON line per labelled report that exists: ids, the audit opinion and each
    section's text (with its pages). Split in ``n_parts`` gzip files so that no part
    outgrows a 20 MB transfer; non-ASCII is escaped so any JSON-lines reader splits
    the lines correctly.
    """
    import gzip

    lab = pd.read_csv(paths.documents_labeled, dtype=str)
    docs = sorted(d for d, h in zip(lab["doc_id"], lab["has_document"]) if str(h) == "True")
    recs = []
    for d in docs:
        f = paths.sections / f"{d}.json"
        if not f.exists():
            continue
        js = json.loads(f.read_text(encoding="utf-8"))
        rec = {"doc_id": d, "firm_id": js.get("firm_id"), "fy": js.get("fy"), "audit_opinion": js.get("audit_opinion"),
               "n_pages": js.get("n_pages"), "n_ocr_pages": js.get("n_ocr_pages")}
        for name in EXPORT_SECTIONS:
            sec = js["sections"].get(name)
            rec[name] = sec.get("text") if sec else None
            if name in ("mdna", "directors_report", "auditor_report", "caro_annexure"):
                rec[name + "_pages"] = [sec.get("start_page"), sec.get("end_page")] if sec else None
        recs.append(rec)
    size = -(-len(recs) // n_parts)
    for k in range(n_parts):
        out = paths.processed / f"report_sections_{k + 1}.jsonl.gz"
        with gzip.open(out, "wt", encoding="utf-8", compresslevel=9) as fh:
            for rec in recs[k * size:(k + 1) * size]:
                fh.write(json.dumps(rec, ensure_ascii=True) + "\n")
    log.info("report text of %d reports -> processed/report_sections_1..%d.jsonl.gz", len(recs), n_parts)


def _cohort_doc_ids(paths: Paths) -> list[str] | None:
    """Reports of the cohort's firms (any year); None when there is no cohort yet."""
    cohort = paths.processed / "cohort.csv"
    if not cohort.exists():
        return None
    firms = set(pd.read_csv(cohort, dtype=str)["firm_id"])
    ids = sorted(p.stem for p in paths.pages.glob("*.json") if p.stem.rsplit("_FY", 1)[0] in firms)
    return ids or None


def step_phases(paths: Paths, cfg: dict, workers: int = 1) -> None:
    """Phases 2-4 on the cohort's reports. Reports of firms that found no peer are
    downloaded too (the insolvent firms' reports come before pairing) but left out here."""
    from bpp.cohort.labels import run_assign_labels
    from bpp.extract.sections import run_extract_sections
    from bpp.features.financials import run_financials
    from bpp.nlp.features import run_language_features
    ids = _cohort_doc_ids(paths)
    log.info("Phases 2-4 on %s reports", len(ids) if ids else "all")
    run_extract_sections(cfg, paths, ids)
    run_assign_labels(cfg, paths)
    run_financials(cfg, paths, ids, workers=workers)
    run_language_features(cfg, paths, ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("step", choices=["text", "cincheck", "phases", "export-text"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cohort-only", action="store_true", help="text: only the cohort's reports")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config()
    paths = Paths(a.data_dir)
    if a.step == "text":
        step_text(paths, cfg, a.workers, a.force, a.cohort_only)
    elif a.step == "export-text":
        step_export_text(paths)
    elif a.step == "cincheck":
        step_cincheck(paths)
    else:
        step_phases(paths, cfg, a.workers)


if __name__ == "__main__":
    main()
