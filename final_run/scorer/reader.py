"""Read one annual report (PDF) into the inputs the models were trained on.

The same code that built the training data does the work - the team's `bpp` package, copied into
scorer/bpp with two fixes found while building this scorer (a CARO clause wrapped over two lines
read as a default; an auditor's sentence naming the balance sheet read as the balance sheet).
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))                  # scorer/bpp: the copy shipped to the GPU computer
if not (HERE / "bpp").exists():                    # inside the repository: the bpp package itself
    sys.path.append(str(HERE.parents[1] / "src"))
try:                                           # tqdm is only used for progress bars
    import tqdm  # noqa: F401
except ImportError:
    sys.path.append(str(HERE / "shims"))

import numpy as np                                                        # noqa: E402
import pandas as pd                                                       # noqa: E402

from bpp.config import load_config                                        # noqa: E402
from bpp.extract.pdf_text_pdfium import extract_pdf_pdfium                # noqa: E402
from bpp.extract.sections import report_year_check, segment_document     # noqa: E402
from bpp.features.financials import (_drop_impossible_components, _hit_row,  # noqa: E402
                                     derive_missing_totals, extract_document, validate_year)
from bpp.features.ratios import ALTMAN_DISTRESS, ALTMAN_SAFE, compute_ratios  # noqa: E402
from bpp.features.statements import STANDARD_FIELDS, best_hits           # noqa: E402
from bpp.nlp import features as nlpf                                      # noqa: E402
from bpp.nlp.drift import drift_features                                  # noqa: E402
from bpp.nlp.lexicon import load_lm_dictionary                            # noqa: E402
from bpp.nlp.lm import perplexity as lm_perplexity                        # noqa: E402

logging.getLogger("bpp").setLevel(logging.ERROR)
REQUIRED = ["total_assets", "current_assets", "current_liabilities", "total_equity"]


def config() -> dict[str, Any]:
    cfg = load_config(HERE / "config.yaml")
    try:                                       # ruled-table reading is an accuracy bonus only
        import pdfplumber  # noqa: F401
    except ImportError:
        cfg["financials"]["use_table_extraction"] = False
    return cfg


_lexicon = None


def lexicon():
    global _lexicon
    if _lexicon is None:
        import os
        for p in (Path(os.environ.get("BPP_LM_DICTIONARY", HERE / "loughran_mcdonald.csv")),
                  HERE / "loughran_mcdonald.csv", HERE.parent / "data" / "loughran_mcdonald.csv",
                  HERE.parents[1] / "data" / "manual" / "loughran_mcdonald.csv"):
            if p.exists():
                _lexicon = load_lm_dictionary(p)
                break
        else:
            raise FileNotFoundError("scorer/loughran_mcdonald.csv is missing (the Loughran-McDonald "
                                    "dictionary, bpp-data/manual/loughran_mcdonald.csv)")
    return _lexicon


_COMPANY = re.compile(r"to\s+the\s+members\s+of\s+([A-Z0-9][\w&.,()'\- ]{2,80}?(?:limited|ltd\.?))", re.I)
_FY_IN_NAME = re.compile(r"FY\s*-?(\d{4})", re.I)


def read_pdf(pdf: Path, cfg: dict[str, Any], fy: int | None = None) -> dict[str, Any]:
    """Pages, sections and the report year of one PDF."""
    res = extract_pdf_pdfium(Path(pdf), cfg)
    pages = res["pages"]
    yc = report_year_check(pages, fy)
    if fy is None:
        m = _FY_IN_NAME.search(Path(pdf).stem)
        fy = yc.get("newest_named_fy") or (int(m.group(1)) if m else None)
    payload = segment_document(pages, cfg)
    payload.update(doc_id=Path(pdf).stem, firm_id="new", fy=fy)
    aud = (payload["sections"].get("auditor_report") or {}).get("text") or ""
    m = _COMPANY.search(aud[:3000]) or _COMPANY.search(" ".join(p["text"] for p in pages[:6]))
    company = re.sub(r"\s+", " ", m.group(1)).strip() if m else None
    return dict(pdf=str(pdf), pages=pages, payload=payload, fy=fy, year_check=yc, company=company,
                n_pages=res["n_pages"], n_ocr=res["n_ocr_pages"], n_needs_ocr=res["n_needs_ocr_pages"],
                n_empty=res["n_empty_pages"], seconds=res["seconds"])


def language(doc: dict[str, Any], lms: list | None = None, prev: dict[str, Any] | None = None) -> dict[str, Any]:
    """The engineered language features, perplexity and (with last year's report) drift."""
    f = nlpf.document_features(doc["payload"], lexicon())
    doc["lang_raw"] = f
    mdna = (doc["payload"]["sections"].get("mdna") or {}).get("text") or ""
    out = dict(f)
    out["perplexity_fold"] = (float(np.mean([lm_perplexity(m, mdna) for m in lms]))
                              if (lms and mdna) else None)
    if prev is not None:
        if "lang_raw" not in prev:
            prev["lang_raw"] = nlpf.document_features(prev["payload"], lexicon())
        pf = prev["lang_raw"]
        pm = (prev["payload"]["sections"].get("mdna") or {}).get("text") or ""
        d = drift_features(mdna, pm,
                           {"lm_negative_ratio": f.get("mdna_lm_negative_ratio"),
                            "hedge_density": f.get("mdna_hedge_density"), "fog_index": f.get("mdna_fog_index")},
                           {"lm_negative_ratio": pf.get("mdna_lm_negative_ratio"),
                            "hedge_density": pf.get("mdna_hedge_density"), "fog_index": pf.get("mdna_fog_index")})
    else:
        d = drift_features(mdna, None)
    out.update({k: v for k, v in d.items() if k.startswith("drift_")})
    return out


def financials(doc: dict[str, Any], cfg: dict[str, Any], override: dict[str, float] | None = None) -> dict[str, Any]:
    """The statement figures for the report year (Rs crore), checks, and the 11 ratios."""
    fy = doc["fy"]
    pj = {"pages": doc["pages"], "doc_id": doc["payload"]["doc_id"], "firm_id": "new", "fy": fy}
    hits, rep = extract_document(pj, cfg, doc["pdf"] if cfg["financials"]["use_table_extraction"] else None)
    rows = [_hit_row(h, pj["doc_id"], rep["fy"], rep["statements"]) for h in best_hits(hits).values()]
    by_year: dict[int, dict[str, float]] = {}
    pages: dict[str, int] = {}
    for r in rows:
        if r["fy"] is None or pd.isna(r["fy"]):
            continue
        by_year.setdefault(int(r["fy"]), {})[r["field"]] = float(r["value_cr"])
        if int(r["fy"]) == fy:
            pages[r["field"]] = r.get("page")
    cur = dict(by_year.get(fy, {}))
    source = {k: "report" for k in cur}
    for k, v in (override or {}).items():
        if k in STANDARD_FIELDS and v is not None and not pd.isna(v):
            cur[k] = float(v); source[k] = "supplied"
    wide = pd.DataFrame([{"firm_id": "new", "fy": fy, **{f: cur.get(f, np.nan) for f in STANDARD_FIELDS}}])
    wide = derive_missing_totals(wide)
    prev_assets = by_year.get(fy - 1, {}).get("total_assets") if fy else None
    val = validate_year(wide.iloc[0], cfg, reference_assets=None, previous_assets=prev_assets)
    wide["validation_flags"] = val.get("validation_flags", "")
    wide = _drop_impossible_components(wide)
    row = wide.iloc[0]
    fin = {f: (None if pd.isna(row.get(f)) else float(row.get(f))) for f in STANDARD_FIELDS}
    for f in str(row.get("derived_fields") or "").split(";"):
        if f:
            source[f] = "derived"
    rr = compute_ratios(fin)
    missing = [f for f in REQUIRED if fin.get(f) is None]
    z = rr.values.get("altman_z_em")
    return dict(fields=fin, source=source, pages=pages, statements=rep["statements"],
                missing_statements=rep["missing_statements"], previous_year=by_year.get(fy - 1, {}) if fy else {},
                ratios={k: rr.values.get(k) for k in rr.values}, ratio_reasons=rr.reasons,
                indicators=rr.indicators, altman_z_dprime=None if z is None else z - 3.25,
                altman_zone=rr.altman_zone, altman_cutoffs=(ALTMAN_DISTRESS, ALTMAN_SAFE),
                validation_flags=str(row.get("validation_flags") or ""),
                missing_required=missing, financials_missing=float(bool(missing)))


# ------------------------------------------------------------------ evidence for the flags
def _clauses(pattern, text, asserted=True):
    out = []
    for clause in nlpf._CLAUSE_SPLIT.split(text or ""):
        c = re.sub(r"\s+", " ", clause).strip()
        for m in pattern.finditer(c):
            neg = bool(nlpf._NEGATION_CUE.search(c[:m.start()]))
            if neg != asserted:
                out.append(c[:400])
                break
    return out


def evidence(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """What the auditor actually wrote, for every flag that is raised."""
    s = doc["payload"]["sections"]
    caro = (s.get("caro_annexure") or {})
    caro_text = caro.get("text") or ""
    ev = []
    for name, key, sec in [("Material uncertainty on going concern", "going_concern", s.get("going_concern")),
                           ("Basis for a modified opinion", "basis_for_modified_opinion", s.get("basis_for_modified_opinion")),
                           ("Emphasis of matter", "emphasis_of_matter", s.get("emphasis_of_matter"))]:
        if sec and sec.get("text"):
            t = re.sub(r"\s+", " ", sec["text"]).strip()
            ev.append(dict(flag=key, title=name, quote=t[:420] + ("..." if len(t) > 420 else ""),
                           where="Auditor's report"))
    if caro_text:
        where = f"CARO annexure, pages {caro.get('start_page')}-{caro.get('end_page')}"
        for q in _clauses(nlpf._CARO_DEFAULT, caro_text)[:2]:
            ev.append(dict(flag="caro_default_flag", title="Default or delay in repaying lenders", quote=q, where=where))
        st = [m.group(0) for m in nlpf._STATUTORY_IRREGULAR.finditer(caro_text)]
        qs = _clauses(nlpf._STATUTORY_FAILURE, caro_text)[:2]
        if st and not qs:
            qs = [_around(caro_text, nlpf._STATUTORY_IRREGULAR)]
        for q in qs:
            ev.append(dict(flag="caro_statutory_dues_flag", title="Statutory dues unpaid or paid late", quote=q, where=where))
    op = doc["payload"].get("audit_opinion")
    if op in ("qualified", "adverse", "disclaimer"):
        ev.insert(0, dict(flag="audit_opinion_severity", title=f"Audit opinion: {op}", quote="", where="Auditor's report"))
    return ev


def _around(text, pattern, width=200):
    m = pattern.search(text)
    if not m:
        return ""
    return re.sub(r"\s+", " ", text[max(0, m.start() - width):m.end() + width]).strip()


def coverage(doc: dict[str, Any], fin: dict[str, Any], has_prev: bool) -> list[tuple[str, str]]:
    """(level, message): what was found and how far to trust the score."""
    s = doc["payload"]["sections"]
    notes = []
    for key, name, level in [("auditor_report", "Auditor's report", "warn"), ("caro_annexure", "CARO annexure", "info"),
                             ("mdna", "Management discussion and analysis (MD&A)", "warn")]:
        sec = s.get(key)
        if sec:
            notes.append(("ok", f"{name}: pages {sec.get('start_page')}-{sec.get('end_page')}, "
                                f"{sec.get('n_chars', 0):,} characters"))
        else:
            notes.append((level, f"{name}: not found - its features are treated as missing"))
    aud = s.get("auditor_report")
    if aud and aud.get("n_chars", 0) < 5000:
        notes.append(("warn", f"The auditor's report read is short ({aud.get('n_chars', 0):,} characters): "
                              f"part of it may not have been found"))
    if doc["payload"].get("audit_opinion") not in ("unmodified", "qualified", "adverse", "disclaimer"):
        notes.append(("warn", "The auditor's opinion could not be identified"))
    got = [k for k, v in fin["fields"].items() if v is not None]
    if fin["missing_required"]:
        notes.append(("warn", f"Financial statements: {len(got)} of {len(fin['fields'])} figures read; missing "
                              f"{', '.join(fin['missing_required'])} - the ratios are incomplete. Supply them with "
                              f"--financials."))
    else:
        st = fin["statements"].get("balance_sheet", {})
        notes.append(("ok", f"Financial statements: {len(got)} of {len(fin['fields'])} figures read "
                            f"(balance sheet p.{st.get('start_page')}, {st.get('scope')}, in Rs {st.get('unit')})"))
    if fin["validation_flags"]:
        notes.append(("warn", f"Checks on the figures: {fin['validation_flags']}"))
    if not has_prev:
        notes.append(("info", "No previous-year report given: the year-on-year change features are missing "
                              "(pass --prev, or keep FY<year-1>.pdf next to the report)"))
    yc = doc["year_check"]
    if yc.get("newest_named_fy") and doc["fy"] and yc["newest_named_fy"] != doc["fy"]:
        notes.append(("warn", f"The report mostly names FY{yc['newest_named_fy']}, not FY{doc['fy']}"))
    if doc["n_needs_ocr"]:
        notes.append(("warn", f"{doc['n_needs_ocr']} scanned pages could not be read (no OCR installed)"))
    return notes
