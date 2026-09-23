"""What the pilot run on real reports produced, and what failed.

    python scripts/pilot_report.py --data-dir D

Reads the tables Phases 1-4 wrote and writes ``interim/qa/pilot/``:

* ``pdf_vs_xbrl_by_field.csv`` / ``pdf_vs_xbrl_mismatches.csv`` - the PDF reader against the
  exchange XBRL filing, wherever both exist (the measured accuracy of Phase 3 on real reports);
* ``unrecoverable_by_class.csv`` - company-years with no usable financials, by class (the number
  that decides whether the pair-matched design survives);
* ``coverage_by_class.csv`` - sections, auditor flags and ratios found, by class;
* ``signal_check.csv`` - the pilot signal check: each feature by class, paired within pair and
  horizon, and whether it is constant (the failure mode of Phase 4);
* ``summary.md`` - the numbers above in one page.

Descriptive only. With ~25 pairs nothing here is a result; it says whether the pipeline
produces features that vary and whether they point the expected way.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bpp.config import Paths  # noqa: E402

RATIOS = ["current_ratio", "quick_ratio", "cash_to_assets", "ebitda_margin", "roce", "roa",
          "debt_to_equity", "debt_to_ebitda", "interest_coverage", "retained_earnings_to_assets",
          "promoter_pledge", "altman_z_em"]


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def _cls(label) -> str:
    if pd.isna(label):
        return "unlabelled"
    return "distressed" if str(label) in ("1", "1.0", "True") else "healthy"


def pdf_vs_xbrl(fig: pd.DataFrame, out: Path) -> dict:
    if fig.empty or "xbrl_pdf_diff" not in fig:
        return {}
    m = fig[fig["xbrl_pdf_diff"].notna()].copy()
    if "xbrl_value_cr" in m:                         # a row the identity check switched to the report
        m["xbrl_cr"] = m["xbrl_value_cr"].where(m["xbrl_value_cr"].notna(), m["value_cr"])
    else:
        m["xbrl_cr"] = m["value_cr"]
    m = m[m["xbrl_cr"].astype(float) != 0]           # XBRL zero = not reported by many small filers
    m["within_1pct"] = m["xbrl_pdf_diff"] <= 0.01
    m["within_5pct"] = m["xbrl_pdf_diff"] <= 0.05
    arb = m["arbitration"].fillna("").astype(str) if "arbitration" in m else pd.Series("", index=m.index)
    uc = m["unit_check"].fillna("").astype(str) if "unit_check" in m else pd.Series("", index=m.index)
    m["disagreement"] = np.select(
        [m["within_1pct"], arb.str.len() > 0, uc == "pdf_unit_suspect"],
        ["agree", "xbrl_error_report_balances", "report_read_at_wrong_scale"],
        default="report_misread_or_undetermined")
    by = (m.groupby("field").agg(n=("within_1pct", "size"), within_1pct=("within_1pct", "mean"),
                                 within_5pct=("within_5pct", "mean"))
          .sort_values("n", ascending=False).round(3))
    by.to_csv(out / "pdf_vs_xbrl_by_field.csv")
    bad = m[~m["within_1pct"]].copy()
    bad["ratio_pdf_to_xbrl"] = (bad["pdf_value_cr"] / bad["xbrl_cr"]).round(4)
    cols = [c for c in ["firm_id", "fy", "field", "xbrl_cr", "pdf_value_cr", "ratio_pdf_to_xbrl",
                        "xbrl_pdf_diff", "disagreement", "value_cr", "source", "pdf_source",
                        "pdf_source_doc_id", "pdf_page", "unit_check", "arbitration"]
            if c in bad]
    bad.sort_values("xbrl_pdf_diff", ascending=False)[cols].to_csv(out / "pdf_vs_xbrl_mismatches.csv", index=False)
    by_src = m.groupby("pdf_source")["within_1pct"].agg(["size", "mean"]).round(3) if "pdf_source" in m else None
    fy_checks = (fig.drop_duplicates(["firm_id", "fy"])["unit_check"].replace("", np.nan).dropna()
                 .value_counts().to_dict() if "unit_check" in fig else {})
    return {"n": len(m), "firm_years": m[["firm_id", "fy"]].drop_duplicates().shape[0],
            "within_1pct": round(m["within_1pct"].mean(), 3), "within_5pct": round(m["within_5pct"].mean(), 3),
            "by_field": by, "by_source": by_src, "unit_checks": fy_checks,
            "disagreements": m["disagreement"].value_counts().to_dict()}


def unrecoverable(fin: pd.DataFrame, out: Path) -> pd.DataFrame:
    if fin.empty:
        return pd.DataFrame()
    f = fin.copy()
    f["class"] = f["label"].map(_cls)
    rows = []
    for cls, g in f.groupby("class"):
        rows.append({"class": cls, "company_years": len(g),
                     "unrecoverable": int(g["financials_missing"].fillna(False).astype(bool).sum()),
                     "included_financials": int(g.get("included_financials", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
                     **{f"source_{k}": int(v) for k, v in g["financials_source"].value_counts().items()}})
    t = pd.DataFrame(rows).fillna(0)
    t.to_csv(out / "unrecoverable_by_class.csv", index=False)
    return t


def coverage(lab: pd.DataFrame, lang: pd.DataFrame, rat: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    if len(lab):
        d = lab.copy()
        d["class"] = d["label"].map(_cls)
        for col in [c for c in d.columns if c.startswith("has_")] + ["included"]:
            for cls, g in d.groupby("class"):
                rows.append({"table": "documents_labeled", "column": col, "class": cls, "n": len(g),
                             "share_true": round(g[col].astype(str).isin(["True", "1", "1.0"]).mean(), 3)})
    if len(lang):
        d = lang.copy()
        d["class"] = d["label"].map(_cls)
        for col in ["has_going_concern", "has_emphasis_of_matter", "has_modified_opinion_basis",
                    "caro_default_flag", "caro_statutory_dues_flag", "audit_opinion_found", "has_mdna",
                    "has_auditor_report", "has_caro_annexure"]:
            if col in d:
                for cls, g in d.groupby("class"):
                    v = pd.to_numeric(g[col].replace({True: 1, False: 0, "True": 1, "False": 0}), errors="coerce")
                    rows.append({"table": "language_features", "column": col, "class": cls, "n": len(g),
                                 "share_true": round(v.mean(), 3) if v.notna().any() else np.nan,
                                 "share_missing": round(v.isna().mean(), 3)})
    if len(rat):
        d = rat.copy()
        d["class"] = d["label"].map(_cls)
        for col in [c for c in RATIOS if c in d]:
            for cls, g in d.groupby("class"):
                rows.append({"table": "ratios", "column": col, "class": cls, "n": len(g),
                             "share_computed": round(pd.to_numeric(g[col], errors="coerce").notna().mean(), 3)})
    t = pd.DataFrame(rows)
    t.to_csv(out / "coverage_by_class.csv", index=False)
    return t


def signal_check(frame: pd.DataFrame, features: list[str], source: str) -> pd.DataFrame:
    """Per feature: class medians, and the share of pairs (same pair, same horizon) where the
    insolvent firm's value is higher. Only rows used in modelling."""
    if frame.empty:
        return pd.DataFrame()
    f = frame.copy()
    if "included" in f:
        f = f[f["included"].astype(str).isin(["True", "1", "1.0"])]
    f["class"] = f["label"].map(_cls)
    rows = []
    for col in features:
        if col not in f:
            continue
        v = pd.to_numeric(f[col].replace({True: 1, False: 0, "True": 1, "False": 0}), errors="coerce")
        f["_v"] = v
        d = f[f["class"] == "distressed"][["pair_id", "horizon", "_v"]]
        h = f[f["class"] == "healthy"][["pair_id", "horizon", "_v"]]
        p = d.merge(h, on=["pair_id", "horizon"], suffixes=("_d", "_h")).dropna()
        diff = p["_v_d"] - p["_v_h"]
        nonzero = diff[diff != 0]
        rows.append({
            "table": source, "feature": col,
            "n_distressed": int(f.loc[f["class"] == "distressed", "_v"].notna().sum()),
            "n_healthy": int(f.loc[f["class"] == "healthy", "_v"].notna().sum()),
            "median_distressed": f.loc[f["class"] == "distressed", "_v"].median(),
            "median_healthy": f.loc[f["class"] == "healthy", "_v"].median(),
            "mean_distressed": f.loc[f["class"] == "distressed", "_v"].mean(),
            "mean_healthy": f.loc[f["class"] == "healthy", "_v"].mean(),
            "n_pairs": len(p),
            "pairs_distressed_higher": int((diff > 0).sum()),
            "pairs_distressed_lower": int((diff < 0).sum()),
            "share_distressed_higher_of_untied": round((nonzero > 0).mean(), 3) if len(nonzero) else np.nan,
            "constant": bool(v.nunique(dropna=True) <= 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    a = ap.parse_args()
    paths = Paths(a.data_dir)
    out = paths.qa / "pilot"
    out.mkdir(parents=True, exist_ok=True)

    cohort = _read(paths.cohort)
    lab = _read(paths.documents_labeled)
    fig = _read(paths.financials_figures)
    fin = _read(paths.financials_extracted)
    rat = _read(paths.ratios)
    lang = _read(paths.language_features)
    cin = _read(paths.interim / "report_cin_check.csv")
    text = _read(paths.interim / "text_extraction_summary.csv")

    acc = pdf_vs_xbrl(fig, out)
    unrec = unrecoverable(fin, out)
    cov = coverage(lab, lang, rat, out)

    from bpp.nlp.features import CORE_LANGUAGE_FEATURES
    extra_lang = ["auditor_lm_litigious_ratio", "mdna_lm_litigious_ratio", "has_emphasis_of_matter",
                  "has_modified_opinion_basis", "drift_cosine"]
    sig = pd.concat([signal_check(lang, CORE_LANGUAGE_FEATURES + extra_lang, "language_features"),
                     signal_check(rat, RATIOS + ["altman_z_dprime"], "ratios")], ignore_index=True)
    sig.to_csv(out / "signal_check.csv", index=False)

    L = ["# Pilot on real reports - what came out", ""]
    if len(cohort):
        L.append(f"- Cohort: {cohort['pair_id'].nunique()} pairs, {len(cohort)} firms.")
    if len(text):
        L.append(f"- Reports read: {len(text)}; pages {int(text['n_pages'].sum()):,}, of which OCR "
                 f"{int(text['n_ocr_pages'].sum()):,}; failures {int(text['error'].notna().sum()) if 'error' in text else 0}.")
    if len(lab):
        inc = lab["included"].astype(str).isin(["True", "1", "1.0"])
        L.append(f"- Documents labelled: {len(lab)}; included for modelling: {int(inc.sum())}; "
                 f"exclusions: {lab.loc[~inc, 'exclude_reason'].value_counts().to_dict()}.")
    if acc:
        L.append(f"- PDF reader vs exchange XBRL: {acc['n']} figures over {acc['firm_years']} company-years; "
                 f"within 1%: {acc['within_1pct']:.1%}, within 5%: {acc['within_5pct']:.1%}. "
                 f"Where they differ: {acc['disagreements']}. "
                 f"Company-year checks: {acc['unit_checks']}.")
    if len(unrec):
        for _, r in unrec.iterrows():
            L.append(f"- Unrecoverable company-years, {r['class']}: {int(r['unrecoverable'])} of {int(r['company_years'])}.")
    if len(cin):
        L.append(f"- Report CIN check (insolvent firms): {cin['report_cin_check'].value_counts().to_dict()}.")
    if len(sig):
        const = sig[sig["constant"]]["feature"].tolist()
        L.append(f"- Constant features in the pilot: {const or 'none'}.")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    if acc:
        print("\nPDF vs XBRL by field\n", acc["by_field"].to_string())
        if acc["by_source"] is not None:
            print("\nby PDF source\n", acc["by_source"].to_string())
    if len(unrec):
        print("\n", unrec.to_string(index=False))
    if len(sig):
        cols = ["table", "feature", "n_pairs", "median_distressed", "median_healthy",
                "pairs_distressed_higher", "pairs_distressed_lower", "constant"]
        print("\n", sig[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
