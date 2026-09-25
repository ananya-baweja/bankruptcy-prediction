"""Assign labels and prediction horizons to each collected annual report.

This is where most data leakage is prevented, so the rules are explicit:

0. The right report. A report whose own text is about another fiscal year (the
   exchange listed it under the wrong year) is excluded as ``report_is_for_another_year``.
1. Publication date. Use the real date the report was filed (from the NSE/BSE
   listing or the manual CSV). If unknown, assume FY end + ~6 months and flag it
   (``pub_date_source = assumed``).
2. Cut-off. A report is excluded if it was published
     * after the insolvency petition was filed (when ``petition_date`` is known), or
     * within ``exclude_within_days_of_admission`` days before admission (or after it).
   Such reports tend to discuss the NCLT case openly, which would let the model
   "cheat".
3. Aligned pairs. With ``keep_pairs_aligned``, if the distressed firm's FY is
   excluded or missing, the healthy peer's same FY is dropped too, so both
   classes cover the same years.
4. Horizon. Remaining FYs of a pair are ranked newest first: t-1, t-2, t-3.
   Only ``n_years_before`` are kept.
5. Review flag. Distressed reports whose text contains CIRP-specific terms
   (resolution professional, committee of creditors, section 7/9/10 petition)
   are flagged ``needs_leakage_review`` for a person to read. The outcome goes in
   ``data/manual/leakage_review.csv`` (``doc_id, decision (exclude|keep), reason``):
   an excluded report gets ``exclude_reason = leakage_review_excluded`` (and its
   pair partner's year follows it out, rule 3); a reviewed report is no longer flagged.

Outputs: data/processed/documents_labeled.csv and data/processed/missing_reports.csv
(missing or late filing is itself a distress signal we will use later).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from bpp.common import doc_id_for, write_csv
from bpp.config import Paths

log = logging.getLogger(__name__)


def _section_summary(paths: Paths, doc_id: str) -> dict[str, Any]:
    f = paths.sections / f"{doc_id}.json"
    if not f.exists():
        return {"sections_extracted": False}
    js = json.loads(f.read_text(encoding="utf-8"))
    out = {"sections_extracted": True,
           "report_year_mismatch": (js.get("year_check") or {}).get("mismatch", ""),
           "audit_opinion": js.get("audit_opinion"),
           "cirp_specific_mentions": js.get("leakage", {}).get("cirp_specific_mentions", 0),
           "ibc_generic_mentions": js.get("leakage", {}).get("ibc_generic_mentions", 0)}
    for name, sec in js.get("sections", {}).items():
        out[f"has_{name}"] = bool(sec and sec.get("n_chars", 0) > 0)
    return out


def assign_labels(frame: pd.DataFrame, documents: pd.DataFrame, cohort: pd.DataFrame,
                  cfg: dict[str, Any], section_info: dict[str, dict[str, Any]] | None = None,
                  leakage_review: dict[str, str] | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    lcfg, n_keep = cfg["labels"], cfg["cohort"]["n_years_before"]
    excl_days = lcfg["exclude_within_days_of_admission"]
    lag = lcfg["assumed_publication_lag_days"]
    section_info = section_info or {}

    frame = frame.copy()
    frame["doc_id"] = [doc_id_for(f, y) for f, y in zip(frame["firm_id"], frame["fy"])]
    docs = documents.copy()
    docs = docs[docs["status"].isin(["downloaded", "registered"])]
    docs = docs.drop_duplicates("doc_id", keep="last")
    df = frame.merge(docs[["doc_id", "source", "local_path", "pub_date"]], on="doc_id", how="left")

    petition = cohort.drop_duplicates("pair_id").set_index("pair_id")["petition_date"]
    df["petition_date"] = pd.to_datetime(df["pair_id"].map(petition), errors="coerce")
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    df["fy_end_date"] = pd.to_datetime(df["fy_end_date"])

    df["has_document"] = df["local_path"].notna()
    df["pub_date"] = pd.to_datetime(df["pub_date"], errors="coerce")
    df["pub_date_source"] = df["pub_date"].notna().map({True: "filing", False: "assumed"})
    df["pub_date"] = df["pub_date"].fillna(df["fy_end_date"] + pd.Timedelta(days=lag))
    df["days_before_reference"] = (df["reference_date"] - df["pub_date"]).dt.days
    df["months_before_reference"] = (df["days_before_reference"] / 30.44).round(1)
    df["filing_delay_days"] = (df["pub_date"] - df["fy_end_date"]).dt.days

    reasons = pd.Series("", index=df.index)
    reasons[~df["has_document"]] = "missing_document"
    # the exchange listed another year's report under this year: as good as missing
    wrong_year = df["doc_id"].map(lambda d: bool(section_info.get(d, {}).get("report_year_mismatch")))
    reasons[wrong_year & (reasons == "")] = "report_is_for_another_year"
    too_close = df["has_document"] & (df["days_before_reference"] < excl_days)
    reasons[too_close & (reasons == "")] = f"published_within_{excl_days}d_of_admission"
    after_pet = df["has_document"] & df["petition_date"].notna() & (df["pub_date"] >= df["petition_date"])
    reasons[after_pet & (reasons == "")] = "published_after_petition"
    leakage_review = leakage_review or {}
    reviewed_out = df["doc_id"].map(lambda d: leakage_review.get(d) == "exclude")
    reasons[reviewed_out & (reasons == "")] = "leakage_review_excluded"
    df["exclude_reason"] = reasons

    if lcfg["keep_pairs_aligned"]:
        bad = df[(df["role"] == "distressed") & (df["exclude_reason"] != "")][["pair_id", "fy"]]
        bad_keys = set(zip(bad["pair_id"], bad["fy"]))
        mask = [(p, y) in bad_keys for p, y in zip(df["pair_id"], df["fy"])]
        mask = pd.Series(mask, index=df.index) & (df["exclude_reason"] == "") & (df["role"] == "healthy")
        df.loc[mask, "exclude_reason"] = "pair_partner_excluded"

    # horizon: rank valid FYs within each pair (using the distressed firm's valid FYs)
    df["horizon"] = pd.NA
    valid = df[df["exclude_reason"] == ""]
    for pair_id, g in valid.groupby("pair_id"):
        fys = sorted(g.loc[g["role"] == "distressed", "fy"].unique(), reverse=True)
        if not lcfg["keep_pairs_aligned"] or not fys:
            fys = sorted(g["fy"].unique(), reverse=True)
        rank = {fy: i + 1 for i, fy in enumerate(fys)}
        idx = g.index
        df.loc[idx, "horizon"] = [f"t-{rank[f]}" if f in rank and rank[f] <= n_keep else pd.NA
                                  for f in g["fy"]]
    over = (df["exclude_reason"] == "") & df["horizon"].isna()
    df.loc[over, "exclude_reason"] = "beyond_n_years_before"

    df["within_12m"] = (df["label"] == 1) & (df["months_before_reference"] <= 12)
    df["within_24m"] = (df["label"] == 1) & (df["months_before_reference"] <= 24)

    extra = pd.DataFrame([{"doc_id": d, **section_info.get(d, {})} for d in df["doc_id"]])
    df = df.merge(extra, on="doc_id", how="left")
    df["leakage_review"] = df["doc_id"].map(lambda d: leakage_review.get(d, ""))
    if "cirp_specific_mentions" in df:
        df["needs_leakage_review"] = (df["label"] == 1) & (df["cirp_specific_mentions"].fillna(0) > 0) \
                                     & (df["exclude_reason"] == "") & (df["leakage_review"] == "")
    df["included"] = df["exclude_reason"] == ""

    missing = df[~df["has_document"]][["pair_id", "firm_id", "company_name", "role", "fy",
                                        "expected_pub_date"]]
    return df, missing


def run_assign_labels(cfg: dict[str, Any], paths: Paths) -> pd.DataFrame:
    frame = pd.read_csv(paths.sample_frame)
    cohort = pd.read_csv(paths.cohort)
    documents = pd.read_csv(paths.documents) if paths.documents.exists() else pd.DataFrame(
        columns=["doc_id", "status", "source", "local_path", "pub_date"])
    frame["doc_id"] = [doc_id_for(f, y) for f, y in zip(frame["firm_id"], frame["fy"])]
    info = {d: _section_summary(paths, d) for d in frame["doc_id"]}
    review: dict[str, str] = {}
    if paths.leakage_review.exists():
        rv = pd.read_csv(paths.leakage_review, dtype=str)
        review = {d: str(x).strip().lower() for d, x in zip(rv["doc_id"], rv["decision"]) if pd.notna(x)}
        log.info("leakage review: %d reports read, %d excluded", len(review),
                 sum(v == "exclude" for v in review.values()))
    df, missing = assign_labels(frame, documents, cohort, cfg, info, review)
    write_csv(df, paths.documents_labeled)
    write_csv(missing, paths.missing_reports)
    inc = df[df["included"]]
    log.info("labelled: %d firm-years, %d included (%d distressed / %d healthy); excluded: %s",
             len(df), len(inc), int((inc["label"] == 1).sum()), int((inc["label"] == 0).sum()),
             df.loc[~df["included"], "exclude_reason"].value_counts().to_dict())
    if "needs_leakage_review" in df and df["needs_leakage_review"].any():
        log.warning("%d included distressed reports mention CIRP terms - read them (needs_leakage_review)",
                    int(df["needs_leakage_review"].sum()))
    return df
