"""Match IBBI corporate-debtor names to listed companies.

IBBI does not say whether a debtor is listed, and names differ in small ways
("Jaypee Infratech Ltd." vs "JAYPEE INFRATECH LIMITED"). We normalise names and
use fuzzy matching (rapidfuzz ``token_sort_ratio``), then sort matches into:

* ``auto_accepted``  score >= auto_accept_score   (accept = Y pre-filled)
* ``needs_review``   review_score <= score < auto  (a person fills accept = Y/N)
* ``no_match``       below review_score            (most IBBI debtors are unlisted)

Human step: open data/interim/ibbi_listed_matches.csv, check the needs_review
rows (and spot-check auto ones), set ``accept`` to Y or N, optionally add
``petition_date`` and ``admission_date_verified`` from the NCLT order / PA PDF,
and save it as data/manual/ibbi_listed_matches_reviewed.csv.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process

from bpp.common import former_names, normalize_company_name, write_csv
from bpp.config import Paths

log = logging.getLogger(__name__)


def match_debtors_to_universe(debtors: pd.DataFrame, universe: pd.DataFrame,
                              cfg: dict[str, Any]) -> pd.DataFrame:
    ncfg = cfg["name_matching"]
    auto, review, k = ncfg["auto_accept_score"], ncfg["review_score"], ncfg["top_k_candidates"]

    uni = universe.dropna(subset=["name_norm"]).reset_index(drop=True)
    choices = uni["name_norm"].tolist()
    out = []
    for _, d in debtors.iterrows():
        queries = [d.get("name_norm") or normalize_company_name(d["corporate_debtor"])]
        queries += [normalize_company_name(n) for n in former_names(d["corporate_debtor"])]
        best: list[tuple[str, float, int]] = []
        for q in queries:
            if not q:
                continue
            best += process.extract(q, choices, scorer=fuzz.token_sort_ratio, limit=k)
        best = sorted(best, key=lambda x: -x[1])
        seen, cands = set(), []
        for _, score, idx in best:
            if idx not in seen:
                seen.add(idx)
                cands.append((score, idx))
        cands = cands[:k]

        rec = {
            "corporate_debtor": d["corporate_debtor"],
            "cirp_announcement_date": d["cirp_announcement_date"],
            "pa_pdf_url": d.get("pa_pdf_url"),
        }
        if cands:
            score, idx = cands[0]
            u = uni.iloc[idx]
            status = ("auto_accepted" if score >= auto else
                      "needs_review" if score >= review else "no_match")
            rec.update({
                "match_status": status,
                "score": round(score, 1),
                "firm_id": u["firm_id"],
                "matched_name": u["company_name"],
                "bse_code": u.get("bse_code"),
                "nse_symbol": u.get("nse_symbol"),
                "isin": u.get("isin"),
                "listing_status": u.get("status"),
                "industry": u.get("industry"),
                "other_candidates": " | ".join(
                    f"{uni.iloc[i]['company_name']} ({s:.0f})" for s, i in cands[1:]),
            })
        else:
            rec.update({"match_status": "no_match", "score": 0})
        rec["accept"] = "Y" if rec["match_status"] == "auto_accepted" else ""
        rec["petition_date"] = ""
        rec["admission_date_verified"] = ""
        rec["business_group"] = ""
        rec["notes"] = ""
        out.append(rec)
    df = pd.DataFrame(out)
    order = {"needs_review": 0, "auto_accepted": 1, "no_match": 2}
    df = df.sort_values(["match_status", "score"], key=lambda s: s.map(order) if s.name == "match_status" else -s)
    return df.reset_index(drop=True)


def run_name_matching(cfg: dict[str, Any], paths: Paths) -> pd.DataFrame:
    debtors = pd.read_csv(paths.cirp_listed_candidates, parse_dates=["cirp_announcement_date"])
    universe = pd.read_csv(paths.listed_universe, dtype={"bse_code": "string"})
    df = match_debtors_to_universe(debtors, universe, cfg)
    counts = df["match_status"].value_counts().to_dict()
    log.info("name matching: %s", counts)
    write_csv(df, paths.name_matches)
    log.info("NEXT: review %s and save as %s", paths.name_matches, paths.reviewed_matches)
    return df


def load_accepted_matches(paths: Paths, use_auto: bool = False) -> pd.DataFrame:
    """Return accepted distressed firms, one row per firm_id."""
    if paths.reviewed_matches.exists():
        df = pd.read_csv(paths.reviewed_matches, dtype={"bse_code": "string"})
        src = paths.reviewed_matches
    elif use_auto:
        df = pd.read_csv(paths.name_matches, dtype={"bse_code": "string"})
        src = paths.name_matches
        log.warning("Using AUTO-accepted matches only (no human review yet).")
    else:
        raise FileNotFoundError(
            f"{paths.reviewed_matches} not found. Review {paths.name_matches} first, "
            "or pass --use-auto-matches for a quick unreviewed run.")
    df = df[df["accept"].astype(str).str.strip().str.upper() == "Y"].copy()
    df["admission_date"] = pd.to_datetime(df.get("admission_date_verified"), errors="coerce")
    df["admission_date_source"] = df["admission_date"].notna().map({True: "verified", False: "ibbi_pa_date"})
    df["admission_date"] = df["admission_date"].fillna(pd.to_datetime(df["cirp_announcement_date"]))
    df["petition_date"] = pd.to_datetime(df.get("petition_date"), errors="coerce")
    df = df.sort_values("admission_date").drop_duplicates("firm_id", keep="first")
    log.info("accepted distressed firms from %s: %d", src.name, len(df))
    return df.reset_index(drop=True)
