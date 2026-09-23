"""Match each distressed firm to a healthy peer, then build the firm-year frame.

Matching rules (from the project plan)
--------------------------------------
For distressed firm D admitted on date A, take its reference fiscal year: the
latest FY that ended before A and has financials. A healthy peer H must:

1. never appear in the IBBI CIRP list (any accepted or auto match is excluded);
2. have the same industry code (fallback: same first N characters of the code);
3. have total assets in the reference FY within +/- ``asset_tolerance`` of D;
4. have financials for at least ``min_financial_years`` FYs up to the reference FY;
5. not already be used as a peer for another distressed firm.

Among valid candidates we pick the closest in log(total assets). Firms with the
fewest candidates are matched first, so scarce peers are not used up early.

Input file (filled by the team): data/manual/firm_financials.csv
    firm_id, company_name, fy, industry_code, total_assets [, business_group, ...]
Get it from CMIE Prowess (export), or from annual reports / XBRL if no access.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from bpp.common import fy_end_date, fy_of_date, write_csv
from bpp.config import Paths

log = logging.getLogger(__name__)

FIN_REQUIRED = ["firm_id", "fy", "total_assets"]


def reference_fy(admission_date: pd.Timestamp, fy_cfg: dict[str, int]) -> int:
    """Latest FY that ended strictly before the admission date."""
    fy = fy_of_date(admission_date, fy_cfg["end_month"])
    if fy_end_date(fy, fy_cfg["end_month"], fy_cfg["end_day"]) >= admission_date:
        fy -= 1
    return fy


def match_peers(distressed: pd.DataFrame, financials: pd.DataFrame, cfg: dict[str, Any],
                exclude_firm_ids: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cohort, unmatched). cohort has one row per firm (distressed + healthy)."""
    ccfg, fy_cfg = cfg["cohort"], cfg["fiscal_year"]
    ind_col = ccfg["industry_column"]
    tol = ccfg["asset_tolerance"]
    min_years = ccfg["min_financial_years"]
    k = ccfg["peers_per_distressed"]
    prefix_len = ccfg["industry_fallback_prefix_len"]
    rng = np.random.default_rng(ccfg["random_seed"])

    fin = financials.copy()
    fin["fy"] = fin["fy"].astype(int)
    fin[ind_col] = fin[ind_col].astype(str).str.strip()
    fin = fin[fin["total_assets"].notna() & (fin["total_assets"] > 0)]

    distressed = distressed.copy()
    earliest = pd.Timestamp(str(ccfg["earliest_admission"])) if ccfg.get("earliest_admission") else None
    latest = pd.Timestamp(str(ccfg["latest_admission"])) if ccfg.get("latest_admission") else None
    if earliest is not None:
        distressed = distressed[distressed["admission_date"] >= earliest]
    if latest is not None:
        distressed = distressed[distressed["admission_date"] <= latest]

    excluded = set(exclude_firm_ids or set()) | set(distressed["firm_id"])
    healthy_pool = fin[~fin["firm_id"].isin(excluded)]
    years_per_firm = healthy_pool.groupby("firm_id")["fy"].apply(set).to_dict()

    # ---- 1. candidate lists per distressed firm
    plans, unmatched = [], []
    for _, d in distressed.iterrows():
        dfin = fin[fin["firm_id"] == d["firm_id"]]
        ref = reference_fy(d["admission_date"], fy_cfg)
        prior = dfin[dfin["fy"] <= ref].sort_values("fy")
        if len(prior) < min_years:
            unmatched.append({**d.to_dict(), "reason": f"only {len(prior)} FYs of financials before admission"})
            continue
        ref_row = prior.iloc[-1]
        ref_fy = int(ref_row["fy"])
        needed_years = set(prior["fy"].iloc[-min_years:])
        assets = float(ref_row["total_assets"])
        industry = str(ref_row[ind_col])

        same_year = healthy_pool[healthy_pool["fy"] == ref_fy].copy()
        same_year = same_year[same_year["firm_id"].map(lambda f: needed_years <= years_per_firm.get(f, set()))]
        same_year = same_year[(same_year["total_assets"] >= assets * (1 - tol)) &
                              (same_year["total_assets"] <= assets * (1 + tol))]
        exact = same_year[same_year[ind_col] == industry]
        fallback = same_year[same_year[ind_col].str[:prefix_len] == industry[:prefix_len]]
        if len(exact):
            cands, quality = exact, "exact_industry"
        elif len(fallback):
            cands, quality = fallback, f"industry_prefix_{prefix_len}"
        else:
            unmatched.append({**d.to_dict(), "reason": "no peer in industry within asset tolerance"})
            continue
        cands = cands.assign(dist=np.abs(np.log(cands["total_assets"] / assets)))
        cands = cands.assign(tiebreak=rng.random(len(cands))).sort_values(["dist", "tiebreak"])
        plans.append({"d": d, "ref_fy": ref_fy, "assets": assets, "industry": industry,
                      "cands": cands, "quality": quality, "company_name": ref_row.get("company_name")})

    # ---- 2. greedy assignment, scarcest first
    plans.sort(key=lambda p: len(p["cands"]))
    used: set[str] = set()
    rows = []
    for pair_no, p in enumerate(plans, start=1):
        d = p["d"]
        avail = p["cands"][~p["cands"]["firm_id"].isin(used)].head(k)
        if avail.empty:
            unmatched.append({**d.to_dict(), "reason": "all candidate peers already used"})
            continue
        pair_id = f"P{pair_no:04d}"
        rows.append({
            "pair_id": pair_id, "firm_id": d["firm_id"],
            "company_name": d.get("matched_name") or p["company_name"],
            "role": "distressed", "label": 1,
            "reference_date": d["admission_date"].date(),
            "admission_date_source": d.get("admission_date_source", "ibbi_pa_date"),
            "petition_date": d["petition_date"].date() if pd.notna(d.get("petition_date")) else None,
            "reference_fy": p["ref_fy"], "industry_code": p["industry"],
            "total_assets_ref_fy": p["assets"], "asset_ratio_to_distressed": 1.0,
            "match_quality": p["quality"], "business_group": d.get("business_group"),
        })
        for _, h in avail.iterrows():
            used.add(h["firm_id"])
            rows.append({
                "pair_id": pair_id, "firm_id": h["firm_id"], "company_name": h.get("company_name"),
                "role": "healthy", "label": 0,
                "reference_date": d["admission_date"].date(),
                "admission_date_source": None, "petition_date": None,
                "reference_fy": p["ref_fy"], "industry_code": h[ind_col],
                "total_assets_ref_fy": float(h["total_assets"]),
                "asset_ratio_to_distressed": round(float(h["total_assets"]) / p["assets"], 3),
                "match_quality": p["quality"], "business_group": h.get("business_group"),
            })
    cohort = pd.DataFrame(rows)
    if not cohort.empty:
        cohort = cohort.sort_values(["pair_id", "label"], ascending=[True, False]).reset_index(drop=True)
    return cohort, pd.DataFrame(unmatched)


def build_sample_frame(cohort: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """One row per firm x fiscal year whose annual report we need.

    Takes ``n_candidate_years`` FYs ending before the pair's reference date
    (one more than we finally keep, because the most recent report is often
    published after the cut-off and gets excluded in ``assign-labels``).
    """
    fy_cfg, n = cfg["fiscal_year"], cfg["cohort"]["n_candidate_years"]
    lag = cfg["labels"]["assumed_publication_lag_days"]
    rows = []
    for _, r in cohort.iterrows():
        ref_date = pd.Timestamp(r["reference_date"])
        last_fy = reference_fy(ref_date, fy_cfg)
        for rank, fy in enumerate(range(last_fy, last_fy - n, -1), start=1):
            end = fy_end_date(fy, fy_cfg["end_month"], fy_cfg["end_day"])
            rows.append({
                "pair_id": r["pair_id"], "firm_id": r["firm_id"], "company_name": r["company_name"],
                "role": r["role"], "label": r["label"], "reference_date": ref_date.date(),
                "fy": fy, "fy_end_date": end.date(), "fy_rank_before_reference": rank,
                "expected_pub_date": (end + pd.Timedelta(days=lag)).date(),
            })
    return pd.DataFrame(rows)


def run_build_cohort(cfg: dict[str, Any], paths: Paths, use_auto: bool = False) -> pd.DataFrame:
    from bpp.cohort.name_match import load_accepted_matches

    distressed = load_accepted_matches(paths, use_auto=use_auto)
    fin = pd.read_csv(paths.financials, dtype={"industry_code": "string"})
    missing = [c for c in FIN_REQUIRED + [cfg["cohort"]["industry_column"]] if c not in fin.columns]
    if missing:
        raise ValueError(f"{paths.financials} is missing columns {missing}")

    # every firm that appears anywhere in the CIRP match file is excluded from the healthy pool
    # (conservative: possible matches count too, unless a reviewer explicitly marked them N)
    exclude: set[str] = set()
    if paths.name_matches.exists():
        m = pd.read_csv(paths.name_matches)
        exclude |= set(m.loc[m["match_status"].isin(["auto_accepted", "needs_review"]), "firm_id"].dropna())
    if paths.reviewed_matches.exists():
        r = pd.read_csv(paths.reviewed_matches)
        decision = r["accept"].astype(str).str.strip().str.upper()
        exclude -= set(r.loc[decision == "N", "firm_id"].dropna())
        exclude |= set(r.loc[decision == "Y", "firm_id"].dropna())

    cohort, unmatched = match_peers(distressed, fin, cfg, exclude)
    write_csv(cohort, paths.cohort)
    write_csv(unmatched, paths.processed / "cohort_unmatched.csv")
    frame = build_sample_frame(cohort, cfg)
    write_csv(frame, paths.sample_frame)
    if not cohort.empty:
        log.info("cohort: %d pairs, %d firms, %d firm-years to collect | match quality: %s",
                 cohort["pair_id"].nunique(), len(cohort), len(frame),
                 cohort.drop_duplicates("pair_id")["match_quality"].value_counts().to_dict())
    return cohort
