"""Build the cohort from exchange data (no Prowess): eligibility, sampling, peers.

This is Phase 1's ``peers.py`` logic fed from what the download agent collects
instead of a hand-filled ``firm_financials.csv``:

* **industry** - BSE's classification, as a hierarchical code
  ``IN`` + sector(2) + industry(2) + group(2) + sub-group(3). A peer must share
  the 11-character sub-group code; if none qualifies, the 8-character group
  (``industry_fallback_prefix_len`` = 8 for this code, agreed 2026-09-23);
* **total assets** - the standalone annual XBRL filing of the reference year
  (or the year before when the reference year was never filed - common for a
  firm about to enter insolvency);
* **reports available** - the BSE annual-report list with filing dates.

Everything else follows ``peers.py`` and the decisions log: never a firm that
appears in the CIRP match list; total assets within +/-30%; closest in log
assets; scarcest distressed firms matched first; each peer used once.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from bpp.cohort.peers import reference_fy

log = logging.getLogger(__name__)


#: Financial firms file Schedule III Division III statements (no current/non-current
#: split), so most of the 12 ratios do not exist for them; standard in Altman-type studies.
EXCLUDED_SECTORS = ("Financial Services",)
#: A CIRP public announcement against a government company is almost always stayed
#: (e.g. Container Corporation of India, 2019, on an operational creditor's claim).
EXCLUDED_CIN_TYPES = ("central_govt", "state_govt")
#: BSE group A = the largest, most liquid companies. An admission against one is
#: usually an operational creditor's disputed claim stayed within days (Zee
#: Entertainment, Feb 2023; Titagarh Wagons, Nov 2022), not insolvency.
#: Provisional rule for the pilot - to be confirmed by the team for the full run.
EXCLUDED_BSE_GROUPS = ("A",)


def distressed_eligibility(matches: pd.DataFrame, headers: pd.DataFrame, xbrl_index: pd.DataFrame,
                           reports: pd.DataFrame, cfg: dict[str, Any],
                           statuses: tuple[str, ...] = ("auto_accepted",),
                           excluded_sectors: tuple[str, ...] = EXCLUDED_SECTORS,
                           excluded_cin_types: tuple[str, ...] = EXCLUDED_CIN_TYPES,
                           excluded_groups: tuple[str, ...] = EXCLUDED_BSE_GROUPS) -> pd.DataFrame:
    """One row per matched distressed firm with what the cohort needs, and why not if not."""
    fy_cfg = cfg["fiscal_year"]
    n_cand = cfg["cohort"]["n_candidate_years"]
    min_years = cfg["cohort"]["min_financial_years"]
    gap_days = cfg["labels"]["exclude_within_days_of_admission"]

    m = matches[matches["match_status"].isin(statuses) & matches["bse_code"].notna()].copy()
    m["bse_code"] = m["bse_code"].astype(str).str.replace(r"\.0$", "", regex=True)
    m["admission_date"] = pd.to_datetime(m["cirp_announcement_date"])
    m = m.sort_values("admission_date").drop_duplicates("bse_code")

    hdr = headers.set_index("bse_code") if len(headers) else pd.DataFrame()
    xb = xbrl_index.dropna(subset=["xbrl_url"]).set_index(["bse_code", "fy"]) if len(xbrl_index) else None
    rep = reports.dropna(subset=["fy"]).copy() if len(reports) else reports
    if len(rep):
        # BSE rarely records a filing time before ~FY2023. The project rule for an
        # unknown publication date applies: FY end + assumed lag, flagged.
        lag = pd.Timedelta(days=cfg["labels"]["assumed_publication_lag_days"])
        fy_end = pd.to_datetime(rep["fy"].astype(int).astype(str) + f"-{fy_cfg['end_month']:02d}-{fy_cfg['end_day']:02d}")
        rep["pub_date_source"] = np.where(rep["filed_at"].notna(), "bse_filing", "assumed")
        rep["filed_at"] = rep["filed_at"].fillna(fy_end + lag)

    rows = []
    for _, d in m.iterrows():
        code, adm = d["bse_code"], d["admission_date"]
        ref = reference_fy(adm, fy_cfg)
        years = list(range(ref, ref - n_cand, -1))
        h = hdr.loc[code] if code in hdr.index else None
        sub = h.get("isubgroup_code") if h is not None else None
        r = rep[(rep["bse_code"] == code) & rep["fy"].isin(years)] if len(rep) else rep
        before = r[r["filed_at"] < adm] if len(r) else r
        well_before = r[r["filed_at"] < adm - pd.Timedelta(days=gap_days)] if len(r) else r
        size_fy = next((fy for fy in (ref, ref - 1) if xb is not None and (code, fy) in xb.index), None)
        reasons = []
        sector = h.get("sector") if h is not None else None
        if sector in excluded_sectors:
            reasons.append("financial_sector")
        if d.get("cin_company_type") in excluded_cin_types:
            reasons.append("government_company")
        if h is not None and h.get("bse_group") in excluded_groups:
            reasons.append("bse_group_A_possible_technical_admission")
        if not isinstance(sub, str) or not sub:
            reasons.append("no_bse_industry")
        if len(before) < min_years:
            reasons.append(f"only_{len(before)}_reports_before_admission")
        if len(well_before) < 2:
            reasons.append("fewer_than_2_reports_outside_leakage_window")
        if size_fy is None:
            reasons.append("no_xbrl_for_size")
        rows.append({
            "bse_code": code, "firm_id": f"BSE{code}", "company_name": d.get("matched_name"),
            "corporate_debtor": d["corporate_debtor"], "cin": d.get("cin"),
            "match_status": d["match_status"], "score": d.get("score"),
            "admission_date": adm.date(), "admission_year": adm.year, "reference_fy": ref,
            "isubgroup_code": sub, "isubgroup": h.get("isubgroup") if h is not None else None,
            "sector": sector, "cin_company_type": d.get("cin_company_type"),
            "bse_group": h.get("bse_group") if h is not None else None,
            "n_reports_candidate_years": int(r["fy"].nunique()) if len(r) else 0,
            "n_reports_before_admission": int(before["fy"].nunique()) if len(before) else 0,
            "n_reports_outside_window": int(well_before["fy"].nunique()) if len(well_before) else 0,
            "size_fy": size_fy,
            "size_xbrl_url": xb.loc[(code, size_fy), "xbrl_url"] if size_fy is not None else None,
            "eligible": not reasons, "ineligible_reason": ";".join(reasons),
        })
    return pd.DataFrame(rows)


def sample_pilot(elig: pd.DataFrame, n: int, seed: int, years: tuple[int, int]) -> pd.DataFrame:
    """Stratified by admission year, one industry sub-group at most twice."""
    pool = elig[elig["eligible"] & elig["admission_year"].between(*years)].copy()
    rng = np.random.default_rng(seed)
    pool["r"] = rng.random(len(pool))
    per_year = {y: g.sort_values("r") for y, g in pool.groupby("admission_year")}
    chosen, used_sub = [], {}
    while len(chosen) < n and any(len(g) for g in per_year.values()):
        for y in sorted(per_year):
            g = per_year[y]
            while len(g):
                row = g.iloc[0]
                g = g.iloc[1:]
                if used_sub.get(row["isubgroup_code"], 0) < 2:
                    chosen.append(row)
                    used_sub[row["isubgroup_code"]] = used_sub.get(row["isubgroup_code"], 0) + 1
                    break
            per_year[y] = g
            if len(chosen) >= n:
                break
    return pd.DataFrame(chosen).drop(columns="r").reset_index(drop=True)


def other_listings(code: str, isin_by_code: dict[str, str], name_by_code: dict[str, str]) -> set[str]:
    """Codes that are not a separate company from ``code``: its own other listings.

    A DVR or partly paid-up share class has its own BSE code and an ``IN9`` ISIN
    (Future Enterprises' DVR shares, BSE570002, were picked as the insolvent firm's
    own "healthy" peer on the full cohort), and a company can appear under two codes
    with the same name. Neither is a peer.
    """
    name = name_by_code.get(code)
    same = {c for c, n in name_by_code.items() if name and n == name and c != code}
    return same | {c for c, i in isin_by_code.items() if isinstance(i, str) and i.upper().startswith("IN9")}


def correct_unit_slips(assets: dict[tuple[str, int], float], tol: float = 0.15,
                       powers: tuple[int, ...] = (2, 3, 5, 7)) -> tuple[dict[tuple[str, int], float], pd.DataFrame]:
    """Total assets from XBRL filings, with filings a power of ten off the firm's other years put right.

    Companies type their XBRL by hand and some get the scale wrong: MAX ALERT filed
    FY2021 total assets of Rs 22.8 lakh crore (its report: Rs 22.82 crore), and Rane
    (Madras) filed FY2022-23 100x too small. Sizing on such a filing leaves a firm
    without a peer or pairs it with a firm a hundred times its size.

    A firm's filings are grouped by order of magnitude (a new group starts where two
    neighbouring values are more than 10x apart). Only when one group holds a strict
    majority of the filings (at least two) is it taken as the firm's scale; a filing
    outside it is corrected when it sits a power of ten (10^2, 10^3, 10^5, 10^7) from
    the group's median, within ``tol`` in log10, and lands inside the group's range
    (widened by 2x). A firm whose filings split evenly between two scales, or with a
    single filing, is left as it is.
    """
    by_firm: dict[str, dict[int, float]] = {}
    for (code, fy), v in assets.items():
        if v and v > 0:
            by_firm.setdefault(code, {})[int(fy)] = float(v)
    out = dict(assets)
    rows = []
    for code, years in by_firm.items():
        if len(years) < 3:
            continue
        logs = sorted((float(np.log10(v)), fy) for fy, v in years.items())
        groups, cur = [], [logs[0]]
        for a, b in zip(logs, logs[1:]):
            if b[0] - a[0] > 1.0:
                groups.append(cur)
                cur = []
            cur.append(b)
        groups.append(cur)
        main = max(groups, key=len)
        if len(main) < 2 or len(main) * 2 <= len(logs):
            continue
        centre = float(np.median([x for x, _ in main]))
        lo, hi = main[0][0] - np.log10(2), main[-1][0] + np.log10(2)
        members = {fy for _, fy in main}
        for fy, v in years.items():
            if fy in members:
                continue
            k = float(np.log10(v)) - centre
            for p in powers:
                for sign in (1, -1):
                    fixed_log = float(np.log10(v)) - sign * p
                    if abs(k - sign * p) < tol and lo <= fixed_log <= hi:
                        out[(code, fy)] = v / 10 ** (sign * p)
                        rows.append({"bse_code": code, "fy": fy, "filed_value_cr": v, "corrected_cr": v / 10 ** (sign * p),
                                     "median_other_years_cr": 10 ** centre, "power": sign * p})
    return out, pd.DataFrame(rows, columns=["bse_code", "fy", "filed_value_cr", "corrected_cr",
                                            "median_other_years_cr", "power"])


def peer_candidates(pilot: pd.DataFrame, members: pd.DataFrame, headers: pd.DataFrame,
                    excluded_codes: set[str], prefix_len: int = 8,
                    isin_by_code: dict[str, str] | None = None,
                    name_by_code: dict[str, str] | None = None) -> pd.DataFrame:
    """All possible peers per distressed firm: same sub-group (else same group), not excluded,
    and never another listing of the firm itself."""
    pool = []
    if len(members):
        pool.append(members[["bse_code", "isubgroup_code"]])
    if len(headers) and "isubgroup_code" in headers:
        pool.append(headers.dropna(subset=["isubgroup_code"])[["bse_code", "isubgroup_code"]])
    pool = pd.concat(pool, ignore_index=True).drop_duplicates("bse_code") if pool else pd.DataFrame(
        columns=["bse_code", "isubgroup_code"])
    pool = pool[~pool["bse_code"].isin(excluded_codes)]
    rows = []
    for _, d in pilot.iterrows():
        exact = pool[pool["isubgroup_code"] == d["isubgroup_code"]]
        quality = "exact_industry"
        if exact.empty:
            exact = pool[pool["isubgroup_code"].str[:prefix_len] == str(d["isubgroup_code"])[:prefix_len]]
            quality = f"industry_prefix_{prefix_len}"
        itself = other_listings(d["bse_code"], isin_by_code or {}, name_by_code or {})
        for code in exact["bse_code"]:
            if code != d["bse_code"] and code not in itself:
                rows.append({"distressed_code": d["bse_code"], "candidate_code": code,
                             "reference_fy": d["reference_fy"], "match_quality": quality})
    return pd.DataFrame(rows)


def choose_peers(pilot: pd.DataFrame, cands: pd.DataFrame, assets: dict[tuple[str, int], float],
                 report_counts: dict[tuple[str, int], int] | None, cfg: dict[str, Any], seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedy 1:1 matching on log total assets, scarcest first (peers.py rules)."""
    tol = cfg["cohort"]["asset_tolerance"]
    min_years = cfg["cohort"]["min_financial_years"]
    rng = np.random.default_rng(seed)
    plans, unmatched = [], []
    for _, d in pilot.iterrows():
        a = assets.get((d["bse_code"], int(d["size_fy"]))) if pd.notna(d.get("size_fy")) else None
        if not a or a <= 0:
            unmatched.append({**d.to_dict(), "reason": "distressed firm has no total assets"})
            continue
        c = cands[cands["distressed_code"] == d["bse_code"]].copy()
        c["assets"] = [assets.get((code, int(d["size_fy"]))) for code in c["candidate_code"]]
        c = c[c["assets"].notna() & (c["assets"] > 0)]
        c = c[(c["assets"] >= a * (1 - tol)) & (c["assets"] <= a * (1 + tol))]
        if report_counts is not None:   # reports for the pair's candidate years, when already known
            ref = int(d["reference_fy"])
            c = c[c["candidate_code"].map(lambda k: report_counts.get((k, ref), 0) >= min_years)]
        if c.empty:
            unmatched.append({**d.to_dict(), "reason": "no peer in industry within asset tolerance"})
            continue
        c["dist"] = np.abs(np.log(c["assets"] / a))
        c["tiebreak"] = rng.random(len(c))
        plans.append((d, a, c.sort_values(["dist", "tiebreak"])))
    plans.sort(key=lambda p: len(p[2]))
    used: set[str] = set()
    rows = []
    for d, a, c in plans:
        avail = c[~c["candidate_code"].isin(used)]
        if avail.empty:
            unmatched.append({**d.to_dict(), "reason": "all candidate peers already used"})
            continue
        h = avail.iloc[0]
        used.add(h["candidate_code"])
        rows.append({"distressed_code": d["bse_code"], "peer_code": h["candidate_code"],
                     "reference_fy": int(d["reference_fy"]), "size_fy": int(d["size_fy"]),
                     "distressed_assets": a, "peer_assets": float(h["assets"]),
                     "asset_ratio": round(float(h["assets"]) / a, 3), "match_quality": h["match_quality"],
                     "n_candidates": len(c)})
    return pd.DataFrame(rows), pd.DataFrame(unmatched)
