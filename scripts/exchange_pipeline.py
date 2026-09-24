"""Build the cohort and the document set from exchange data collected by the download agent.

Run from the repo root with the data folder that mirrors the laptop's
``bpp-data`` (``--data-dir``). Each step reads what the agent collected under
``raw/`` and writes interim tables and, where more data is needed, the next job
file for the agent (``--jobs-out``).

    python scripts/exchange_pipeline.py --data-dir D parse
    python scripts/exchange_pipeline.py --data-dir D select --n 40 --jobs-out J
    python scripts/exchange_pipeline.py --data-dir D peers --jobs-out J
    python scripts/exchange_pipeline.py --data-dir D finalize --n-pairs 30 --jobs-out J
    python scripts/exchange_pipeline.py --data-dir D xbrl
    python scripts/exchange_pipeline.py --data-dir D manifest

Full cohort (the pilot's pairs kept as they are), one step per download round:

    full-select -> full-candidates -> full-sizing -> full-shortlist -> full-finalize
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bpp.cohort.cin_match import match_debtors  # noqa: E402
from bpp.cohort.peers import build_sample_frame  # noqa: E402
from bpp.cohort.pilot import choose_peers, distressed_eligibility, peer_candidates, sample_pilot  # noqa: E402
from bpp.common import doc_id_for  # noqa: E402
from bpp.scrape.listed import build_universe, parse_bse_scrips, parse_nse_equity_list  # noqa: E402
from bpp.sources import jobs as J  # noqa: E402
from bpp.sources.bse import (annual_standalone_xbrl, iter_jsonl, parse_annual_report_list,  # noqa: E402
                             parse_company_headers, parse_industry_list, parse_result_archive,
                             parse_traded_members)
from bpp.sources.ibbi_export import cirp_debtors, possibly_listed, read_ibbi_export  # noqa: E402
from bpp.sources.xbrl_results import parse_results_xbrl, result_to_rows  # noqa: E402

log = logging.getLogger("exchange_pipeline")


def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))


def code_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True)


# ------------------------------------------------------------------ loaders
def load_tables(D: Path) -> dict[str, pd.DataFrame]:
    raw = D / "raw"
    ind = parse_industry_list(raw / "listed/bse_industry_list.json") if (raw / "listed/bse_industry_list.json").exists() else None
    t = {
        "industry_list": ind if ind is not None else pd.DataFrame(columns=["isubgroup", "isubgroup_code"]),
        "headers": parse_company_headers(raw / "listed/bse_company_header.jsonl", ind),
        "archive": parse_result_archive(raw / "xbrl/_listings/bse_result_archive.jsonl"),
        "reports": parse_annual_report_list(raw / "annual_reports/_listings/bse_annual_reports.jsonl"),
        "members": parse_traded_members(raw / "listed/bse_industry_members_traded.jsonl"),
    }
    t["xbrl_index"] = annual_standalone_xbrl(t["archive"]) if len(t["archive"]) else pd.DataFrame(
        columns=["bse_code", "fy", "xbrl_url", "filed_at", "revised_at"])
    return t


def xbrl_assets(D: Path) -> tuple[dict[tuple[str, int], float], pd.DataFrame]:
    """(bse_code, fy) -> total assets (Rs crore) from every collected XBRL filing, plus all rows."""
    out, rows = {}, []
    for rec in iter_jsonl(D / "raw/xbrl/results_standalone_annual.jsonl"):
        if rec.get("http") != 200 or not rec.get("body"):
            continue
        meta = rec.get("meta") or {}
        try:
            res = parse_results_xbrl(rec["body"])
        except Exception as exc:  # noqa: BLE001
            log.warning("unreadable XBRL %s: %s", rec.get("key"), exc)
            continue
        firm_id = meta.get("firm_id") or str(rec["key"]).rsplit("_FY", 1)[0]
        code = firm_id.replace("BSE", "")
        fy = res.fy or meta.get("fy")
        if res.meta.get("standalone") is False:
            log.warning("%s is a consolidated filing - skipped", rec.get("key"))
            continue
        if res.values_cr.get("total_assets"):
            out[(code, int(fy))] = res.values_cr["total_assets"]
        for r in result_to_rows(res, firm_id, rec.get("url", "")):
            r["fy"] = int(fy)
            r["filed_meta_fy"] = meta.get("fy")
            r["audit_modified"] = res.meta.get("audit_modified")
            rows.append(r)
    return out, pd.DataFrame(rows)


# ------------------------------------------------------------------ steps
def step_parse(D: Path, cfg: dict) -> None:
    ann = read_ibbi_export(D / "raw/ibbi/ibbi_cirp_export.tsv")
    (D / "interim").mkdir(parents=True, exist_ok=True)
    ann.drop(columns=["ip_address"], errors="ignore").to_csv(D / "raw/ibbi/public_announcements_export.csv", index=False)
    deb = cirp_debtors(ann)
    deb.to_csv(D / "interim/ibbi_cirp_debtors.csv", index=False)
    frames = []
    for st in ("active", "suspended", "delisted"):
        recs = json.loads((D / f"raw/listed/bse_scrips_{st}.json").read_text(encoding="utf-8"))
        recs = recs if isinstance(recs, list) else recs.get("Table", [])
        frames.append(parse_bse_scrips(recs, st.capitalize()))
    bse = pd.concat(frames, ignore_index=True)
    nse = parse_nse_equity_list((D / "raw/listed/nse_equity_list.csv").read_text(encoding="utf-8"))
    uni = build_universe(bse, nse)
    uni.to_csv(D / "interim/listed_universe.csv", index=False)
    m = match_debtors(possibly_listed(deb), uni, cfg["name_matching"]["auto_accept_score"],
                      cfg["name_matching"]["review_score"], cfg["name_matching"]["top_k_candidates"])
    m.to_csv(D / "interim/ibbi_listed_matches.csv", index=False)
    log.info("debtors %d | universe %d | matches: %s", len(deb), len(uni), m["match_status"].value_counts().to_dict())


def step_select(D: Path, cfg: dict, n: int, years: tuple[int, int], jobs_out: Path) -> None:
    t = load_tables(D)
    m = pd.read_csv(D / "interim/ibbi_listed_matches.csv", dtype={"bse_code": "string"})
    elig = distressed_eligibility(m, t["headers"], t["xbrl_index"], t["reports"], cfg)
    elig.to_csv(D / "interim/distressed_eligibility.csv", index=False)
    log.info("eligible distressed firms: %d of %d", int(elig["eligible"].sum()), len(elig))
    log.info("by admission year:\n%s", elig.groupby("admission_year")["eligible"].agg(["size", "sum"]).to_string())
    pilot = sample_pilot(elig, n, cfg["cohort"]["random_seed"], years)
    pilot.to_csv(D / "interim/pilot_distressed.csv", index=False)
    log.info("pilot sample: %d firms, years %s", len(pilot), pilot["admission_year"].value_counts().sort_index().to_dict())

    items = [J.xbrl_file(r["firm_id"], int(r["size_fy"]), r["size_xbrl_url"]) for _, r in pilot.iterrows()]
    path = J.write_job(jobs_out, "005_pilot_distressed_xbrl", items,
                       "Pilot: standalone annual XBRL of the sampled insolvent firms for their size year.", delay_s=0.4)
    log.info("wrote %s with %d items", path, len(items))


def step_candidates(D: Path, cfg: dict, jobs_out: Path) -> None:
    """Candidate peers per pilot firm; the job for their results archives and report lists."""
    t = load_tables(D)
    m = pd.read_csv(D / "interim/ibbi_listed_matches.csv", dtype={"bse_code": "string"})
    pilot = pd.read_csv(D / "interim/pilot_distressed.csv", dtype={"bse_code": "string", "isubgroup_code": "string"})
    # never a peer: anything in the CIRP match list, auto or review (decisions log, 16 Sep)
    excluded = set(code_str(m.loc[m["match_status"].isin(["auto_accepted", "needs_review"]), "bse_code"].dropna()))
    cands = peer_candidates(pilot, t["members"], t["headers"], excluded, prefix_len=8)
    cands.to_csv(D / "interim/pilot_peer_candidates.csv", index=False)
    items = []
    have = set(t["archive"]["bse_code"]) if len(t["archive"]) else set()
    for code in sorted(set(cands["candidate_code"]) - have):
        items.append(J.bse_result_archive(code))
    for code in sorted(set(cands["candidate_code"]) - set(t["reports"]["bse_code"] if len(t["reports"]) else [])):
        items.append(J.bse_annual_reports(code))
    path = J.write_job(jobs_out, "006_pilot_candidates", items,
                       "Pilot: results archive and annual-report list of every possible peer in the pilot "
                       "firms' industries.", delay_s=0.4)
    log.info("candidates: %d (distressed, candidate) pairs, %d distinct firms; wrote %s with %d items",
             len(cands), cands["candidate_code"].nunique() if len(cands) else 0, path, len(items))


def step_peers(D: Path, cfg: dict, jobs_out: Path) -> None:
    """Second pass: XBRL of every candidate for the pair's size year."""
    t = load_tables(D)
    pilot = pd.read_csv(D / "interim/pilot_distressed.csv", dtype={"bse_code": "string", "isubgroup_code": "string"})
    cands = pd.read_csv(D / "interim/pilot_peer_candidates.csv", dtype={"distressed_code": "string", "candidate_code": "string"})
    xb = t["xbrl_index"].dropna(subset=["xbrl_url"]).set_index(["bse_code", "fy"])
    items, missing = [], 0
    for _, d in pilot.iterrows():
        fy = int(d["size_fy"])
        for code in cands.loc[cands["distressed_code"] == d["bse_code"], "candidate_code"]:
            if (code, fy) in xb.index:
                items.append(J.xbrl_file(f"BSE{code}", fy, xb.loc[(code, fy), "xbrl_url"]))
            else:
                missing += 1
    uniq = {i["key"]: i for i in items}
    path = J.write_job(jobs_out, "007_pilot_candidate_xbrl", uniq.values(),
                       "Pilot: standalone annual XBRL of every possible peer for its pair's size year.", delay_s=0.4)
    log.info("candidate XBRL files: %d (no XBRL for %d candidate-years); wrote %s", len(uniq), missing, path)


def step_finalize(D: Path, cfg: dict, n_pairs: int, jobs_out: Path) -> None:
    t = load_tables(D)
    assets, _ = xbrl_assets(D)
    pilot = pd.read_csv(D / "interim/pilot_distressed.csv", dtype={"bse_code": "string", "isubgroup_code": "string"})
    cands = pd.read_csv(D / "interim/pilot_peer_candidates.csv", dtype={"distressed_code": "string", "candidate_code": "string"})
    rep = t["reports"]
    n_cand = cfg["cohort"]["n_candidate_years"]
    counts = {}
    for _, d in pilot.iterrows():
        ref = int(d["reference_fy"])
        yrs = set(range(ref - n_cand + 1, ref + 1))
        for code in cands.loc[cands["distressed_code"] == d["bse_code"], "candidate_code"]:
            counts[(code, ref)] = int(rep[(rep["bse_code"] == code) & rep["fy"].isin(yrs)]["fy"].nunique())
    pairs, unmatched = choose_peers(pilot, cands, assets, counts, cfg, cfg["cohort"]["random_seed"])
    pairs = pairs.merge(pilot[["bse_code", "admission_date", "admission_year"]].rename(columns={"bse_code": "distressed_code"}),
                        on="distressed_code", how="left")
    # keep the year stratification when more pairs matched than wanted: round-robin over years,
    # in the sample's own random order (pilot rows were drawn in that order)
    order = {c: i for i, c in enumerate(pilot["bse_code"])}
    pairs["draw"] = pairs["distressed_code"].map(order)
    by_year = {y: g.sort_values("draw") for y, g in pairs.groupby("admission_year")}
    keep = []
    while len(keep) < n_pairs and any(len(g) for g in by_year.values()):
        for y in sorted(by_year):
            if len(by_year[y]) and len(keep) < n_pairs:
                keep.append(by_year[y].iloc[0])
                by_year[y] = by_year[y].iloc[1:]
    pairs = pd.DataFrame(keep).drop(columns="draw").sort_values(["admission_year", "distressed_code"]).reset_index(drop=True)
    pairs.to_csv(D / "interim/pilot_pairs.csv", index=False)
    unmatched.to_csv(D / "interim/pilot_unmatched.csv", index=False)
    log.info("pairs: %d (unmatched %d: %s)", len(pairs), len(unmatched),
             unmatched["reason"].value_counts().to_dict() if len(unmatched) else {})

    # cohort.csv + sample_frame.csv in the Phase 1 layout
    uni = pd.read_csv(D / "interim/listed_universe.csv", dtype={"bse_code": "string"})
    names = dict(zip(code_str(uni["bse_code"].dropna()), uni.loc[uni["bse_code"].notna(), "company_name"]))
    hdr = t["headers"].set_index("bse_code")
    pil = pilot.set_index("bse_code")
    rows = []
    for i, p in pairs.iterrows():
        pid = f"P{i + 1:04d}"
        d = pil.loc[p["distressed_code"]]
        for role, code, assets_v in (("distressed", p["distressed_code"], p["distressed_assets"]),
                                     ("healthy", p["peer_code"], p["peer_assets"])):
            rows.append({
                "pair_id": pid, "firm_id": f"BSE{code}", "company_name": names.get(code),
                "role": role, "label": 1 if role == "distressed" else 0,
                "reference_date": d["admission_date"],
                "admission_date_source": "ibbi_pa_date" if role == "distressed" else None,
                "petition_date": None, "reference_fy": int(p["reference_fy"]),
                "industry_code": hdr["isubgroup_code"].get(code) if code in hdr.index else d["isubgroup_code"],
                "total_assets_ref_fy": round(float(assets_v), 2),
                "asset_ratio_to_distressed": 1.0 if role == "distressed" else p["asset_ratio"],
                "match_quality": p["match_quality"], "business_group": None,
                "cin": d["cin"] if role == "distressed" else None,
                "size_fy": int(p["size_fy"]),
            })
    cohort = pd.DataFrame(rows)
    (D / "processed").mkdir(parents=True, exist_ok=True)
    cohort.to_csv(D / "processed/cohort.csv", index=False)
    frame = build_sample_frame(cohort, cfg)
    frame.to_csv(D / "processed/sample_frame.csv", index=False)
    log.info("cohort: %d firms, sample frame: %d firm-years", len(cohort), len(frame))

    # next job: reports + XBRL for every firm-year in the frame
    have_rep = rep.set_index(["bse_code", "fy"]) if len(rep) else None
    xb = t["xbrl_index"].dropna(subset=["xbrl_url"]).set_index(["bse_code", "fy"])
    items, missing = [], []
    for _, r in frame.iterrows():
        code, fy = r["firm_id"].replace("BSE", ""), int(r["fy"])
        if have_rep is not None and (code, fy) in have_rep.index:
            url = have_rep.loc[(code, fy)]
            url = url.iloc[0]["url"] if isinstance(url, pd.DataFrame) else url["url"]
            items.append(J.report_pdf(r["firm_id"], fy, url))
        else:
            missing.append((r["firm_id"], fy))
        if (code, fy) in xb.index:
            items.append(J.xbrl_file(r["firm_id"], fy, xb.loc[(code, fy), "xbrl_url"]))
    # the year after the last one too: its report's prior-year column cross-checks
    uniq = list({(i.get("path") or i["key"]): i for i in items}.values())
    path = J.write_job(jobs_out, "008_pilot_reports", uniq,
                       "Pilot: annual-report PDFs and standalone XBRL results for every firm-year of the 30 pairs.",
                       delay_s=1.0)
    log.info("report job: %d items (%d PDFs); no report listed for %d firm-years; wrote %s",
             len(uniq), sum(1 for i in uniq if i.get("expect") == "pdf"), len(missing), path)


# ------------------------------------------------------------------ full cohort (approved 24 Sep 2026)
# The pilot's 27 pairs stay exactly as they are (same pair ids, same peers); the full
# run adds every other eligible insolvent firm. Peers come from the full BSE industry
# classification (job 010) instead of the traded-members list.
def _collected(D: Path, relpath: str) -> set[str]:
    """Keys already collected (any part), so a job never asks for them again."""
    return {str(r.get("key")) for r in iter_jsonl(D / relpath) if r.get("http") == 200 and r.get("body")}


def _new_items(D: Path, items: list[dict]) -> list[dict]:
    seen: dict[str, set[str]] = {}
    out = []
    for it in items:
        if it.get("kind") == "collect":
            base = it["collect"]
            if base not in seen:
                seen[base] = _collected(D, base)
            if str(it["key"]) in seen[base]:
                continue
        elif it.get("path") and (D / it["path"]).exists():
            continue
        out.append(it)
    return out


def _pilot_pairs(D: Path) -> pd.DataFrame:
    f = D / "interim/pilot_pairs.csv"
    return pd.read_csv(f, dtype={"distressed_code": "string", "peer_code": "string"}) if f.exists() else \
        pd.DataFrame(columns=["distressed_code", "peer_code"])


def step_full_select(D: Path, cfg: dict, jobs_out: Path) -> None:
    t = load_tables(D)
    m = pd.read_csv(D / "interim/ibbi_listed_matches.csv", dtype={"bse_code": "string"})
    elig = distressed_eligibility(m, t["headers"], t["xbrl_index"], t["reports"], cfg)
    elig.to_csv(D / "interim/distressed_eligibility.csv", index=False)
    full = elig[elig["eligible"]].sort_values(["admission_date", "bse_code"]).reset_index(drop=True)
    full.to_csv(D / "interim/full_distressed.csv", index=False)
    pilot = set(_pilot_pairs(D)["distressed_code"])
    log.info("eligible insolvent firms: %d (%d already paired in the pilot)", len(full),
             int(full["bse_code"].isin(pilot).sum()))
    items = _new_items(D, [J.xbrl_file(r["firm_id"], int(r["size_fy"]), r["size_xbrl_url"]) for _, r in full.iterrows()])
    path = J.write_jobs(jobs_out, "011_full_distressed_xbrl", items,
                       "Full cohort: standalone annual XBRL of every eligible insolvent firm for its size year "
                       "(already collected ones are skipped).", delay_s=0.4)
    log.info("wrote %s with %d items", path, len(items))


def _full_new(D: Path) -> pd.DataFrame:
    full = pd.read_csv(D / "interim/full_distressed.csv", dtype={"bse_code": "string", "isubgroup_code": "string"})
    return full[~full["bse_code"].isin(set(_pilot_pairs(D)["distressed_code"]))].reset_index(drop=True)


def step_full_candidates(D: Path, cfg: dict, jobs_out: Path) -> None:
    t = load_tables(D)
    m = pd.read_csv(D / "interim/ibbi_listed_matches.csv", dtype={"bse_code": "string"})
    new = _full_new(D)
    excluded = set(code_str(m.loc[m["match_status"].isin(["auto_accepted", "needs_review"]), "bse_code"].dropna()))
    excluded |= set(_pilot_pairs(D)["peer_code"])               # a pilot peer is already used
    cands = peer_candidates(new, t["members"], t["headers"], excluded, prefix_len=8)
    cands.to_csv(D / "interim/full_peer_candidates.csv", index=False)
    have = set(t["archive"]["bse_code"]) if len(t["archive"]) else set()
    items = _new_items(D, [J.bse_result_archive(c) for c in sorted(set(cands["candidate_code"]) - have)])
    path = J.write_jobs(jobs_out, "012_full_candidate_archives", items,
                       "Full cohort: results archive (the list of filings) of every possible peer, to find "
                       "its standalone annual XBRL.", delay_s=0.4)
    log.info("%d insolvent firms to pair; %d (firm, candidate) pairs, %d distinct candidates; wrote %s with %d items",
             len(new), len(cands), cands["candidate_code"].nunique() if len(cands) else 0, path, len(items))


def step_full_sizing(D: Path, cfg: dict, jobs_out: Path) -> None:
    t = load_tables(D)
    new = _full_new(D)
    cands = pd.read_csv(D / "interim/full_peer_candidates.csv", dtype={"distressed_code": "string", "candidate_code": "string"})
    xb = t["xbrl_index"].dropna(subset=["xbrl_url"]).set_index(["bse_code", "fy"])
    items, missing = [], 0
    for _, d in new.iterrows():
        fy = int(d["size_fy"])
        for code in cands.loc[cands["distressed_code"] == d["bse_code"], "candidate_code"]:
            if (code, fy) in xb.index:
                items.append(J.xbrl_file(f"BSE{code}", fy, xb.loc[(code, fy), "xbrl_url"]))
            else:
                missing += 1
    uniq = {i["key"]: i for i in _new_items(D, items)}
    path = J.write_jobs(jobs_out, "013_full_candidate_xbrl", uniq.values(),
                       "Full cohort: standalone annual XBRL of every possible peer for its pair's size year.",
                       delay_s=0.4, per_job=1000)
    log.info("candidate XBRL files: %d (no XBRL for %d candidate-years); wrote %s", len(uniq), missing, path)


def _firm_year_items(t: dict, firm_years: list[tuple[str, int]]) -> tuple[list[dict], list[tuple[str, int]]]:
    """Report PDF and XBRL download items for (bse_code, fy) pairs, and those with no report listed."""
    rep = t["reports"]
    have_rep = rep.set_index(["bse_code", "fy"]).sort_index() if len(rep) else None
    xb = t["xbrl_index"].dropna(subset=["xbrl_url"]).set_index(["bse_code", "fy"])
    items, missing = [], []
    for code, fy in firm_years:
        firm_id = f"BSE{code}"
        if have_rep is not None and (code, fy) in have_rep.index:
            hit = have_rep.loc[(code, fy)]
            url = hit.iloc[0]["url"] if isinstance(hit, pd.DataFrame) else hit["url"]
            items.append(J.report_pdf(firm_id, fy, url))
        else:
            missing.append((firm_id, fy))
        if (code, fy) in xb.index:
            items.append(J.xbrl_file(firm_id, fy, xb.loc[(code, fy), "xbrl_url"]))
    return items, missing


def step_full_distressed_reports(D: Path, cfg: dict, jobs_out: Path) -> None:
    """The insolvent firms' own reports can come down while their peers are still being found.

    Written in small jobs so that a matching job written later (a lower number) runs
    between them instead of waiting for every report.
    """
    t = load_tables(D)
    new = _full_new(D)
    n_cand = cfg["cohort"]["n_candidate_years"]
    fys = [(r["bse_code"], fy) for _, r in new.iterrows()
           for fy in range(int(r["reference_fy"]) - n_cand + 1, int(r["reference_fy"]) + 1)]
    items, missing = _firm_year_items(t, fys)
    items = _new_items(D, list({(i.get("path") or i["key"]): i for i in items}.values()))
    paths = J.write_jobs(jobs_out, "019_full_distressed_reports", items,
                         "Full cohort: annual-report PDFs and XBRL of the insolvent firms' candidate years "
                         "(their peers' follow once matched).", delay_s=1.0, per_job=120)
    log.info("%d items (%d PDFs) in %d jobs; no report listed for %d firm-years",
             len(items), sum(1 for i in items if i.get("expect") == "pdf"), len(paths), len(missing))


def _in_tolerance(new: pd.DataFrame, cands: pd.DataFrame, assets: dict, tol: float) -> pd.DataFrame:
    rows = []
    for _, d in new.iterrows():
        a = assets.get((d["bse_code"], int(d["size_fy"])))
        if not a or a <= 0:
            continue
        for code in cands.loc[cands["distressed_code"] == d["bse_code"], "candidate_code"]:
            v = assets.get((code, int(d["size_fy"])))
            if v and a * (1 - tol) <= v <= a * (1 + tol):
                rows.append({"distressed_code": d["bse_code"], "candidate_code": code, "assets": v})
    return pd.DataFrame(rows, columns=["distressed_code", "candidate_code", "assets"])


def step_full_shortlist(D: Path, cfg: dict, jobs_out: Path) -> None:
    t = load_tables(D)
    assets, _ = xbrl_assets(D)
    new = _full_new(D)
    cands = pd.read_csv(D / "interim/full_peer_candidates.csv", dtype={"distressed_code": "string", "candidate_code": "string"})
    short = _in_tolerance(new, cands, assets, cfg["cohort"]["asset_tolerance"])
    have = set(t["reports"]["bse_code"]) if len(t["reports"]) else set()
    items = _new_items(D, [J.bse_annual_reports(c) for c in sorted(set(short["candidate_code"]) - have)])
    path = J.write_jobs(jobs_out, "014_full_candidate_reports", items,
                       "Full cohort: annual-report list of every possible peer within +/-30% of its insolvent "
                       "firm's total assets (a peer needs reports for the pair's years).", delay_s=0.4)
    log.info("%d insolvent firms have a candidate within tolerance; %d candidates; wrote %s with %d items",
             short["distressed_code"].nunique() if len(short) else 0,
             short["candidate_code"].nunique() if len(short) else 0, path, len(items))


def step_full_finalize(D: Path, cfg: dict, jobs_out: Path) -> None:
    t = load_tables(D)
    assets, _ = xbrl_assets(D)
    new = _full_new(D)
    cands = pd.read_csv(D / "interim/full_peer_candidates.csv", dtype={"distressed_code": "string", "candidate_code": "string"})
    rep = t["reports"]
    n_cand = cfg["cohort"]["n_candidate_years"]
    counts = {}
    for _, d in new.iterrows():
        ref = int(d["reference_fy"])
        yrs = set(range(ref - n_cand + 1, ref + 1))
        for code in cands.loc[cands["distressed_code"] == d["bse_code"], "candidate_code"]:
            counts[(code, ref)] = int(rep[(rep["bse_code"] == code) & rep["fy"].isin(yrs)]["fy"].nunique())
    pairs, unmatched = choose_peers(new, cands, assets, counts, cfg, cfg["cohort"]["random_seed"])
    pairs = pairs.merge(new[["bse_code", "admission_date", "admission_year"]].rename(columns={"bse_code": "distressed_code"}),
                        on="distressed_code", how="left")
    pairs = pairs.sort_values(["admission_year", "distressed_code"]).reset_index(drop=True)
    pairs.to_csv(D / "interim/full_pairs.csv", index=False)
    unmatched.to_csv(D / "interim/full_unmatched.csv", index=False)
    log.info("new pairs: %d (unmatched %d: %s)", len(pairs), len(unmatched),
             unmatched["reason"].value_counts().to_dict() if len(unmatched) else {})

    # the pilot's cohort keeps its pair ids; new pairs are numbered after it
    pilot_cohort = pd.read_csv(D / "processed/cohort_pilot.csv") if (D / "processed/cohort_pilot.csv").exists() \
        else pd.read_csv(D / "processed/cohort.csv")
    if not (D / "processed/cohort_pilot.csv").exists():
        pilot_cohort.to_csv(D / "processed/cohort_pilot.csv", index=False)
    start = int(pilot_cohort["pair_id"].str[1:].astype(int).max()) + 1 if len(pilot_cohort) else 1
    uni = pd.read_csv(D / "interim/listed_universe.csv", dtype={"bse_code": "string"})
    names = dict(zip(code_str(uni["bse_code"].dropna()), uni.loc[uni["bse_code"].notna(), "company_name"]))
    hdr = t["headers"].set_index("bse_code")
    nd = new.set_index("bse_code")
    rows = []
    for i, p in pairs.iterrows():
        pid = f"P{start + i:04d}"
        d = nd.loc[p["distressed_code"]]
        for role, code, assets_v in (("distressed", p["distressed_code"], p["distressed_assets"]),
                                     ("healthy", p["peer_code"], p["peer_assets"])):
            rows.append({
                "pair_id": pid, "firm_id": f"BSE{code}", "company_name": names.get(code),
                "role": role, "label": 1 if role == "distressed" else 0,
                "reference_date": d["admission_date"],
                "admission_date_source": "ibbi_pa_date" if role == "distressed" else None,
                "petition_date": None, "reference_fy": int(p["reference_fy"]),
                "industry_code": hdr["isubgroup_code"].get(code) if code in hdr.index else d["isubgroup_code"],
                "total_assets_ref_fy": round(float(assets_v), 2),
                "asset_ratio_to_distressed": 1.0 if role == "distressed" else p["asset_ratio"],
                "match_quality": p["match_quality"], "business_group": None,
                "cin": d["cin"] if role == "distressed" else None,
                "size_fy": int(p["size_fy"]),
            })
    cohort = pd.concat([pilot_cohort, pd.DataFrame(rows)], ignore_index=True)
    cohort.to_csv(D / "processed/cohort.csv", index=False)
    frame = build_sample_frame(cohort, cfg)
    frame.to_csv(D / "processed/sample_frame.csv", index=False)
    log.info("full cohort: %d pairs, %d firms; sample frame %d firm-years",
             cohort["pair_id"].nunique(), len(cohort), len(frame))

    have_rep = rep.set_index(["bse_code", "fy"]) if len(rep) else None
    xb = t["xbrl_index"].dropna(subset=["xbrl_url"]).set_index(["bse_code", "fy"])
    new_firms = {f"BSE{c}" for c in set(pairs["distressed_code"]) | set(pairs["peer_code"])}
    items, missing = [], []
    for _, r in frame[frame["firm_id"].isin(new_firms)].iterrows():
        code, fy = r["firm_id"].replace("BSE", ""), int(r["fy"])
        if have_rep is not None and (code, fy) in have_rep.index:
            url = have_rep.loc[(code, fy)]
            url = url.iloc[0]["url"] if isinstance(url, pd.DataFrame) else url["url"]
            items.append(J.report_pdf(r["firm_id"], fy, url))
        else:
            missing.append((r["firm_id"], fy))
        if (code, fy) in xb.index:
            items.append(J.xbrl_file(r["firm_id"], fy, xb.loc[(code, fy), "xbrl_url"]))
    uniq = _new_items(D, list({(i.get("path") or i["key"]): i for i in items}.values()))
    path = J.write_jobs(jobs_out, "015_full_reports", uniq,
                       "Full cohort: annual-report PDFs and standalone XBRL results for every firm-year of the "
                       "new pairs.", delay_s=1.0, per_job=120)
    log.info("report job: %d items (%d PDFs); no report listed for %d firm-years; wrote %s",
             len(uniq), sum(1 for i in uniq if i.get("expect") == "pdf"), len(missing), path)


def step_xbrl(D: Path) -> None:
    _, rows = xbrl_assets(D)
    out = D / "interim/xbrl_financials_exchange.csv"
    rows.to_csv(out, index=False)
    log.info("exchange XBRL figures: %d rows for %d firm-years -> %s", len(rows),
             rows[["firm_id", "fy"]].drop_duplicates().shape[0] if len(rows) else 0, out)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def step_manifest(D: Path) -> None:
    """interim/documents.csv from the PDFs present under raw/annual_reports."""
    t = load_tables(D)
    rep = t["reports"].copy()
    rows = []
    for pdf in sorted((D / "raw/annual_reports").glob("BSE*/FY*.pdf")):
        firm_id, fy = pdf.parent.name, int(pdf.stem[2:])
        code = firm_id.replace("BSE", "")
        hit = rep[(rep["bse_code"] == code) & (rep["fy"] == fy)]
        filed = hit["filed_at"].iloc[0] if len(hit) else pd.NaT
        rows.append({"doc_id": doc_id_for(firm_id, fy), "firm_id": firm_id, "fy": fy, "source": "bse",
                     "url": hit["url"].iloc[0] if len(hit) else "", "local_path": str(pdf),
                     "pub_date": filed.date().isoformat() if pd.notna(filed) else "",
                     "sha256": _sha256(pdf), "n_bytes": pdf.stat().st_size, "status": "downloaded",
                     "note": "" if pd.notna(filed) else "no filing date on BSE; assumed rule applies"})
    man = pd.DataFrame(rows)
    man.to_csv(D / "interim/documents.csv", index=False)
    log.info("manifest: %d documents", len(man))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("step", choices=["parse", "select", "candidates", "peers", "finalize", "xbrl", "manifest",
                                     "full-select", "full-candidates", "full-sizing", "full-shortlist",
                                     "full-finalize", "full-distressed-reports"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--n-pairs", type=int, default=30)
    ap.add_argument("--years", default="2019-2025")
    ap.add_argument("--jobs-out", type=Path, default=Path("jobs_out"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_cfg()
    y0, y1 = (int(x) for x in a.years.split("-"))
    if a.step == "parse":
        step_parse(a.data_dir, cfg)
    elif a.step == "select":
        step_select(a.data_dir, cfg, a.n, (y0, y1), a.jobs_out)
    elif a.step == "candidates":
        step_candidates(a.data_dir, cfg, a.jobs_out)
    elif a.step == "peers":
        step_peers(a.data_dir, cfg, a.jobs_out)
    elif a.step == "finalize":
        step_finalize(a.data_dir, cfg, a.n_pairs, a.jobs_out)
    elif a.step == "xbrl":
        step_xbrl(a.data_dir)
    elif a.step == "manifest":
        step_manifest(a.data_dir)
    else:
        {"full-select": step_full_select, "full-candidates": step_full_candidates,
         "full-sizing": step_full_sizing, "full-shortlist": step_full_shortlist,
         "full-finalize": step_full_finalize,
         "full-distressed-reports": step_full_distressed_reports}[a.step](a.data_dir, cfg, a.jobs_out)


if __name__ == "__main__":
    main()
