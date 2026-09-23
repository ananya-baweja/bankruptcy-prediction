"""Build the universe of BSE/NSE listed companies (active, suspended and delisted).

Why suspended/delisted matter: most firms admitted to CIRP are suspended or
delisted by the time we study them. A list of only *active* companies would
miss exactly the firms we need.

Sources
-------
* BSE "List of Scrips" API (``ListofScripData``), queried once per status
  (Active / Suspended / Delisted). Parameter names follow the open-source
  BseIndiaApi client (github.com/BennyThadikaran/BseIndiaApi).
* NSE equity list CSV (``EQUITY_L.csv``) - active NSE companies only.
* ``data/manual/listed_companies_extra.csv`` - anything the team adds by hand
  (e.g. firms found only in news or the IBBI newsletter).

Output: data/interim/listed_universe.csv with one row per company:
firm_id, company_name, name_norm, bse_code, nse_symbol, isin, status, industry, sources
"""

from __future__ import annotations

import io
import logging
from typing import Any

import pandas as pd

from bpp.common import PoliteSession, normalize_company_name, write_csv
from bpp.config import Paths

log = logging.getLogger(__name__)


def _pick(rec: dict[str, Any], *candidates: str) -> Any:
    """Case-insensitive field lookup; APIs rename fields without notice."""
    low = {k.lower(): v for k, v in rec.items()}
    for c in candidates:
        if c.lower() in low and low[c.lower()] not in (None, ""):
            return low[c.lower()]
    return None


def parse_bse_scrips(records: list[dict[str, Any]], status: str) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            "bse_code": str(_pick(r, "SCRIP_CD", "scripcode", "Scrip_Code") or "").strip(),
            "company_name": _pick(r, "Issuer_Name", "Scrip_Name", "scrip_name", "LONG_NAME"),
            "bse_symbol": _pick(r, "scrip_id", "Scrip_Id", "symbol"),
            "isin": _pick(r, "ISIN_NUMBER", "isin"),
            "industry": _pick(r, "INDUSTRY", "Industry"),
            "status": _pick(r, "Status") or status,
        })
    return pd.DataFrame(rows)


def fetch_bse_universe(cfg: dict[str, Any]) -> pd.DataFrame:
    lcfg = cfg["listed"]
    sess = PoliteSession(delay_s=lcfg["request_delay_s"], timeout_s=lcfg["timeout_s"],
                         headers={"Referer": "https://www.bseindia.com/",
                                  "Origin": "https://www.bseindia.com",
                                  "Accept": "application/json, text/plain, */*"})
    frames = []
    for status in lcfg["bse_statuses"]:
        url = f"{lcfg['bse_api']}/ListofScripData/w"
        params = {"Group": "", "scripcode": "", "industry": "", "segment": "Equity", "status": status}
        try:
            data = sess.get(url, params=params).json()
        except Exception as exc:  # noqa: BLE001 - network errors should not kill the run
            log.warning("BSE list for status=%s failed: %s", status, exc)
            continue
        records = data if isinstance(data, list) else data.get("Table", [])
        df = parse_bse_scrips(records, status)
        log.info("BSE %s: %d scrips", status, len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_nse_universe(cfg: dict[str, Any]) -> pd.DataFrame:
    lcfg = cfg["listed"]
    sess = PoliteSession(delay_s=lcfg["request_delay_s"], timeout_s=lcfg["timeout_s"])
    try:
        text = sess.get(lcfg["nse_equity_list_url"]).text
    except Exception as exc:  # noqa: BLE001
        log.warning("NSE equity list failed: %s", exc)
        return pd.DataFrame()
    return parse_nse_equity_list(text)


def parse_nse_equity_list(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(csv_text))
    df.columns = [c.strip().upper() for c in df.columns]
    return pd.DataFrame({
        "nse_symbol": df.get("SYMBOL"),
        "company_name": df.get("NAME OF COMPANY"),
        "isin": df.get("ISIN NUMBER"),
        "status": "Active",
    })


def build_universe(bse: pd.DataFrame, nse: pd.DataFrame, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge BSE + NSE (+ manual) on ISIN, falling back to normalised name."""
    frames = []
    if not bse.empty:
        b = bse.copy()
        b["sources"] = "bse"
        frames.append(b)
    if not nse.empty:
        n = nse.copy()
        n["sources"] = "nse"
        frames.append(n)
    if extra is not None and not extra.empty:
        e = extra.copy()
        e["sources"] = "manual"
        frames.append(e)
    if not frames:
        return pd.DataFrame(columns=["firm_id", "company_name", "name_norm", "bse_code", "nse_symbol",
                                     "isin", "status", "industry", "sources"])
    allrows = pd.concat(frames, ignore_index=True)
    for col in ["bse_code", "nse_symbol", "isin", "industry", "status", "bse_symbol"]:
        if col not in allrows:
            allrows[col] = None
    allrows["bse_code"] = allrows["bse_code"].astype("string").str.replace(r"\.0$", "", regex=True)
    allrows["name_norm"] = allrows["company_name"].apply(normalize_company_name)
    allrows["key"] = allrows["isin"].fillna("").astype(str).str.strip()
    allrows.loc[allrows["key"] == "", "key"] = "NAME:" + allrows["name_norm"]

    def first_valid(s: pd.Series) -> Any:
        s = s.dropna()
        s = s[s.astype(str).str.strip() != ""]
        return s.iloc[0] if len(s) else None

    merged = allrows.groupby("key", as_index=False).agg(
        company_name=("company_name", first_valid),
        name_norm=("name_norm", first_valid),
        bse_code=("bse_code", first_valid),
        nse_symbol=("nse_symbol", first_valid),
        isin=("isin", first_valid),
        status=("status", first_valid),
        industry=("industry", first_valid),
        sources=("sources", lambda s: "+".join(sorted(set(s)))),
    )
    merged["firm_id"] = merged.apply(_firm_id, axis=1)
    merged = merged.drop(columns="key").drop_duplicates("firm_id")
    cols = ["firm_id", "company_name", "name_norm", "bse_code", "nse_symbol", "isin", "status",
            "industry", "sources"]
    return merged[cols].sort_values("company_name").reset_index(drop=True)


def _firm_id(row: pd.Series) -> str:
    """Stable id: BSE code if known, else NSE symbol, else ISIN."""
    if pd.notna(row.get("bse_code")) and str(row["bse_code"]).strip():
        return f"BSE{str(row['bse_code']).strip()}"
    if pd.notna(row.get("nse_symbol")) and str(row["nse_symbol"]).strip():
        return f"NSE_{str(row['nse_symbol']).strip()}"
    return f"ISIN_{row.get('isin')}"


def fetch_listed_universe(cfg: dict[str, Any], paths: Paths, sources: list[str]) -> pd.DataFrame:
    bse = fetch_bse_universe(cfg) if "bse" in sources else pd.DataFrame()
    nse = fetch_nse_universe(cfg) if "nse" in sources else pd.DataFrame()
    if not bse.empty:
        write_csv(bse, paths.raw_listed / "bse_scrips.csv")
    if not nse.empty:
        write_csv(nse, paths.raw_listed / "nse_equity_list.csv")
    extra = pd.read_csv(paths.extra_listed) if paths.extra_listed.exists() else None
    uni = build_universe(bse, nse, extra)
    if uni.empty:
        log.error("Universe is empty. Download failed? Add companies to %s by hand.", paths.extra_listed)
    write_csv(uni, paths.listed_universe)
    return uni
