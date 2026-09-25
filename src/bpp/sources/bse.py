"""Parse what the download agent collected from the BSE API.

Every ``*.jsonl`` file written by ``bpp_fetch.py`` holds one JSON line per
request: ``{"key", "url", "http", "fetched_at", "body"}`` with the raw response
text in ``body``. The functions here turn those into tables.

* ``ComHeadernew``      - one company's industry classification
  (Sector > Industry > Group > Sub-group). Empty for many delisted scrips.
* ``Result_Arch_ng``    - every results filing with its XBRL links and filing
  time. The annual standalone filing is the row whose ``Quarter`` reads
  ``Standalone-Mar-YY;MC<fy>;<code>;D`` (MC = March, cumulative = full year).
* ``AnnualReport_New``  - the annual reports with PDF link and filing time.
* ``GetINDUSTRYWATCHLIST_ng`` - the 186 sub-groups with their hierarchical codes
  ``IN`` + sector(2) + industry(2) + group(2) + sub-group(3), e.g. IN020101002.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

BSE_WWW = "https://www.bseindia.com"


def collect_parts(path: Path | str) -> list[Path]:
    """``name.jsonl`` and its parts ``name_<job>.jsonl``, oldest first.

    Large jobs write their own part so that each file stays small enough to move
    from the laptop in one transfer (a single growing file passed 100 MB).
    """
    p = Path(path)
    parts = sorted(p.parent.glob(f"{p.stem}_*{p.suffix}")) if p.parent.exists() else []
    return ([p] if p.exists() else []) + parts


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Records of a collect file and its parts; the latest record wins when a key repeats."""
    latest: dict[str, dict[str, Any]] = {}
    for p in collect_parts(path):
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                latest[str(rec.get("key"))] = rec
    return iter(latest.values())


def _body_json(rec: dict[str, Any]) -> Any:
    if rec.get("http") != 200:
        return None
    try:
        return json.loads(rec.get("body") or "")
    except json.JSONDecodeError:
        return None


def _rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("Table", "table", "Table1", "data"):
            if isinstance(data.get(k), list):
                return data[k]
    return []


def _blank(v: Any) -> Any:
    return None if v in (None, "", "-", "--") else v


# ----------------------------------------------------------------- industry
def parse_industry_list(path: Path | str) -> pd.DataFrame:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    df = pd.DataFrame(_rows(data))
    df = df.rename(columns={"INDUSTRY_NAME": "isubgroup", "ISUBGROUP_CODE": "isubgroup_code", "cnt": "n_companies"})
    return df[["isubgroup", "isubgroup_code", "n_companies"]]


def parse_company_headers(path: Path | str, industry_list: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for rec in iter_jsonl(path):
        d = _body_json(rec)
        if not isinstance(d, dict):
            continue
        rows.append({
            "bse_code": str(rec["key"]),
            "security_id": _blank(d.get("SecurityId")),
            "isin": _blank(d.get("ISIN")),
            "bse_group": _blank(d.get("Group")),
            "industry_old": _blank(d.get("Industry")),
            "sector": _blank(d.get("Sector")),
            "industry_new": _blank(d.get("IndustryNew")),
            "igroup": _blank(d.get("IGroup")),
            "isubgroup": _blank(d.get("ISubGroup")),
        })
    df = pd.DataFrame(rows)
    if industry_list is not None and len(df):
        codes = dict(zip(industry_list["isubgroup"].str.strip().str.lower(), industry_list["isubgroup_code"]))
        df["isubgroup_code"] = df["isubgroup"].fillna("").str.strip().str.lower().map(codes)
    return df


def parse_traded_members(path: Path | str) -> pd.DataFrame:
    """``HeatMap_ng`` per sub-group: 'bse$#$SYM,chg,SYM,open,high,low,close,diff,CODE,...|...'."""
    rows = []
    for rec in iter_jsonl(path):
        body = (rec.get("body") or "").strip().strip('"')
        body = body.split("$#$", 1)[-1]
        for part in body.split("|"):
            f = part.split(",")
            if len(f) >= 9 and f[8].strip().isdigit():
                rows.append({"isubgroup_code": rec["key"], "symbol": f[0].strip(), "bse_code": f[8].strip()})
    return pd.DataFrame(rows).drop_duplicates(["isubgroup_code", "bse_code"]) if rows else pd.DataFrame(
        columns=["isubgroup_code", "symbol", "bse_code"])


# ----------------------------------------------------------------- results / XBRL
_QTR = re.compile(r"^(?P<nature>Standalone|Consolidated)-(?P<mon>[A-Za-z]{3})-(?P<yy>\d{2});(?P<kind>[A-Z]{2})"
                  r"(?P<fy1>\d{4})-(?P<fy2>\d{4});(?P<code>[\d.]+);(?P<rt>\w+)$")


def parse_result_archive(path: Path | str) -> pd.DataFrame:
    rows = []
    for rec in iter_jsonl(path):
        for r in _rows(_body_json(rec)):
            q = str(r.get("Quarter") or "")
            m = _QTR.match(q)
            link = r.get("stand_xbrl_link") or r.get("conso_xbrl_link")
            link = link if link and not str(link).endswith("/") else None
            rows.append({
                "bse_code": str(rec["key"]),
                "quarter_label": q,
                "nature": m.group("nature").lower() if m else None,
                "period_kind": m.group("kind") if m else None,          # MC = full year, MQ = quarter, ...
                "month": m.group("mon") if m else None,
                "fy": int(m.group("fy2")) if m else None,
                "qtr_code": r.get("qtr"),
                "xbrl_url": (BSE_WWW + link) if link and link.startswith("/") else link,
                "filed_at": _blank(r.get("Filing_Date_Time")),
                "revised_at": _blank(r.get("Revised_Date_Time")),
                "revision_reason": _blank(r.get("Revision_Reason")),
            })
    df = pd.DataFrame(rows)
    if len(df):
        df["filed_at"] = pd.to_datetime(df["filed_at"], errors="coerce")
        df["revised_at"] = pd.to_datetime(df["revised_at"], errors="coerce")
    return df


def annual_standalone_xbrl(archive: pd.DataFrame) -> pd.DataFrame:
    """One row per (bse_code, fy): the full-year standalone filing for a March year end."""
    a = archive[(archive["nature"] == "standalone") & (archive["period_kind"] == "MC") & (archive["month"] == "Mar")]
    a = a.sort_values("filed_at").drop_duplicates(["bse_code", "fy"], keep="first")
    return a[["bse_code", "fy", "xbrl_url", "filed_at", "revised_at"]].reset_index(drop=True)


# ----------------------------------------------------------------- annual reports
def parse_annual_report_list(path: Path | str) -> pd.DataFrame:
    rows = []
    for rec in iter_jsonl(path):
        for r in _rows(_body_json(rec)):
            url = str(r.get("PDFDownload") or "").strip().replace("\\", "/")
            url = re.sub(r"(?<!:)//+", "/", url)
            if not url or not url.lower().endswith(".pdf"):
                continue
            year = pd.to_numeric(r.get("Year"), errors="coerce")
            rows.append({
                "bse_code": str(rec["key"]),
                "fy": int(year) if pd.notna(year) else None,
                "url": url,
                "filed_at": _blank(r.get("Fld_AuthoriseDate")),
                "revised_at": _blank(r.get("revised_date_time")),
                "status": _blank(r.get("status")),
                "delisted_flag": _blank(r.get("StatusForDelisted")),
                "suspended_flag": _blank(r.get("StatusForSuS")),
            })
    df = pd.DataFrame(rows)
    if len(df):
        df["filed_at"] = pd.to_datetime(df["filed_at"], errors="coerce")
    return df
