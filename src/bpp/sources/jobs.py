"""Write job files for the laptop download agent (``bpp_fetch.py``).

The agent runs on a team member's laptop (an Indian address: the exchanges
block cloud addresses) and processes ``jobs/*.json`` in name order. A job is a
list of items; each item downloads one URL either to its own file (``save``) or
as one line of a JSON-lines file (``collect``, for thousands of small API
responses that would otherwise be thousands of files to move around).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

BSE_API = "https://api.bseindia.com/BseIndiaAPI/api/"
NSE_API = "https://www.nseindia.com/api/"


def bse_header(code: str) -> dict[str, Any]:
    return {"url": f"{BSE_API}ComHeadernew/w?quotetype=EQ&scripcode={code}&seriesid=", "kind": "collect",
            "collect": "raw/listed/bse_company_header.jsonl", "key": str(code)}


def bse_result_archive(code: str) -> dict[str, Any]:
    return {"url": f"{BSE_API}Result_Arch_ng/w?scrip_cd={code}", "kind": "collect",
            "collect": "raw/xbrl/_listings/bse_result_archive.jsonl", "key": str(code)}


def bse_annual_reports(code: str) -> dict[str, Any]:
    return {"url": f"{BSE_API}AnnualReport_New/w?scripcode={code}", "kind": "collect",
            "collect": "raw/annual_reports/_listings/bse_annual_reports.jsonl", "key": str(code)}


def xbrl_file(firm_id: str, fy: int, url: str) -> dict[str, Any]:
    return {"url": url, "kind": "collect", "collect": "raw/xbrl/results_standalone_annual.jsonl",
            "key": f"{firm_id}_FY{int(fy)}", "meta": {"firm_id": firm_id, "fy": int(fy)}}


def report_pdf(firm_id: str, fy: int, url: str) -> dict[str, Any]:
    return {"url": url, "path": f"raw/annual_reports/{firm_id}/FY{int(fy)}.pdf", "expect": "pdf",
            "key": f"{firm_id}_FY{int(fy)}"}


def write_job(out_dir: Path | str, name: str, items: Iterable[dict[str, Any]], note: str,
              delay_s: float = 0.8) -> Path:
    items = list(items)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    path.write_text(json.dumps({"name": name, "delay_s": delay_s, "note": note, "items": items}),
                    encoding="utf-8")
    return path
