"""Small helpers shared across phases: logging, HTTP, names, fiscal years."""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("bpp")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# --------------------------------------------------------------------------- HTTP
class PoliteSession:
    """requests.Session with retries and a minimum delay between requests.

    Public websites (IBBI, BSE, NSE) will block you if you hammer them, so
    every request waits at least ``delay_s`` seconds after the previous one.
    """

    def __init__(self, delay_s: float = 2.0, timeout_s: float = 30,
                 headers: dict[str, str] | None = None):
        self.delay_s = delay_s
        self.timeout_s = timeout_s
        self._last = 0.0
        self.session = requests.Session()
        retry = Retry(total=4, backoff_factor=2.0,
                      status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=["GET", "HEAD"])
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.headers.update({
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
            "Accept-Language": "en-US,en;q=0.9",
        })
        if headers:
            self.session.headers.update(headers)

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        wait = self.delay_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        kwargs.setdefault("timeout", self.timeout_s)
        try:
            resp = self.session.get(url, **kwargs)
        finally:
            self._last = time.monotonic()
        resp.raise_for_status()
        return resp


# --------------------------------------------------------------------------- names
# Only drop words that never distinguish two companies. "INDIA" is kept on purpose:
# "XYZ India Ltd" and "XYZ Ltd" are often different legal entities (parent vs subsidiary).
_SUFFIXES = {"LIMITED", "LTD", "THE"}


def clean_text(s: Any) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"\s+", " ", s).strip()


def normalize_company_name(name: Any) -> str:
    """Canonical form used for fuzzy matching company names.

    'The Jaypee Infratech Ltd.' -> 'JAYPEE INFRATECH'
    Removes punctuation, legal suffixes and bracketed notes like '(In Liquidation)'.
    """
    s = clean_text(name).upper()
    s = re.sub(r"\(.*?\)", " ", s)                    # bracketed remarks
    s = re.split(r"\bFORMERLY\b|\bEARLIER\b|\bF/K/A\b|\bNOW KNOWN AS\b", s)[0]
    s = s.replace("&", " AND ")
    s = re.sub(r"\bPVT\b\.?|\bPRIVATE\b", " PRIVATE ", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    tokens = [t for t in s.split() if t]
    stripped = [t for t in tokens if t not in _SUFFIXES]
    if not stripped:                                   # e.g. a company literally named "India Ltd"
        stripped = tokens
    return " ".join(stripped)


def former_names(name: Any) -> list[str]:
    """Extract 'formerly known as X' aliases, which IBBI often includes."""
    s = clean_text(name)
    out = []
    for m in re.finditer(r"(?:formerly|earlier)\s+(?:known\s+as\s+)?[:\-]?\s*([^()]+)", s, flags=re.I):
        out.append(m.group(1).strip(" .,)"))
    return out


def is_private_company(name: Any) -> bool:
    """Private companies cannot be listed, so they are dropped early."""
    s = clean_text(name).upper()
    return bool(re.search(r"PRIVATE\s+LIMITED|PVT\.?\s*LTD|PVT\.?\s+LIMITED|\(OPC\)|\bLLP\b", s))


# --------------------------------------------------------------------------- fiscal years
def fy_end_date(fy: int, end_month: int = 3, end_day: int = 31) -> pd.Timestamp:
    """FY2023 -> 2023-03-31 (Indian fiscal years are labelled by the year they end)."""
    return pd.Timestamp(date(int(fy), end_month, end_day))


def fy_of_date(d: pd.Timestamp, end_month: int = 3) -> int:
    """Fiscal year that contains date d. 2022-05-10 -> FY2023."""
    d = pd.Timestamp(d)
    return d.year + 1 if d.month > end_month else d.year


def parse_fy_label(value: Any) -> int | None:
    """Parse many FY spellings into an int: '2022-23', 'FY23', '2023', 2023 -> 2023."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().upper()
    m = re.search(r"(19|20)(\d{2})\s*[-/–]\s*(\d{2,4})", s)
    if m:
        end = m.group(3)
        return int(end) if len(end) == 4 else int(m.group(1) + end)
    m = re.search(r"FY\s*'?(\d{2,4})", s)
    if m:
        y = m.group(1)
        return int(y) if len(y) == 4 else 2000 + int(y)
    m = re.search(r"(19|20)\d{2}", s)
    return int(m.group(0)) if m else None


def to_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return None
    s = str(value).strip()
    if re.match(r"^\d{4}-\d{1,2}-\d{1,2}", s):          # ISO 2019-08-30 (never day-first)
        ts = pd.to_datetime(s[:10], format="%Y-%m-%d", errors="coerce")
    else:                                                 # Indian style 30-08-2019 / 30/08/2019
        ts = pd.to_datetime(s, dayfirst=True, errors="coerce")
    return None if pd.isna(ts) else ts


# --------------------------------------------------------------------------- io
def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    log.info("wrote %s (%d rows)", path, len(df))
    return path


def read_csv(path: Path, required: Iterable[str] = (), **kwargs: Any) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} does not exist. See docs/ for which step creates it or how to fill it in.")
    df = pd.read_csv(path, **kwargs)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def doc_id_for(firm_id: str, fy: int) -> str:
    return f"{firm_id}_FY{int(fy)}"
