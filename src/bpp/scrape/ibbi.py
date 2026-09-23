"""Scrape IBBI public announcements to find companies admitted to CIRP.

Why this source
---------------
When the NCLT admits a company into the Corporate Insolvency Resolution
Process (CIRP), the Interim Resolution Professional must publish a Public
Announcement (Form A) within about 3 days. IBBI lists every announcement at
https://ibbi.gov.in/en/public-announcement with the columns:

    Type of PA | Date of Announcement | Last date of Submission |
    Name of Corporate Debtor | Name of Applicant | Name of Insolvency Professional |
    Public Announcement (PDF) | Remarks

So the announcement date is a close proxy (within days) for the admission date.
The exact Insolvency Commencement Date is printed inside the PDF; for the final
cohort, check it by hand for your ~100 firms (column ``admission_date_verified``).

How the scraper works
---------------------
* Pages are fetched as ``?page=N`` (starting at 0, which is the first page on
  Drupal-style pagers; if the site is 1-indexed, page 0 simply repeats page 1 and
  duplicates are removed) and cached to data/raw/ibbi/pages/ so a
  crash or re-run never re-downloads pages already saved.
* The table is parsed by header text, not column position, so small layout
  changes on the site do not silently shift columns.
* Scraping stops at an empty page, at ``max_pages``, or once every row on a page
  is older than ``earliest_date`` (the list is newest-first).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from bpp.common import (PoliteSession, clean_text, is_private_company,
                        normalize_company_name, to_timestamp, write_csv)
from bpp.config import Paths

log = logging.getLogger(__name__)

# header keyword -> output column. Order matters: first match wins.
HEADER_MAP = [
    ("corporate debtor", "corporate_debtor"),
    ("type", "pa_type"),
    ("date of announcement", "announcement_date"),
    ("last date", "last_submission_date"),
    ("applicant", "applicant"),
    ("insolvency professional", "insolvency_professional"),
    ("public announcement", "pa_pdf_url"),
    ("remark", "remarks"),
]
COLUMNS = [c for _, c in HEADER_MAP]


def _map_header(text: str) -> str | None:
    t = clean_text(text).lower()
    for key, col in HEADER_MAP:
        if key in t:
            return col
    return None


def parse_announcement_page(html: str, base_url: str = "https://ibbi.gov.in") -> list[dict[str, Any]]:
    """Parse one IBBI announcement listing page into row dicts."""
    soup = BeautifulSoup(html, "lxml")
    target, header_row = None, None
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if any("corporate debtor" in c.get_text(" ").lower() for c in cells):
                target, header_row = table, tr         # header may be <th> or a first row of <td>
                break
        if target is not None:
            break
    if target is None:
        return []

    cols = [_map_header(c.get_text(" ")) for c in header_row.find_all(["th", "td"])]
    rows = []
    for tr in target.find_all("tr"):
        if tr is header_row:
            continue
        tds = tr.find_all("td")
        if not tds or len(tds) < 4:
            continue
        rec: dict[str, Any] = {c: None for c in COLUMNS}
        for col, td in zip(cols, tds):
            if col is None:
                continue
            if col == "pa_pdf_url":
                a = td.find("a", href=True)
                href = a["href"] if a else None
                if href and not href.startswith("http"):
                    href = base_url.rstrip("/") + "/" + href.lstrip("/")
                rec[col] = href
            else:
                rec[col] = clean_text(td.get_text(" "))
        if rec["pa_pdf_url"] is None:                       # PDF link sometimes sits in another cell
            a = tr.find("a", href=lambda h: h and h.lower().endswith(".pdf"))
            if a:
                href = a["href"]
                rec["pa_pdf_url"] = href if href.startswith("http") else base_url.rstrip("/") + "/" + href.lstrip("/")
        rows.append(rec)
    return rows


def scrape_public_announcements(cfg: dict[str, Any], paths: Paths, max_pages: int | None = None,
                                resume: bool = True, start_page: int = 0) -> pd.DataFrame:
    """Download all announcement pages (cached) and save one combined CSV."""
    icfg = cfg["ibbi"]
    max_pages = max_pages or icfg.get("max_pages")
    earliest = to_timestamp(str(icfg.get("earliest_date"))) if icfg.get("earliest_date") else None
    sess = PoliteSession(delay_s=icfg["request_delay_s"], timeout_s=icfg["timeout_s"])

    page = start_page
    while True:
        if max_pages and page > start_page + max_pages - 1:
            break
        cache = paths.ibbi_pages / f"page_{page:05d}.html"
        if resume and cache.exists():
            html = cache.read_text(encoding="utf-8")
        else:
            log.info("fetching IBBI page %d", page)
            html = sess.get(icfg["url"], params={"page": page}).text
            cache.write_text(html, encoding="utf-8")
        rows = parse_announcement_page(html)
        if not rows:
            log.info("page %d has no rows - stopping", page)
            break
        if earliest is not None:
            dates = [to_timestamp(r["announcement_date"]) for r in rows]
            dates = [d for d in dates if d is not None]
            if dates and max(dates) < earliest:
                log.info("page %d is entirely before %s - stopping", page, earliest.date())
                break
        page += 1

    return combine_cached_pages(paths)


def combine_cached_pages(paths: Paths) -> pd.DataFrame:
    """Parse every cached page and write data/raw/ibbi/public_announcements.csv."""
    frames = []
    for f in sorted(paths.ibbi_pages.glob("page_*.html")):
        rows = parse_announcement_page(f.read_text(encoding="utf-8"))
        if rows:
            df = pd.DataFrame(rows)
            df["source_page"] = f.stem
            frames.append(df)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS + ["source_page"])
    df = df.drop_duplicates(subset=["pa_type", "announcement_date", "corporate_debtor", "pa_pdf_url"])
    write_csv(df, paths.ibbi_announcements)
    return df


def filter_cirp_announcements(cfg: dict[str, Any], paths: Paths,
                              df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Keep CIRP announcements for companies that could be listed; one row per debtor.

    * Type of PA must contain ``cirp_type_keyword`` (drops liquidation, voluntary
      liquidation and pre-pack announcements).
    * Private limited companies, OPCs and LLPs are dropped (cannot be listed).
    * A debtor can have several CIRP announcements (corrections, re-admission);
      the earliest date is kept as the admission proxy.
    """
    if df is None:
        df = pd.read_csv(paths.ibbi_announcements)
    kw = cfg["ibbi"]["cirp_type_keyword"].lower()
    out = df[df["pa_type"].fillna("").str.lower().str.contains(kw)].copy()
    n_cirp = len(out)
    out = out[~out["corporate_debtor"].apply(is_private_company)]
    out["announcement_date"] = pd.to_datetime(out["announcement_date"], dayfirst=True, errors="coerce")
    out["name_norm"] = out["corporate_debtor"].apply(normalize_company_name)
    out = (out.sort_values("announcement_date")
              .groupby("name_norm", as_index=False)
              .agg(corporate_debtor=("corporate_debtor", "first"),
                   cirp_announcement_date=("announcement_date", "first"),
                   n_announcements=("announcement_date", "size"),
                   applicant=("applicant", "first"),
                   pa_pdf_url=("pa_pdf_url", "first")))
    log.info("CIRP announcements: %d, non-private: %d unique debtors", n_cirp, len(out))
    write_csv(out, paths.cirp_listed_candidates)
    return out
