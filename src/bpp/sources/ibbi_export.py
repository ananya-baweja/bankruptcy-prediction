"""Read IBBI's own export of CIRP public announcements (the "Export to Excel" file).

The listing pages scraped by ``bpp ibbi-scrape`` do not carry the debtor's CIN.
The export does, and the CIN settles two questions Phase 1 otherwise answers by
fuzzy name matching:

* **listed or not** - a CIN starting with ``L`` belongs to a listed company
  (MCA assigns ``L`` to companies with listed securities; it can outlive a
  delisting, so ``L`` means "listed at some point", which is what we want);
* **which company** - the CIN is printed on the cover or first pages of every
  annual report, so a name match can be confirmed exactly later.

The file is tab-separated despite the ``.xls`` MIME type, one row per
announcement, newest first:

    Announcement Type | Date of Announcement | Last date of Submission |
    Name of Corporate Debtor | CIN No. | Name of Applicant |
    Name of Insolvency Professional | Address of Insolvency Professional | Remarks
"""

from __future__ import annotations

import csv
import html
import io
import re
from pathlib import Path

import pandas as pd

from bpp.common import clean_text, former_names, is_private_company, normalize_company_name

CIN_RE = re.compile(r"^[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}$")
COLUMN_MAP = {
    "announcement type": "pa_type",
    "date of announcement": "announcement_date",
    "last date of submission": "last_submission_date",
    "name of corporate debtor": "corporate_debtor",
    "cin no.": "cin",
    "name of applicant": "applicant",
    "name of insolvency professional": "insolvency_professional",
    "address of insolvency professional": "ip_address",
    "remarks": "remarks",
}
# CIN characters 13-15: the kind of company
CIN_COMPANY_TYPE = {
    "PLC": "public", "PTC": "private", "GOI": "central_govt", "SGC": "state_govt",
    "FLC": "public_financial", "FTC": "private_financial", "NPL": "not_for_profit",
    "ULL": "unlimited_public", "ULT": "unlimited_private", "GAP": "guarantee_public",
    "GAT": "guarantee_private", "OPC": "one_person",
}


def clean_cin(value: object) -> str | None:
    s = re.sub(r"\s+", "", clean_text(value)).upper()
    return s if CIN_RE.match(s) else None


def read_ibbi_export(path: Path | str) -> pd.DataFrame:
    """One row per announcement, with parsed dates and CIN fields."""
    raw = Path(path).read_bytes().decode("utf-8", "replace")
    reader = csv.reader(io.StringIO(raw), delimiter="\t", quoting=csv.QUOTE_NONE)
    header = [clean_text(h).lower() for h in next(reader)]
    cols = [COLUMN_MAP.get(h, h.replace(" ", "_")) for h in header]
    rows = []
    for rec in reader:
        if not any(x.strip() for x in rec):
            continue
        rec = (rec + [""] * len(cols))[: len(cols)]
        rows.append({c: clean_text(html.unescape(v)) for c, v in zip(cols, rec)})
    df = pd.DataFrame(rows)
    for c in ("announcement_date", "last_submission_date"):
        df[c] = pd.to_datetime(df[c], format="%d-%m-%Y", errors="coerce")
    df["cin_raw"] = df["cin"]
    df["cin"] = df["cin_raw"].map(clean_cin)
    df["cin_listed"] = df["cin"].str[0].eq("L")
    df["cin_nic"] = df["cin"].str[1:6]
    df["cin_state"] = df["cin"].str[6:8]
    df["cin_incorporation_year"] = pd.to_numeric(df["cin"].str[8:12], errors="coerce")
    df["cin_company_type"] = df["cin"].str[12:15].map(CIN_COMPANY_TYPE)
    df["name_norm"] = df["corporate_debtor"].map(normalize_company_name)
    df["pa_pdf_url"] = None                     # the export has no PDF link
    df["source"] = "ibbi_export"
    return df


def cirp_debtors(announcements: pd.DataFrame, keyword: str = "corporate insolvency resolution") -> pd.DataFrame:
    """One row per corporate debtor with its earliest CIRP announcement.

    Debtors are keyed by CIN where there is one, otherwise by normalised name,
    so a company that changed its name between two announcements is still one
    debtor. The earliest date is the admission proxy (decisions log, 16 Sep).
    """
    df = announcements[announcements["pa_type"].str.lower().str.contains(keyword, na=False)].copy()
    df["debtor_key"] = df["cin"].where(df["cin"].notna(), "NAME:" + df["name_norm"])
    df = df.sort_values("announcement_date")
    agg = df.groupby("debtor_key", as_index=False).agg(
        corporate_debtor=("corporate_debtor", "first"),
        name_norm=("name_norm", "first"),
        all_names=("corporate_debtor", lambda s: " | ".join(dict.fromkeys(s))),
        cin=("cin", "first"),
        cirp_announcement_date=("announcement_date", "first"),
        last_announcement_date=("announcement_date", "last"),
        n_announcements=("announcement_date", "size"),
        applicant=("applicant", "first"),
        remarks=("remarks", "first"),
    )
    agg["cin_listed"] = agg["cin"].str[0].eq("L")
    agg["cin_nic"] = agg["cin"].str[1:6]
    agg["cin_company_type"] = agg["cin"].str[12:15].map(CIN_COMPANY_TYPE)
    agg["private_by_name"] = agg["corporate_debtor"].map(is_private_company)
    agg["former_names"] = agg["corporate_debtor"].map(lambda n: " | ".join(former_names(n)))
    # corporate applicant = the debtor filed itself (IBC s.10)
    agg["self_filed"] = [normalize_company_name(a) == n for a, n in zip(agg["applicant"], agg["name_norm"])]
    return agg.sort_values("cirp_announcement_date").reset_index(drop=True)


def possibly_listed(debtors: pd.DataFrame) -> pd.DataFrame:
    """Debtors worth matching against the exchange lists.

    * an ``L`` CIN: listed (at some point);
    * no usable CIN and not a private company by name: cannot be ruled out.
    A ``U`` CIN is unlisted by definition and is dropped.
    """
    has_l = debtors["cin_listed"].fillna(False)
    no_cin = debtors["cin"].isna() & ~debtors["private_by_name"]
    out = debtors[has_l | no_cin].copy()
    out["listing_evidence"] = ["cin_L" if x else "no_cin" for x in has_l[has_l | no_cin]]
    return out.reset_index(drop=True)
