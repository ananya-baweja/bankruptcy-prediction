"""Synthetic (fake) data so the whole Phase 1-2 pipeline can run offline.

Used by ``bpp demo`` and the tests. All company names are invented. Nothing here
is real data and none of it may be used in results.

What gets generated
-------------------
* a cached IBBI announcements page (HTML, same table layout as the real site)
* BSE/NSE company lists -> data/interim/listed_universe.csv
* data/manual/firm_financials.csv (industry + total assets per FY)
* after the cohort is built: annual report PDFs laid out like Indian reports
  (cover, contents page, Board's Report with cross-reference sub-headings, MD&A
  with running headers, Corporate Governance report, standalone auditor's
  report with opinion/going-concern/emphasis sections, CARO Annexure A,
  Annexure B, balance sheet, consolidated auditor's report). One page is an
  image (scanned) to exercise OCR, one firm-year is missing on purpose, and the
  most recent distressed report mentions the CIRP to exercise leakage rules.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import pymupdf as fitz
import numpy as np
import pandas as pd

from bpp.common import doc_id_for, fy_end_date, write_csv
from bpp.config import Paths
from bpp.scrape.listed import build_universe

DISTRESSED = [
    # (bse_code, nse_symbol, name, industry_code, base_assets_cr, ibbi_name, admission_date)
    ("900101", "ZENTHRA", "Zenthra Infraprojects Limited", "42101", 5200, "ZENTHRA INFRAPROJECTS LTD.", "14-08-2019"),
    ("900102", "KALPAVIK", "Kalpavik Textiles Limited", "13111", 1800,
     "Kalpavik Textiles Limited (formerly known as Kalpavik Spinners Limited)", "10-02-2020"),
    ("900103", "RUDRASEN", "Rudrasen Power Limited", "35102", 9100, "RUDRASEN POWER LIMITED", "03-11-2021"),
    ("900104", "MIROVAN", "Mirovan Shipyards Limited", "30111", 3900, "Mirovan Shipyards Ltd", "20-06-2018"),
]
HEALTHY = [
    ("900201", "TARANGI", "Tarangi Infra Limited", "42101", 5000),
    ("900202", "VELMORA", "Velmora Buildcon Limited", "42101", 6100),
    ("900203", "SUTRAVA", "Sutrava Fabrics Limited", "13111", 1700),
    ("900204", "NEELAKSH", "Neelaksh Spinning Mills Limited", "13111", 2150),
    ("900205", "PRAVAHI", "Pravahi Energy Limited", "35102", 8800),
    ("900206", "OJASWI", "Ojaswi Thermal Limited", "35101", 9500),
    ("900207", "SAGARIK", "Sagarik Marine Engineering Limited", "30111", 4100),
    ("900208", "DHRUVAM", "Dhruvam Ship Repairs Limited", "30112", 3600),
]
CIRP_PRIVATE = [("Halcyon Agro Foods Private Limited", "05-03-2020"), ("Brisk Logistics Pvt. Ltd.", "11-09-2019")]
NEAR_MISS = ("Tarang Infra Limited", "22-07-2020")   # looks like "Tarangi Infra" -> needs human review

HEALTHY_SENTENCES = [
    "The Company recorded healthy growth in revenue driven by strong order inflows and improved capacity utilisation.",
    "Operating margins improved on account of better realisations and prudent cost management.",
    "The Company continued to maintain a comfortable liquidity position and met all debt obligations on time.",
    "Capital expenditure during the year was funded through internal accruals.",
    "The outlook for the sector remains positive, supported by government spending and steady demand.",
    "Working capital cycles remained stable and receivables were collected within the agreed credit period.",
    "Credit rating agencies reaffirmed the Company's long-term rating with a stable outlook.",
    "The Board is confident of sustaining profitable growth in the coming years.",
]
DISTRESS_SENTENCES = [
    "The Company may face challenges in meeting its obligations to lenders due to continued stress on cash flows.",
    "Delays in realisation of receivables have stretched the working capital cycle significantly.",
    "The Company is in discussions with its lenders for restructuring of debt and a one time settlement.",
    "Certain loan accounts of the Company have been classified as non-performing assets by the lenders.",
    "There could be delays in payment of statutory dues owing to the tight liquidity position.",
    "Lenders have invoked the pledge on part of the promoter shareholding during the year.",
    "The management believes that, subject to successful restructuring, operations could be sustained.",
    "Execution of projects was affected by funding constraints and cost overruns.",
]
NEUTRAL_SENTENCES = [
    "The Board of Directors presents the Annual Report together with the audited financial statements for the year.",
    "The Company has complied with the applicable Secretarial Standards issued by the Institute of Company Secretaries.",
    "Details of related party transactions are disclosed in the notes to the financial statements.",
    "The Company has in place adequate internal financial controls with reference to financial statements.",
    "Particulars of employees are available for inspection at the registered office of the Company.",
    "Information on conservation of energy and technology absorption is provided as required.",
]


def _paras(rng: random.Random, pool: list[str], n_paras: int, per: int = 5) -> str:
    return "\n\n".join(" ".join(rng.choice(pool) for _ in range(per)) for _ in range(n_paras))


def ibbi_html(rows: list[dict[str, str]]) -> str:
    head = ("<tr><th>Type of PA</th><th>Date of Announcement</th><th>Last date of Submission</th>"
            "<th>Name of Corporate Debtor</th><th>Name of Applicant</th><th>Name of Insolvency Professional</th>"
            "<th>Public Announcement</th><th>Remarks</th></tr>")
    body = "".join(
        f"<tr><td>{r['type']}</td><td>{r['date']}</td><td>{r['date']}</td><td>{r['debtor']}</td>"
        f"<td>{r.get('applicant', 'State Bank of India')}</td><td>Demo IP</td>"
        f"<td><a href='/uploads/announcement/demo{i:04d}.pdf'>View</a></td><td></td></tr>"
        for i, r in enumerate(rows))
    return f"<html><body><table class='table'><thead>{head}</thead><tbody>{body}</tbody></table></body></html>"


def make_synthetic_inputs(paths: Paths, seed: int = 0) -> None:
    """IBBI page cache, listed universe and financials for the demo."""
    rng = np.random.default_rng(seed)
    rows = [{"type": "Corporate Insolvency Resolution Process", "date": d[6], "debtor": d[5]} for d in DISTRESSED]
    rows += [{"type": "Corporate Insolvency Resolution Process", "date": d, "debtor": n} for n, d in CIRP_PRIVATE]
    rows += [{"type": "Corporate Insolvency Resolution Process", "date": NEAR_MISS[1], "debtor": NEAR_MISS[0]}]
    rows += [{"type": "Liquidation Process", "date": "01-02-2022", "debtor": DISTRESSED[3][5]},
             {"type": "Voluntary Liquidation Process", "date": "15-04-2021", "debtor": "Quietbrook Traders Limited"}]
    rows.sort(key=lambda r: pd.to_datetime(r["date"], dayfirst=True), reverse=True)
    paths.ibbi_pages.mkdir(parents=True, exist_ok=True)
    (paths.ibbi_pages / "page_00001.html").write_text(ibbi_html(rows), encoding="utf-8")

    bse = pd.DataFrame(
        [{"bse_code": d[0], "company_name": d[2], "isin": f"INE{d[0]}01", "industry": d[3],
          "status": "Suspended"} for d in DISTRESSED] +
        [{"bse_code": h[0], "company_name": h[2], "isin": f"INE{h[0]}01", "industry": h[3],
          "status": "Active"} for h in HEALTHY])
    nse = pd.DataFrame([{"nse_symbol": x[1], "company_name": x[2], "isin": f"INE{x[0]}01", "status": "Active"}
                        for x in DISTRESSED + HEALTHY])
    write_csv(bse, paths.raw_listed / "bse_scrips.csv")
    write_csv(nse, paths.raw_listed / "nse_equity_list.csv")
    write_csv(build_universe(bse, nse), paths.listed_universe)

    fin = []
    for code, _, name, ind, base, *_ in DISTRESSED:
        for fy in range(2012, 2025):
            fin.append({"firm_id": f"BSE{code}", "company_name": name, "fy": fy, "industry_code": ind,
                        "total_assets": round(base * (1 + 0.03 * (fy - 2016)) * rng.uniform(0.97, 1.03), 1)})
    for code, _, name, ind, base in HEALTHY:
        for fy in range(2012, 2025):
            fin.append({"firm_id": f"BSE{code}", "company_name": name, "fy": fy, "industry_code": ind,
                        "total_assets": round(base * (1 + 0.03 * (fy - 2016)) * rng.uniform(0.97, 1.03), 1)})
    write_csv(pd.DataFrame(fin), paths.financials)


# ----------------------------------------------------------------------------- PDFs
class _PdfWriter:
    W, H, MARGIN, FONT = 595, 842, 50, 9.5

    def __init__(self) -> None:
        self.doc = fitz.open()

    def page(self, text: str, running_header: str | None = None) -> None:
        pg = self.doc.new_page(width=self.W, height=self.H)
        y = self.MARGIN
        if running_header:
            pg.insert_text((self.MARGIN, 30), running_header, fontsize=8, fontname="helv")
        rect = fitz.Rect(self.MARGIN, y, self.W - self.MARGIN, self.H - self.MARGIN)
        if pg.insert_textbox(rect, text, fontsize=self.FONT, fontname="helv") < 0:
            # text did not fit (PyMuPDF then writes nothing): split it over two pages
            self.doc.delete_page(-1)
            cut = text.rfind("\n\n", 0, len(text) // 2 + 200)
            cut = cut if cut > 0 else len(text) // 2
            self.page(text[:cut], running_header)
            self.page(text[cut:].lstrip(), running_header)

    def flow(self, heading: str, body: str, running_header: str | None = None, chars_per_page: int = 2600) -> None:
        """Write a heading + body over as many pages as needed."""
        chunks, cur = [], heading + "\n\n"
        for para in body.split("\n\n"):
            if len(cur) + len(para) > chars_per_page and cur.strip():
                chunks.append(cur)
                cur = ""
            cur += para + "\n\n"
        chunks.append(cur)
        for i, ch in enumerate(chunks):
            self.page(ch, running_header if i > 0 else None)

    def scanned_page(self, text: str) -> None:
        tmp = fitz.open()
        p = tmp.new_page(width=self.W, height=self.H)
        p.insert_textbox(fitz.Rect(self.MARGIN, self.MARGIN, self.W - self.MARGIN, self.H - self.MARGIN),
                         text, fontsize=12, fontname="helv")
        pix = p.get_pixmap(dpi=150)
        pg = self.doc.new_page(width=self.W, height=self.H)
        pg.insert_image(pg.rect, pixmap=pix)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(path)


# ----------------------------------------------------------------------------- statements
# Phase 3 reads the standalone balance sheet, statement of profit and loss and
# cash flow statement. The one-page stub further down is enough for Phase 2
# (which only needs a heading to end the auditor's report at), but not for
# testing extraction: that needs a prior-year column, a stated unit, bracketed
# negatives, a note column, Schedule III sub-headings and a consolidated set to
# be discriminated against. Built as text so the figures can be tested without
# rendering a PDF; ``write_report_pdf`` puts them on pages.


def _fmt(value: float, indian: bool = False) -> str:
    """Format a figure the way a statement prints it: grouped, brackets if negative."""
    negative = value < 0
    body = f"{abs(value):,.2f}"
    if indian:                       # 12,34,567.89 rather than 1,234,567.89
        whole, _, frac = body.replace(",", "").partition(".")
        if len(whole) > 3:
            head, tail = whole[:-3], whole[-3:]
            head = re.sub(r"(?<=\d)(?=(\d{2})+$)", ",", head)
            whole = f"{head},{tail}"
        body = f"{whole}.{frac}"
    return f"({body})" if negative else body


#: the year a distressed firm's decline bottoms out, so a firm-year's figures
#: are the same number whichever report they are read from
TREND_REFERENCE_FY = 2019


def year_figures(fy: int, distressed: bool, seed: int = 0) -> dict[str, float]:
    """Figures for one fiscal year, in ₹ crore, balancing exactly.

    A company-year always produces the same numbers, whether it is read as a
    report's own year or as the next report's comparative column. That matters:
    if the fixture drifted between the two, every year would look restated and
    the restatement test could never fail. Restatements are introduced
    deliberately, by :func:`statement_figures`.
    """
    rng = random.Random(f"{seed}|{fy}|{int(distressed)}")
    years_out = max(0, TREND_REFERENCE_FY - fy)      # 0 = the worst year
    jitter = rng.uniform(0.98, 1.02)

    if distressed:
        equity_share = -0.20 + 0.22 * min(years_out, 2)      # 2017 +0.24, 2018 +0.02, 2019 -0.20
        size = 1.0 + 0.06 * min(years_out, 3)                 # shrinking as trouble builds
    else:
        equity_share, size = 0.60, 1.0 - 0.04 * min(years_out, 3)

    total_assets = round(7800 * size * jitter, 2)
    current_assets = round(total_assets * 0.36, 2)
    equity = round(total_assets * equity_share, 2)
    non_current_liabilities = round(total_assets * (0.52 if distressed else 0.16), 2)
    current_liabilities = round(total_assets - equity - non_current_liabilities, 2)
    borrowings_short = round(current_liabilities * 0.51, 2)
    share_capital = 450.00

    revenue = round(total_assets * (0.41 if distressed else 0.80), 2)
    other_income = round(revenue * 0.014, 2)
    finance_costs = round(total_assets * (0.078 if distressed else 0.012), 2)
    depreciation = round(total_assets * 0.048, 2)
    other_expenses = round(revenue * (0.94 if distressed else 0.74), 2)
    total_income = round(revenue + other_income, 2)
    total_expenses = round(other_expenses + finance_costs + depreciation, 2)
    pbt = round(total_income - total_expenses, 2)
    tax = round(max(pbt, 0.0) * 0.26, 2)

    return {
        "fy": fy,
        "total_assets": total_assets,
        "current_assets": current_assets,
        "non_current_assets": round(total_assets - current_assets, 2),
        "inventories": round(current_assets * 0.42, 2),
        "cash": round(current_assets * (0.031 if distressed else 0.17), 2),
        "other_bank": round(current_assets * 0.005, 2),
        "share_capital": share_capital,
        "other_equity": round(equity - share_capital, 2),
        "total_equity": equity,
        "borrowings_long": non_current_liabilities,
        "non_current_liabilities": non_current_liabilities,
        "borrowings_short": borrowings_short,
        "trade_payables": round(current_liabilities - borrowings_short, 2),
        "current_liabilities": current_liabilities,
        "revenue": revenue,
        "other_income": other_income,
        "total_income": total_income,
        "other_expenses": other_expenses,
        "finance_costs": finance_costs,
        "depreciation": depreciation,
        "total_expenses": total_expenses,
        "pbt": pbt,
        "tax": tax,
        "net_profit": round(pbt - tax, 2),
    }


def statement_figures(fy: int, distressed: bool, seed: int = 0,
                      restate_prior: float = 0.0) -> dict[str, dict[str, float]]:
    """The current and prior-year columns one report prints.

    ``restate_prior`` multiplies the comparative column, to simulate a company
    restating last year's figures -- which is what the restatement flag exists
    to catch.
    """
    current = year_figures(fy, distressed, seed)
    prior = year_figures(fy - 1, distressed, seed)
    if restate_prior:
        prior = {k: (round(v * (1 + restate_prior), 2) if k != "fy" else v)
                 for k, v in prior.items()}
    return {"current": current, "prior": prior}


def statement_pages(company: str, fy: int, distressed: bool, seed: int = 0,
                    unit: str = "crore", restate_prior: float = 0.0) -> list[str]:
    """Page texts for the standalone statements, then a consolidated balance sheet."""
    figures = statement_figures(fy, distressed, seed, restate_prior)
    cur, pri = figures["current"], figures["prior"]
    scale = {"crore": 1.0, "lakh": 100.0, "million": 10.0}[unit]
    caption = {"crore": "(All amounts in Rs. crore, unless otherwise stated)",
               "lakh": "(Rs. in lakhs)",
               "million": "(All amounts in INR million)"}[unit]
    indian = unit == "lakh"

    def row(label: str, key: str, note: str = "") -> str:
        return (f"{label} {note} {_fmt(cur[key] * scale, indian)} "
                f"{_fmt(pri[key] * scale, indian)}").replace("  ", " ")

    balance_sheet = "\n".join([
        f"STANDALONE BALANCE SHEET AS AT 31ST MARCH, {fy}",
        caption,
        f"Particulars Note As at 31 March {fy} As at 31 March {fy - 1}",
        "ASSETS",
        "Non-current assets",
        row("Property, plant and equipment", "non_current_assets", "3"),
        row("Total non-current assets", "non_current_assets"),
        "Current assets",
        row("Inventories", "inventories", "7"),
        row("Cash and cash equivalents", "cash", "9"),
        row("Bank balances other than cash and cash equivalents", "other_bank", "10"),
        row("Total current assets", "current_assets"),
        row("Total assets", "total_assets"),
        "EQUITY AND LIABILITIES",
        "Equity",
        row("Equity share capital", "share_capital", "13"),
        row("Other equity", "other_equity", "14"),
        row("Total equity", "total_equity"),
        "Non-current liabilities",
        row("Borrowings", "borrowings_long", "15"),
        row("Total non-current liabilities", "non_current_liabilities"),
        "Current liabilities",
        row("Borrowings", "borrowings_short", "18"),
        row("Trade payables", "trade_payables", "19"),
        row("Total current liabilities", "current_liabilities"),
        row("Total equity and liabilities", "total_assets"),
    ])

    profit_and_loss = "\n".join([
        f"STANDALONE STATEMENT OF PROFIT AND LOSS FOR THE YEAR ENDED 31ST MARCH, {fy}",
        caption,
        f"Particulars Note Year ended 31 March {fy} Year ended 31 March {fy - 1}",
        row("Revenue from operations", "revenue", "21"),
        row("Other income", "other_income", "22"),
        row("Total income", "total_income"),
        row("Other expenses", "other_expenses", "23"),
        row("Finance costs", "finance_costs", "25"),
        row("Depreciation and amortisation expense", "depreciation", "26"),
        row("Total expenses", "total_expenses"),
        row("Profit/(loss) before tax", "pbt"),
        row("Tax expense", "tax"),
        row("Profit/(loss) for the year", "net_profit"),
    ])

    cash_flow = "\n".join([
        f"STANDALONE CASH FLOW STATEMENT FOR THE YEAR ENDED 31ST MARCH, {fy}",
        caption,
        f"Particulars Year ended 31 March {fy} Year ended 31 March {fy - 1}",
        row("Profit/(loss) before tax", "pbt"),
        row("Depreciation and amortisation expense", "depreciation"),
        row("Finance costs", "finance_costs"),
        row("Cash and cash equivalents at the end of the year", "cash"),
    ])

    # The consolidated set repeats every heading with different totals, so the
    # extractor has to prefer the standalone one rather than the first it meets.
    consolidated = "\n".join([
        f"CONSOLIDATED BALANCE SHEET AS AT 31ST MARCH, {fy}",
        caption,
        f"Particulars Note As at 31 March {fy} As at 31 March {fy - 1}",
        f"Total assets {_fmt(cur['total_assets'] * 1.38 * scale, indian)} "
        f"{_fmt(pri['total_assets'] * 1.38 * scale, indian)}",
        f"Total equity {_fmt(cur['total_equity'] * 1.22 * scale, indian)} "
        f"{_fmt(pri['total_equity'] * 1.22 * scale, indian)}",
    ])

    return [balance_sheet, profit_and_loss, cash_flow, consolidated]


def write_report_pdf(path: Path, company: str, fy: int, distressed: bool, mention_cirp: bool,
                     scanned_mdna_page: bool, seed: int,
                     financial_statements: bool = False, statement_unit: str = "crore") -> None:
    rng = random.Random(seed)
    tone = DISTRESS_SENTENCES if distressed else HEALTHY_SENTENCES
    yr = f"{fy - 1}-{str(fy)[2:]}"
    w = _PdfWriter()
    w.page(f"{company.upper()}\n\n\nANNUAL REPORT {yr}\n\nBuilding on strength, delivering on commitments")
    w.page("CONTENTS\n\nBoard's Report ........................ 3\n"
           "Management Discussion and Analysis .......... 6\n"
           "Report on Corporate Governance ............ 9\n"
           "Independent Auditor's Report .............. 11\n"
           "Standalone Balance Sheet as at 31st March .. 16")

    board = ("To the Members,\n\n" + _paras(rng, NEUTRAL_SENTENCES, 3) + "\n\nFINANCIAL PERFORMANCE\n\n"
             + _paras(rng, tone, 2) +
             "\n\nManagement Discussion and Analysis\n\nThe Management Discussion and Analysis Report for the "
             "year under review, as stipulated under Regulation 34 of the SEBI Listing Regulations, is presented "
             "in a separate section forming part of this Annual Report.\n\n"
             "Corporate Governance\n\nThe Report on Corporate Governance forms an integral part of this "
             "Annual Report.\n\n" + _paras(rng, NEUTRAL_SENTENCES, 4))
    if mention_cirp:
        board += ("\n\nMATERIAL EVENTS\n\nAn operational creditor has filed an application under Section 9 of the "
                  "Insolvency and Bankruptcy Code before the National Company Law Tribunal. The corporate "
                  "insolvency resolution process may be initiated if the application is admitted.")
    w.flow("BOARD'S REPORT", board, running_header=f"Annual Report {yr} | Board's Report")

    mdna_body = ("INDUSTRY STRUCTURE AND DEVELOPMENTS\n\n" + _paras(rng, NEUTRAL_SENTENCES + tone, 2) +
                 "\n\nOPPORTUNITIES AND THREATS\n\n" + _paras(rng, tone, 3) +
                 "\n\nRISKS AND CONCERNS\n\n" + _paras(rng, tone, 3) +
                 "\n\nOUTLOOK\n\n" + _paras(rng, tone, 2))
    w.flow("MANAGEMENT DISCUSSION AND ANALYSIS REPORT", mdna_body, running_header="Management Discussion and Analysis")
    if scanned_mdna_page:
        w.scanned_page("RISKS AND CONCERNS (CONTINUED)\n\n" + " ".join(rng.choice(tone) for _ in range(8)))

    w.flow("REPORT ON CORPORATE GOVERNANCE", _paras(rng, NEUTRAL_SENTENCES, 6))

    opinion = "Qualified Opinion" if distressed else "Opinion"
    aud = (f"To the Members of {company}\n\nReport on the Audit of the Standalone Financial Statements\n\n"
           f"{opinion}\n\nWe have audited the standalone financial statements of the Company. In our opinion "
           + ("except for the possible effects of the matter described in the Basis for Qualified Opinion section, "
              if distressed else "") +
           "the aforesaid standalone financial statements give a true and fair view.\n\n")
    if distressed:
        aud += ("Basis for Qualified Opinion\n\nThe Company has not provided for interest on loans classified as "
                "non-performing assets by lenders amounting to Rs. 412 crore, which is not in accordance with "
                "Ind AS 109. Had the interest been provided, the loss for the year would have been higher.\n\n"
                "Material Uncertainty Related to Going Concern\n\nThe Company has incurred losses, its net worth "
                "has eroded and it has defaulted in repayment of borrowings. These conditions indicate that a "
                "material uncertainty exists that may cast significant doubt on the Company's ability to continue "
                "as a going concern.\n\n")
    else:
        aud += "Basis for Opinion\n\nWe conducted our audit in accordance with the Standards on Auditing.\n\n"
    aud += ("Emphasis of Matter\n\nWe draw attention to the note regarding recoverability of trade receivables. "
            "Our opinion is not modified in respect of this matter.\n\n"
            "Key Audit Matters\n\n" + _paras(rng, NEUTRAL_SENTENCES, 2) +
            "\n\nManagement's Responsibility for the Standalone Financial Statements\n\n"
            + _paras(rng, NEUTRAL_SENTENCES, 2) +
            "\n\nAuditor's Responsibilities for the Audit of the Standalone Financial Statements\n\n"
            + _paras(rng, NEUTRAL_SENTENCES, 2) +
            "\n\nReport on Other Legal and Regulatory Requirements\n\nAs required by the Companies (Auditor's "
            "Report) Order, 2020, we give in Annexure A a statement on the matters specified.")
    w.flow("INDEPENDENT AUDITOR'S REPORT", aud)

    caro = ("(Referred to in paragraph 1 under Report on Other Legal and Regulatory Requirements)\n\n"
            "(i) The Company has maintained proper records showing full particulars of property, plant and "
            "equipment.\n\n")
    if distressed:
        caro += ("(vii) The Company has not been regular in depositing undisputed statutory dues including "
                 "provident fund and income tax, and dues outstanding for more than six months are Rs. 38 crore.\n\n"
                 "(ix) The Company has defaulted in repayment of loans and interest to banks and financial "
                 "institutions. The lenders have declared the Company a wilful defaulter.\n\n")
    else:
        caro += ("(vii) The Company has been regular in depositing undisputed statutory dues.\n\n"
                 "(ix) The Company has not defaulted in repayment of loans or borrowings to any lender.\n\n")
    caro += _paras(rng, NEUTRAL_SENTENCES, 5)
    w.flow("ANNEXURE 'A' TO THE INDEPENDENT AUDITOR'S REPORT", caro)
    w.flow("ANNEXURE 'B' TO THE INDEPENDENT AUDITORS' REPORT",
           "Report on the Internal Financial Controls under Clause (i) of Sub-section 3 of Section 143 of the "
           "Companies Act, 2013\n\n" + _paras(rng, NEUTRAL_SENTENCES, 4))
    if financial_statements:
        # full Schedule III statements, for Phase 3 extraction
        for page_text in statement_pages(company, fy, distressed, seed, statement_unit):
            w.page(page_text)
    else:
        w.page(f"STANDALONE BALANCE SHEET AS AT 31ST MARCH, {fy}\n\n(Rs. in crore)\n\nASSETS\nNon-current assets\n"
               "Property, plant and equipment 1,234.5\nCurrent assets\nInventories 345.6\nTotal assets 2,468.1")
    w.flow("INDEPENDENT AUDITOR'S REPORT",
           f"To the Members of {company}\n\nReport on the Audit of the Consolidated Financial Statements\n\n"
           "Opinion\n\nWe have audited the consolidated financial statements of the Holding Company and its "
           "subsidiaries.\n\n" + _paras(rng, NEUTRAL_SENTENCES, 4))
    w.save(path)


def make_synthetic_reports(paths: Paths, cfg: dict[str, Any], seed: int = 0) -> pd.DataFrame:
    """Create PDFs for the sample frame (one firm-year left missing) + manual_reports.csv."""
    frame = pd.read_csv(paths.sample_frame)
    fy_cfg = cfg["fiscal_year"]
    manual_rows = []
    missing_done = False
    scanned_done = False
    for i, r in frame.sort_values(["pair_id", "role", "fy"]).reset_index(drop=True).iterrows():
        distressed = r["role"] == "distressed"
        if distressed and r["fy_rank_before_reference"] == 4 and not missing_done:
            missing_done = True            # simulate a report nobody could find
            continue
        pub = fy_end_date(r["fy"], fy_cfg["end_month"], fy_cfg["end_day"]) + pd.Timedelta(days=150 + (i % 20))
        mention = distressed and r["fy_rank_before_reference"] == 1
        scanned = distressed and not scanned_done
        scanned_done = scanned_done or scanned
        path = paths.raw_reports / r["firm_id"] / f"FY{int(r['fy'])}.pdf"
        write_report_pdf(path, r["company_name"], int(r["fy"]), distressed, mention, scanned, seed + i)
        manual_rows.append({"firm_id": r["firm_id"], "fy": int(r["fy"]), "local_path": "",
                            "pub_date": pub.date().isoformat(), "source_url": "synthetic", "note": "demo"})
    df = pd.DataFrame(manual_rows)
    write_csv(df, paths.manual_reports)
    return df
