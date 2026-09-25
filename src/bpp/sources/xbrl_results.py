"""Read the exchange XBRL filing of a company's audited annual results.

Since FY2018 every listed company files its quarterly and annual results with
NSE/BSE in XBRL, tagged with the exchanges' Ind AS taxonomy (``in-bse-fin``).
The annual filing (quarter ending March, "Audited") carries the standalone
statement of profit and loss for the full year and the statement of assets and
liabilities at the year end - the same audited figures the annual report prints
a few months later, already tagged line by line. That is what makes it the
closest public substitute for a CMIE Prowess record.

What this module does
---------------------
* finds, for each tag, the fact whose context is the FULL fiscal year (P&L) or
  the fiscal year END (balance sheet). The file also carries the March quarter
  on its own ("OneD" = Jan-Mar) - reading that by mistake gives a quarter's
  revenue labelled as the year's, so contexts are chosen by their dates, never
  by their ids;
* skips dimensional contexts (segments, expense breakdowns);
* converts to Rs crore. Facts are stored in rupees (``unitRef="INR"``); the
  "level of rounding" tag says how the company *presented* them, not their unit;
* maps the tags onto the standard fields used by Phase 3.

Older non-Ind AS filings use other tag names; the alternatives are listed in
``TAGS`` and the first one present wins. A field with no tag present is simply
absent - never zero.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date
from typing import Any

RUPEES_PER_CRORE = 1e7

# standard field -> candidate tags (local names), first present wins
TAGS: dict[str, list[str]] = {
    # balance sheet (instant)
    "total_assets": ["Assets", "TotalAssets"],
    "current_assets": ["CurrentAssets", "TotalCurrentAssets"],
    "non_current_assets": ["NoncurrentAssets", "TotalNonCurrentAssets"],
    "inventories": ["Inventories"],
    "cash_and_equivalents": ["CashAndCashEquivalents", "CashAndBankBalances"],
    "bank_balances_other": ["BankBalanceOtherThanCashAndCashEquivalents"],
    "trade_receivables": ["TradeReceivablesCurrent", "TradeReceivables"],
    "total_equity": ["Equity", "EquityAttributableToOwnersOfParent", "ShareholdersFunds"],
    "equity_share_capital": ["EquityShareCapital", "PaidUpValueOfEquityShareCapital"],
    "other_equity": ["OtherEquity", "ReservesAndSurplus"],
    "borrowings_long_term": ["BorrowingsNoncurrent", "LongTermBorrowings"],
    "borrowings_short_term": ["BorrowingsCurrent", "ShortTermBorrowings"],
    "current_liabilities": ["CurrentLiabilities", "TotalCurrentLiabilities"],
    "non_current_liabilities": ["NoncurrentLiabilities", "TotalNonCurrentLiabilities"],
    "total_liabilities": ["Liabilities"],
    "total_equity_and_liabilities": ["EquityAndLiabilities", "EquityAndLiabilitiesTotal"],
    # profit and loss (duration)
    "revenue": ["RevenueFromOperations", "NetSalesIncomeFromOperations", "IncomeFromOperations"],
    "other_income": ["OtherIncome"],
    "total_income": ["Income", "TotalIncome", "TotalIncomeFromOperations"],
    "total_expenses": ["Expenses", "TotalExpenses"],
    "finance_costs": ["FinanceCosts", "FinanceCost", "InterestExpense"],
    "depreciation": ["DepreciationDepletionAndAmortisationExpense", "DepreciationAndAmortisationExpense"],
    "exceptional_items": ["ExceptionalItemsBeforeTax", "ExceptionalItems"],
    "pbt": ["ProfitBeforeTax", "ProfitLossBeforeTax"],
    "tax_expense": ["TaxExpense"],
    "net_profit": ["ProfitLossForPeriod", "ProfitLossForThePeriod", "NetProfitLossForThePeriod"],
    # cash flow (duration)
    "cfo": ["CashFlowsFromUsedInOperatingActivities"],
}
INSTANT_FIELDS = {
    "total_assets", "current_assets", "non_current_assets", "inventories", "cash_and_equivalents",
    "bank_balances_other", "trade_receivables", "total_equity", "equity_share_capital",
    "other_equity", "borrowings_long_term", "borrowings_short_term", "current_liabilities",
    "non_current_liabilities", "total_liabilities", "total_equity_and_liabilities",
}
TEXT_TAGS = {
    "company_name": "NameOfTheCompany",
    "scrip_code": "ScripCode",
    "symbol": "Symbol",
    "fy_start": "DateOfStartOfFinancialYear",
    "fy_end": "DateOfEndOfFinancialYear",
    "nature": "NatureOfReportStandaloneConsolidated",
    "audited": "WhetherResultsAreAuditedOrUnaudited",
    "rounding": "LevelOfRoundingUsedInFinancialStatements",
    "currency": "DescriptionOfPresentationCurrency",
    "audit_declaration": "DeclarationOfUnmodifiedOpinionOrStatementOnImpactOfAuditQualification",
    "board_meeting_date": "DateOfBoardMeetingWhenFinancialResultsWereApproved",
}


@dataclass
class Context:
    id: str
    start: date | None = None
    end: date | None = None
    instant: date | None = None
    dimensional: bool = False

    @property
    def days(self) -> int | None:
        if self.start and self.end:
            return (self.end - self.start).days + 1
        return None


@dataclass
class XbrlResult:
    meta: dict[str, Any] = field(default_factory=dict)
    values_cr: dict[str, float] = field(default_factory=dict)
    tags_used: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def fy(self) -> int | None:
        end = self.meta.get("fy_end_date")
        return end.year if isinstance(end, date) and end.month <= 3 else (end.year + 1 if isinstance(end, date) else None)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    m = re.match(r"\s*(\d{4})-(\d{2})-(\d{2})", s)
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def _contexts(root: ET.Element) -> dict[str, Context]:
    out: dict[str, Context] = {}
    for el in root:
        if _local(el.tag) != "context":
            continue
        c = Context(id=el.get("id", ""))
        for sub in el.iter():
            name = _local(sub.tag)
            if name == "startDate":
                c.start = _parse_date(sub.text)
            elif name == "endDate":
                c.end = _parse_date(sub.text)
            elif name == "instant":
                c.instant = _parse_date(sub.text)
            elif name in ("explicitMember", "typedMember"):
                c.dimensional = True
        out[c.id] = c
    return out


def _number(text: str | None) -> float | None:
    if text is None:
        return None
    s = text.strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def parse_results_xbrl(xml_text: str) -> XbrlResult:
    """Parse one annual-results XBRL instance into standard fields (Rs crore)."""
    res = XbrlResult()
    root = ET.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text)
    ctx = _contexts(root)

    facts: dict[str, list[tuple[Context, ET.Element]]] = {}
    for el in root:
        name = _local(el.tag)
        ref = el.get("contextRef")
        if ref is None or ref not in ctx:
            continue
        facts.setdefault(name, []).append((ctx[ref], el))

    def text_fact(tag: str) -> str | None:
        for c, el in facts.get(tag, []):
            if not c.dimensional and el.text and el.text.strip():
                return el.text.strip()
        return None

    for key, tag in TEXT_TAGS.items():
        res.meta[key] = text_fact(tag)

    fy_end = _parse_date(res.meta.get("fy_end"))
    fy_start = _parse_date(res.meta.get("fy_start"))
    if fy_end is None:
        # fall back to the longest non-dimensional duration in the file
        durations = [c for c in ctx.values() if not c.dimensional and c.days]
        if durations:
            best = max(durations, key=lambda c: (c.days, c.end))
            fy_start, fy_end = best.start, best.end
            res.warnings.append("fy_end_inferred_from_contexts")
    res.meta["fy_start_date"], res.meta["fy_end_date"] = fy_start, fy_end
    if fy_end is None:
        res.warnings.append("no_fiscal_year_found")
        return res

    def pick(tag: str, instant: bool) -> tuple[float | None, str | None]:
        cands = []
        for c, el in facts.get(tag, []):
            if c.dimensional:
                continue
            if el.get("unitRef") and el.get("unitRef").upper() not in ("INR", "U_INR", "INRS"):
                continue
            v = _number(el.text)
            if v is None:
                continue
            if instant and c.instant == fy_end:
                cands.append((0, v))
            elif not instant and c.end == fy_end and c.days and c.days >= 360:
                cands.append((0 if (fy_start is None or c.start == fy_start) else 1, v))
        if not cands:
            return None, None
        cands.sort(key=lambda t: t[0])
        values = {round(v, 2) for _, v in cands if _ == cands[0][0]}
        if len(values) > 1:
            res.warnings.append(f"conflicting_values:{tag}")
        return cands[0][1], tag

    for fld, tags in TAGS.items():
        instant = fld in INSTANT_FIELDS
        for tag in tags:
            v, used = pick(tag, instant)
            if v is not None:
                res.values_cr[fld] = v / RUPEES_PER_CRORE
                res.tags_used[fld] = used
                break

    # a year-to-date duration that is really a quarter is the classic trap: flag it
    q_only = [t for t in ("RevenueFromOperations", "ProfitBeforeTax")
              if t in facts and all(c.days and c.days < 120 for c, _ in facts[t] if not c.dimensional)]
    if q_only and "revenue" not in res.values_cr:
        res.warnings.append("only_quarterly_figures")

    nature = (res.meta.get("nature") or "").lower()
    res.meta["standalone"] = None if not nature else ("standalone" in nature or "non" in nature)
    decl = (res.meta.get("audit_declaration") or "").lower()
    res.meta["audit_modified"] = None if not decl else ("qualification" in decl and "unmodified" not in decl)
    return res


def result_to_rows(res: XbrlResult, firm_id: str, source_url: str, note: str = "") -> list[dict[str, Any]]:
    """Rows in the ``manual/xbrl_financials.csv`` layout Phase 3 reads."""
    fy = res.fy
    rows = []
    for fld, v in res.values_cr.items():
        rows.append({"firm_id": firm_id, "fy": fy, "field": fld, "value_cr": round(v, 4),
                     "source_url": source_url,
                     "note": ";".join(filter(None, [note, f"tag={res.tags_used.get(fld)}"]))})
    return rows
