"""Find the standalone financial statements in a report and read the figures off them.

Input is what Phase 2 already produced -- ``data/interim/pages/<doc_id>.json``,
one entry per page with its text and whether that text came from the PDF layer
or from OCR. Nothing here re-reads a PDF for text, so OCR is never repeated.

Four steps
----------
1. **Locate.** Heading patterns find the Balance Sheet, Statement of Profit and
   Loss and Cash Flow Statement, and decide whether each one is *standalone* or
   *consolidated*. Consolidated statements mix in subsidiaries, so they are the
   wrong denominator for every ratio and are only used when nothing else exists
   (and then flagged). CARO and the standalone auditor's report already follow
   the same preference in Phase 2 -- see ``decisions_log.md``, 2026-09-16.
2. **Read the header.** The unit ("₹ in crore") and the period columns
   ("As at 31 March 2019 | 2018") are read once per statement and applied to
   every line beneath.
3. **Walk the lines.** Sub-headings (``Current assets``, ``Non-current
   liabilities``...) are tracked as context, because Schedule III prints the
   same label twice: ``Borrowings`` under non-current liabilities is long-term
   debt and under current liabilities is short-term debt. Without the context
   the two are indistinguishable.
4. **Map.** Labels are matched to the standard fields by pattern first and
   fuzzy similarity second, keeping the score so low-confidence matches can be
   sent to the spot-check sheet.

Tables from pdfplumber or camelot are used when the statement is ruled, because
column geometry is more reliable than whitespace. They are an improvement on
the text walk, not a replacement: a scanned statement has no table structure at
all, and the text path is what makes OCR'd reports work.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from bpp.features.numbers import (ParsedNumber, detect_unit, is_note_reference, parse_number,
                                  split_label_and_figures)

log = logging.getLogger(__name__)

STATEMENTS = ["balance_sheet", "profit_and_loss", "cash_flow"]

# --------------------------------------------------------------------------- locating
_SCOPE = r"(?P<scope>standalone|consolidated|separate)?\s*"

STATEMENT_PATTERNS: list[tuple[str, str]] = [
    ("balance_sheet", _SCOPE + r"balance\s+sheet\s+as\s+(?:at|on)\b"),
    ("balance_sheet", _SCOPE + r"statement\s+of\s+(?:assets\s+and\s+liabilities|financial\s+position)\b"),
    ("profit_and_loss", _SCOPE + r"statement\s+of\s+profit\s+(?:and|&)\s+loss\b"),
    ("profit_and_loss", _SCOPE + r"profit\s+(?:and|&)\s+loss\s+(?:statement|account)\b"),
    ("cash_flow", _SCOPE + r"(?:statement\s+of\s+)?cash\s+flows?\s+statement\b"),
    ("cash_flow", _SCOPE + r"statement\s+of\s+cash\s+flows?\b"),
    ("cash_flow", _SCOPE + r"cash\s+flow\s+statement\b"),
]
_STATEMENT_COMPILED = [(name, re.compile(pat, re.I)) for name, pat in STATEMENT_PATTERNS]

# headings that mean the statements have ended
_END_OF_STATEMENT = re.compile(
    r"notes?\s+(?:to|forming\s+part)\b|significant\s+accounting\s+policies\b"
    r"|independent\s+auditor|statement\s+of\s+changes\s+in\s+equity\b", re.I)

_CONSOLIDATED_CONTEXT = re.compile(r"\bconsolidated\b", re.I)
_STANDALONE_CONTEXT = re.compile(r"\bstandalone\b|\bseparate\s+financial\s+statements\b", re.I)


@dataclass
class StatementLocation:
    """Where one statement lives in one report, and how it should be read."""

    statement: str
    scope: str                       # "standalone" | "consolidated" | "unscoped"
    start_page: int
    end_page: int
    heading: str
    unit: str = "crore"
    unit_confidence: float = 0.20
    unit_source: str = ""
    period_fys: list[int] = field(default_factory=list)
    period_confidence: float = 0.20
    ocr_pages: int = 0
    flags: list[str] = field(default_factory=list)


def _page_scope(pages: list[dict[str, Any]], idx: int, heading_scope: str | None) -> str:
    """Decide standalone vs consolidated for one statement heading.

    The heading wins when it says so. Otherwise the surrounding pages decide:
    Indian reports put the whole consolidated block together, so a page that
    talks about consolidation while never mentioning standalone is consolidated.
    """
    if heading_scope:
        low = heading_scope.lower()
        return "consolidated" if low == "consolidated" else "standalone"
    window = " ".join(p["text"][:1200] for p in pages[max(0, idx - 1): idx + 1])
    has_cons = bool(_CONSOLIDATED_CONTEXT.search(window))
    has_std = bool(_STANDALONE_CONTEXT.search(window))
    if has_cons and not has_std:
        return "consolidated"
    if has_std and not has_cons:
        return "standalone"
    return "unscoped"


#: the longest a real statement heading's tail can be -- "as at 31st March, 2019"
#: is about 22 characters. The auditor's report says "...which comprise the
#: Balance Sheet as at March 31, 2019, and the Statement of Profit and Loss...",
#: which matches the same pattern but trails off into a sentence.
_MAX_HEADING_TAIL = 45


def count_money_lines(page: dict[str, Any], n_periods: int = 2) -> int:
    """How many lines on this page carry money columns.

    The real balance sheet has dozens; a page that merely mentions one in a
    sentence has almost none. This is what keeps the auditor's report from
    being mistaken for the statement it describes.
    """
    total = 0
    for line in page["text"].split("\n"):
        _, figures = split_label_and_figures(line, n_periods)
        if any(parse_number(token).ok for token in figures):
            total += 1
    return total


def _heading_hits(pages: list[dict[str, Any]]) -> list[tuple[str, str, int, str]]:
    """(statement, scope, page_number, heading_line) for every statement heading."""
    hits: list[tuple[str, str, int, str]] = []
    for idx, page in enumerate(pages):
        for line in page["text"].split("\n"):
            candidate = re.sub(r"\s+", " ", line).strip().strip("*#|:-")
            if not candidate or len(candidate) > 160:
                continue
            for name, rx in _STATEMENT_COMPILED:
                m = rx.match(candidate)
                if not m:
                    continue
                if len(candidate[m.end():].strip()) > _MAX_HEADING_TAIL:
                    continue                     # a sentence, not a heading
                scope = _page_scope(pages, idx, m.groupdict().get("scope"))
                hits.append((name, scope, page["page"], candidate))
                break
    return hits


def _statement_end(pages: list[dict[str, Any]], start_page: int, max_pages: int,
                   other_starts: set[int]) -> int:
    """A statement ends at the next statement, at the notes, or after max_pages."""
    last = start_page
    for page in pages:
        p = page["page"]
        if p <= start_page:
            continue
        if p - start_page >= max_pages or p in other_starts:
            break
        head = page["text"][:400]
        if _END_OF_STATEMENT.search(head) or not re.search(r"\d", head):
            break
        last = p
    return last


def locate_statements(pages: list[dict[str, Any]], cfg: dict[str, Any],
                      report_fy: int | None = None) -> dict[str, StatementLocation]:
    """Pick one location per statement, preferring standalone over consolidated."""
    fcfg = cfg["financials"]
    max_pages = fcfg["max_pages_per_statement"]
    hits = _heading_hits(pages)
    by_page = {p["page"]: p for p in pages}
    all_starts = {page for _, _, page, _ in hits}

    min_money_lines = fcfg["min_money_lines"]
    money_lines = {p["page"]: count_money_lines(p) for p in pages}

    out: dict[str, StatementLocation] = {}
    for name in STATEMENTS:
        candidates = [h for h in hits if h[0] == name]
        if not candidates:
            continue
        # A page that only mentions the statement carries no figures, so pages
        # that actually look like statements are preferred outright; then
        # standalone over consolidated; then the earliest page within a tier.
        tier = {"standalone": 0, "unscoped": 1, "consolidated": 2}
        substantive = [h for h in candidates if money_lines.get(h[2], 0) >= min_money_lines]
        pool = substantive or candidates
        anchor = out.get("balance_sheet")
        if name != "balance_sheet" and anchor is not None:
            # The statements are printed as a block: balance sheet, then P&L, then
            # cash flow. A "Profit and Loss" table earlier in the report is the
            # directors' financial highlights, not the statement (found on real
            # reports: it won on "earliest page" and fed the model a summary).
            after = [h for h in pool if h[2] >= anchor.start_page and tier[h[1]] <= tier[anchor.scope]]
            pool = after or pool
            _, scope, start_page, heading = min(pool, key=lambda h: (tier[h[1]], h[2] - anchor.start_page
                                                                    if h[2] >= anchor.start_page else 10_000 + h[2]))
        else:
            _, scope, start_page, heading = min(pool, key=lambda h: (tier[h[1]], h[2]))
        end_page = _statement_end(pages, start_page, max_pages, all_starts - {start_page})

        loc = StatementLocation(statement=name, scope=scope, start_page=start_page,
                                end_page=end_page, heading=heading)
        if scope == "consolidated":
            loc.flags.append("consolidated_only")
        elif scope == "unscoped":
            loc.flags.append("scope_not_stated")

        if money_lines.get(start_page, 0) < min_money_lines:
            loc.flags.append("few_money_lines_on_page")

        head_text = by_page[start_page]["text"][:1500]
        loc.unit, loc.unit_confidence = detect_unit(head_text, fcfg["default_unit"])
        loc.unit_source = _unit_source_line(head_text)
        if loc.unit_confidence < 0.5:
            loc.flags.append("unit_not_stated_assumed_" + loc.unit)
        elif loc.unit_confidence < 0.9:
            loc.flags.append("unit_ambiguous_read_as_" + loc.unit)
        loc.period_fys, loc.period_confidence = detect_period_columns(head_text, report_fy)
        if loc.period_confidence < 0.5:
            loc.flags.append("period_columns_assumed")
        if loc.period_fys and report_fy and loc.period_fys[0] != report_fy:
            loc.flags.append("first_column_is_not_the_report_year")
        if not loc.period_fys:
            # without a year per column no figure can be filed anywhere; say so
            # rather than returning an empty statement that looks merely sparse
            loc.flags.append("period_columns_unknown_figures_dropped")
        loc.ocr_pages = sum(1 for p in pages
                            if start_page <= p["page"] <= end_page and p.get("method") == "ocr")
        out[name] = loc
    return out


def _unit_source_line(text: str) -> str:
    for line in text.split("\n"):
        if re.search(r"crore|lakh|lac|million|thousand|'000|rupees", line, re.I):
            return re.sub(r"\s+", " ", line).strip()[:120]
    return ""


_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
#: "2018-19" and "FY19" both mean FY2019; common.parse_fy_label knows the spellings.
#: The span form is first *and* absorbs an "FY" prefix, so "FY 2018-19" is read
#: whole as FY2019 rather than stopping at "FY 2018" and yielding the wrong year.
_FY_LABEL = re.compile(
    r"\b(?:FY\s*'?)?(?:19|20)\d{2}\s*[-/–]\s*\d{2,4}\b"
    r"|\bFY\s*'?\d{2,4}\b", re.I)
#: prose that happens to contain years -- a regrouping note, not a column header
_NOT_A_COLUMN_HEADER = re.compile(
    r"regroup|reclassif|restat|refer\s+note|previous\s+year\s+figures|"
    r"have\s+been|comparativ\w*\s+(?:figures|information)\s+", re.I)


def _years_in(line: str) -> list[int]:
    """Fiscal years named on a line, in the order they appear, de-duplicated."""
    from bpp.common import parse_fy_label

    found: list[int] = []
    for match in _FY_LABEL.finditer(line):
        year = parse_fy_label(match.group(0))
        if year and 1990 <= year <= 2100 and year not in found:
            found.append(year)
    if found:
        return found
    for match in _YEAR.finditer(line):
        year = int(match.group(0))
        if 1990 <= year <= 2100 and year not in found:
            found.append(year)
    return found


def detect_period_columns(header_text: str, report_fy: int | None = None,
                          max_columns: int = 2) -> tuple[list[int], float]:
    """Work out which fiscal year each money column belongs to.

    Indian statements print the current year first and the comparative second,
    so the years are read off the column header and kept in the order printed.
    Three guards, each for a way this silently mis-files a whole statement:

    * prose is rejected. "previous year figures as at 31 March 2018 and 1 April
      2017 regrouped" names three years but is not a column header, and taking
      it would shift every line by one column.
    * at most two periods by default, which is what the face of an Indian
      statement carries.
    * years must descend. An ascending pair means the line was misread, so the
      confidence drops rather than the columns being quietly swapped.

    When nothing can be read, the report's own FY and the year before it are
    assumed and the caller is told the confidence is low.
    """
    for line in header_text.split("\n"):
        if not re.search(r"as\s+at|year\s+ended|period\s+ended|31\s*(?:st)?\s*march|\bFY\b", line, re.I):
            continue
        if _NOT_A_COLUMN_HEADER.search(line):
            continue
        years = _years_in(line)[:max_columns]
        if len(years) >= 2:
            descending = all(a > b for a, b in zip(years, years[1:]))
            return years, 0.95 if descending else 0.40

    # a single header year, or years split across two header lines
    ordered = _years_in(header_text)[:max_columns]
    if len(ordered) >= 2 and all(a > b for a, b in zip(ordered, ordered[1:])):
        return ordered, 0.70
    if report_fy:
        return [int(report_fy), int(report_fy) - 1], 0.20
    return [], 0.0


# --------------------------------------------------------------------------- context
#: Schedule III sub-headings. "Borrowings" means different things beneath each.
CONTEXT_PATTERNS: list[tuple[str, str]] = [
    ("non_current_assets", r"non[\s-]*current\s+assets\b"),
    ("current_assets", r"^current\s+assets\b"),
    ("equity", r"^equity\b(?!\s+and\s+liabilities)|shareholders?'?\s+funds?\b"),
    ("non_current_liabilities", r"non[\s-]*current\s+liabilit(?:y|ies)\b"),
    ("current_liabilities", r"^current\s+liabilit(?:y|ies)\b"),
    ("liabilities", r"^liabilities\b"),
    ("assets", r"^assets\b"),
    ("equity_and_liabilities", r"^equity\s+and\s+liabilit(?:y|ies)\b"),
]
_CONTEXT_COMPILED = [(name, re.compile(pat, re.I)) for name, pat in CONTEXT_PATTERNS]

#: Schedule III enumerators in front of a label: "(2)", "(iii)", "a)", "1.", "II", "B".
#: Left in place they defeat every anchored pattern: "(2) Current Assets" was not
#: seen as the current-assets heading, so borrowings lost their context (real reports).
_ENUMERATOR = re.compile(
    r"^\s*(?:(?i:\(\s*(?:[ivxlc]{1,5}|[a-z]|\d{1,2})\s*\)|(?:[ivxlc]{1,5}|[a-z]|\d{1,2})\s*[.)])"
    r"|(?:[IVX]{1,4}|[A-H])(?=\s))\s*")


def strip_enumerator(label: str) -> str:
    """Drop leading list markers, repeatedly: ``"B) (i) Borrowings"`` -> ``"Borrowings"``."""
    prev = None
    s = label
    while s != prev:
        prev = s
        s = _ENUMERATOR.sub("", s, count=1)
    return s


def detect_context(label: str) -> str | None:
    cand = strip_enumerator(re.sub(r"\s+", " ", label).strip()).strip(":-– ")
    if len(cand) > 60:
        return None
    for name, rx in _CONTEXT_COMPILED:
        if rx.search(cand):
            return name
    return None


#: a section heading printed with its own total ("Current liabilities 8,002.68 8,354.75")
_SECTION_HEADINGS: dict[str, str] = {
    "non current assets": "non_current_assets", "current assets": "current_assets",
    "non current liabilities": "non_current_liabilities", "current liabilities": "current_liabilities",
    "equity": "total_equity", "shareholders funds": "total_equity", "shareholder s funds": "total_equity",
}

#: the unlabelled line that closes a section is its total, when it equals the section's sum
_SUBTOTAL_FIELD: dict[str, str] = {
    "non_current_assets": "non_current_assets", "current_assets": "current_assets",
    "equity": "total_equity", "non_current_liabilities": "non_current_liabilities",
    "current_liabilities": "current_liabilities",
}


@dataclass
class LineItem:
    label: str
    figures: list[ParsedNumber]
    page: int
    context: str | None
    raw: str
    source: str = "text"          # "text" | "pdfplumber" | "camelot"


_BARE_TOTAL = re.compile(r"^total(?:\s+(?:rupees|rs|amount|inr))?$")
_ASSET_SIDE = {"non_current_assets", "current_assets", "assets"}
_LIABILITY_SIDE = {"equity", "non_current_liabilities", "current_liabilities", "liabilities",
                   "equity_and_liabilities"}


def _adds_up(value: float | None, total: float, n_items: int) -> bool:
    if value is None:
        return False
    tol = max(0.005 * max(abs(value), abs(total)), 0.011 * max(n_items, 1))
    return abs(value - total) <= tol


def _join_following_figures(lines: list[str], i: int, label: str, n_periods: int) -> tuple[list[str], int]:
    """The figures of a row printed on the lines after its label (common in PDF text of
    rupee statements: ``Revenue from Operations 19`` / ``27,053,549`` / ``32,663,618``)."""
    if not label or len(label) > 90 or label.rstrip().endswith(":"):
        return [], i
    got: list[str] = []
    j = i
    while j < len(lines) and len(got) < n_periods and j - i < 6:
        nxt = lines[j].strip()
        if not nxt:
            j += 1
            continue
        lab2, toks2 = split_label_and_figures(nxt, n_periods)
        if lab2 or not toks2:
            break
        got.extend(toks2)
        j += 1
    if not got:
        return [], i
    return got[:n_periods], j


def parse_statement_lines(pages: list[dict[str, Any]], loc: StatementLocation) -> list[LineItem]:
    """Walk the statement's pages and turn every money line into a LineItem.

    Besides labelled lines this reads three layouts found on real reports:

    * a row whose figures sit on the lines after its label (joined to the label);
    * a section total printed with no label, or as a bare ``Total``: taken as the
      section's total only when it equals the sum of the section's rows, so a
      grand total or a stray figure is never mistaken for one;
    * a section heading that carries the section total on its own line.
    """
    n_periods = max(1, len(loc.period_fys) or 2)
    items: list[LineItem] = []
    context: str | None = None
    section_sum, section_n = 0.0, 0
    subtotals: dict[str, LineItem] = {}

    def switch(ctx: str | None) -> None:
        nonlocal context, section_sum, section_n
        if ctx != context:
            section_sum, section_n = 0.0, 0
        context = ctx

    for page in pages:
        if not (loc.start_page <= page["page"] <= loc.end_page):
            continue
        lines = [raw.rstrip() for raw in page["text"].split("\n")]
        i = 0
        while i < len(lines):
            line = lines[i]
            i += 1
            if not line.strip():
                continue
            raw = line.strip()
            label, figure_tokens = split_label_and_figures(line, n_periods)
            if not figure_tokens:
                ctx = detect_context(line)
                if ctx:
                    switch(ctx)
                    continue
                figure_tokens, i = _join_following_figures(lines, i, label, n_periods)
                if not figure_tokens:
                    continue
                raw = f"{raw} {' '.join(figure_tokens)}"
            figures = [parse_number(tok, loc.unit) for tok in figure_tokens]
            first = figures[0].printed if figures and figures[0].ok else None

            norm = _normalise_label(label) if label else ""
            if not label or _BARE_TOTAL.match(norm):
                # an unlabelled (or bare "Total") line: a section total if it adds up
                if first is None:
                    continue
                if context in _SUBTOTAL_FIELD and section_n >= 2 and _adds_up(first, section_sum, section_n):
                    subtotals[context] = LineItem(
                        label="total " + context.replace("_", " "), figures=figures, page=page["page"],
                        context=context, raw=raw, source="subtotal")
                elif label and context in _ASSET_SIDE:
                    items.append(LineItem(label="total assets", figures=figures, page=page["page"],
                                          context=context, raw=raw, source="bare_total"))
                elif label and context in _LIABILITY_SIDE:
                    items.append(LineItem(label="total equity and liabilities", figures=figures,
                                          page=page["page"], context=context, raw=raw, source="bare_total"))
                continue

            heading = _SECTION_HEADINGS.get(_heading_key(norm))
            ctx_here = detect_context(label)
            if ctx_here:
                switch(ctx_here)
            if not any(f.ok for f in figures):
                if ctx_here is None:
                    section_n += 1              # an explicitly empty row still belongs to the section
                continue
            if heading:
                # "Current liabilities 8,002.68 8,354.75": the heading carries the total
                items.append(LineItem(label="total " + _heading_key(norm), figures=figures, page=page["page"],
                                      context=context, raw=raw, source="heading_total"))
                continue
            items.append(LineItem(label=label, figures=figures, page=page["page"],
                                  context=context, raw=raw))
            if not norm.startswith("total") and first is not None:
                section_sum += first
                section_n += 1
    items.extend(subtotals.values())
    return items


# --------------------------------------------------------------------------- field mapping
#: (field, required context or None, pattern, priority). Highest priority wins.
#: Patterns are matched against the normalised label with ``re.match``.
FIELD_PATTERNS: list[tuple[str, str | None, str, int]] = [
    # --- balance sheet: assets
    ("total_assets", None, r"total\s+assets\b", 100),
    ("current_assets", None, r"total\s+current\s+assets\b", 100),
    ("non_current_assets", None, r"total\s+non[\s-]*current\s+assets\b", 100),
    ("inventories", None, r"inventor(?:y|ies)\b", 90),
    # not the cash-flow statement's opening balance or movement lines: "Cash and cash
    # equivalents at the beginning of the year" in last year's column is the year
    # before last (read that way on real reports)
    ("cash_and_equivalents", None,
     r"cash\s+and\s+cash\s+equivalents?\b(?!.*\b(?:beginning|opening|increase|decrease|change|movement)\b)", 100),
    ("cash_and_equivalents", None,
     r"cash\s+and\s+bank\s+balances?\b(?!.*\b(?:beginning|opening|increase|decrease|change|movement)\b)", 60),
    # --- balance sheet: equity
    ("total_equity", None, r"total\s+equity\b(?!\s+and\s+liabilit)", 100),
    ("total_equity", None, r"total\s+shareholders?'?\s+funds?\b", 90),
    ("equity_share_capital", None, r"(?:equity\s+)?share\s+capital\b", 80),
    ("other_equity", None, r"other\s+equity\b", 100),
    ("other_equity", None, r"reserves?\s+and\s+surplus\b", 90),
    ("retained_earnings", None, r"retained\s+earnings\b", 100),
    ("retained_earnings", None, r"surplus\s*/?\s*\(?deficit\)?\s+in\s+the\s+statement\s+of\s+profit", 80),
    ("total_equity_and_liabilities", None, r"total\s+equity\s+and\s+liabilit(?:y|ies)\b", 100),
    # "Total liabilities" is non-current + current liabilities -- a different
    # quantity from "Total equity and liabilities", which equals total assets.
    # Conflating them makes every balance-sheet check fail on a correct report.
    ("total_liabilities", None, r"total\s+liabilit(?:y|ies)\b", 100),
    # --- balance sheet: liabilities (context decides which borrowings)
    ("current_liabilities", None, r"total\s+current\s+liabilit(?:y|ies)\b", 100),
    ("non_current_liabilities", None, r"total\s+non[\s-]*current\s+liabilit(?:y|ies)\b", 100),
    ("borrowings_long_term", "non_current_liabilities", r"borrowings?\b", 100),
    ("borrowings_long_term", None, r"long[\s-]*term\s+borrowings?\b", 95),
    ("borrowings_short_term", "current_liabilities", r"borrowings?\b", 100),
    ("borrowings_short_term", None, r"short[\s-]*term\s+borrowings?\b", 95),
    # --- profit and loss
    ("revenue", None, r"revenue\s+from\s+operations\b", 100),
    # anchored: a bare "sales" prefix would swallow "Sales tax" and
    # "Sales promotion expenses", which are costs, not revenue
    ("revenue", None, r"(?:gross\s+|net\s+)?sales(?:\s*(?:and|&)\s*services)?$", 60),
    ("total_income", None, r"total\s+income\b", 100),
    ("finance_costs", None, r"finance\s+costs?\b", 100),
    ("finance_costs", None, r"interest\s+(?:expense|and\s+finance\s+charges)\b", 80),
    ("depreciation", None, r"depreciation(?:\s*,?\s*(?:depletion\s*)?and\s+amorti[sz]ation)?", 100),
    # "before taxation" is standard older Indian wording and must match too
    # Schedule III prints "Profit before exceptional items and tax" (V) above
    # "Profit before tax" (VII). Only VII is PBT: a label that still mentions the
    # exceptional items is the line before them (found on real reports, where it
    # replaced a loss of 28.85 cr with a profit of 74.50 cr).
    ("pbt", None, r"(?:profit|loss)\s*/?\s*\(?(?:loss|profit)?\)?\s*before\s+tax(?:ation)?\b(?![^\n]*exceptional)", 100),
    ("pbt", None, r"(?:profit|loss)[^\n]{0,40}before\s+tax(?:ation)?\b(?![^\n]*exceptional)", 80),
    ("pbt", None, r"(?:profit|loss)[^\n]{0,40}after\s+exceptional[^\n]{0,30}before\s+tax(?:ation)?\b", 90),
    ("pbt", None, r"(?:profit|loss)[^\n]{0,40}before\s+tax(?:ation)?\s+(?:and|but)\s+after\s+exceptional", 95),
    ("net_profit", None, r"(?:profit|loss)\s*/?\s*\(?(?:loss|profit)?\)?\s*for\s+the\s+(?:year|period)\b", 100),
    ("net_profit", None, r"(?:profit|loss)\s+after\s+tax\b", 90),
]
_FIELD_COMPILED = [(f, ctx, re.compile(p, re.I), pri) for f, ctx, p, pri in FIELD_PATTERNS]

#: canonical spellings used for the fuzzy fallback
FIELD_SYNONYMS: dict[str, list[str]] = {
    "total_assets": ["total assets"],
    "current_assets": ["total current assets"],
    "current_liabilities": ["total current liabilities"],
    "inventories": ["inventories"],
    "cash_and_equivalents": ["cash and cash equivalents"],
    "total_equity": ["total equity"],
    "other_equity": ["other equity", "reserves and surplus"],
    "retained_earnings": ["retained earnings"],
    "revenue": ["revenue from operations"],
    "finance_costs": ["finance costs"],
    "depreciation": ["depreciation and amortisation expense"],
    "pbt": ["profit before tax", "loss before tax"],
    "net_profit": ["profit for the year", "loss for the year"],
    "total_equity_and_liabilities": ["total equity and liabilities"],
}

#: which statement each field is expected to come from (used to break ties)
FIELD_STATEMENT: dict[str, str] = {
    **{f: "balance_sheet" for f in
       ["total_assets", "current_assets", "non_current_assets", "inventories",
        "cash_and_equivalents", "total_equity", "equity_share_capital", "other_equity",
        "retained_earnings", "total_equity_and_liabilities", "total_liabilities",
        "current_liabilities", "non_current_liabilities", "borrowings_long_term",
        "borrowings_short_term"]},
    **{f: "profit_and_loss" for f in
       ["revenue", "total_income", "finance_costs", "depreciation", "pbt", "net_profit"]},
}

STANDARD_FIELDS = sorted(FIELD_STATEMENT)


def _normalise_label(label: str) -> str:
    s = strip_enumerator(label).lower().replace("&", "and")
    s = re.sub(r"\(.*?\)", " ", s)                       # "(refer note 7)"
    s = re.sub(r"[^a-z0-9\s/-]", " ", s)
    s = re.sub(r"\s+-+\s+|^-+\s*|\s*-+$", " ", s)          # "TOTAL - ASSETS", "- Borrowings"
    s = re.sub(r"\bamortisation\b", "amortization", s)
    return re.sub(r"\s+", " ", s).strip()


def _heading_key(norm: str) -> str:
    return re.sub(r"\s+", " ", norm.replace("-", " ").replace("'", " ")).strip()


def match_field(label: str, context: str | None = None,
                fuzzy_threshold: float = 0.88) -> tuple[str, float, str] | None:
    """Map one statement label to a standard field.

    Returns ``(field, score, how)``. Patterns are tried first because they cope
    with the wording Indian statements actually use; fuzzy similarity is the
    fallback for a label the patterns have not seen, and its score is kept so a
    weak match can be routed to the spot-check sheet rather than trusted.
    """
    norm = _normalise_label(label)
    if not norm:
        return None

    best: tuple[str, float, str] | None = None
    best_priority = -1
    for fieldname, req_ctx, rx, priority in _FIELD_COMPILED:
        if not rx.match(norm):
            continue
        if req_ctx is not None and context != req_ctx:
            continue
        # a pattern that names its context explicitly outranks a bare one
        effective = priority + (5 if req_ctx else 0)
        if effective > best_priority:
            best_priority = effective
            best = (fieldname, 1.0 if priority >= 90 else 0.85, "pattern")
    if best:
        return best

    first_word = norm.split(" ", 1)[0]
    for fieldname, spellings in FIELD_SYNONYMS.items():
        for spelling in spellings:
            # the first word must agree (OCR slips allowed): "Other current liabilities"
            # scored 0.90 against "total current liabilities" and replaced the total with
            # one of its components on real reports
            if difflib.SequenceMatcher(None, first_word, spelling.split(" ", 1)[0]).ratio() < 0.75:
                continue
            score = difflib.SequenceMatcher(None, norm, spelling).ratio()
            if score >= fuzzy_threshold and (best is None or score > best[1]):
                best = (fieldname, round(score, 3), "fuzzy")
    return best


@dataclass
class FieldHit:
    field: str
    fy: int
    value: float                  # ₹ crore
    column: int                   # 0 = current year column
    page: int
    statement: str
    label: str
    match_how: str
    match_score: float
    parse_confidence: float
    parse_flags: list[str] = field(default_factory=list)
    unit: str = "crore"
    printed: float | None = None


def map_statement(items: Iterable[LineItem], loc: StatementLocation) -> list[FieldHit]:
    """Turn the line items of one statement into field hits, one per column."""
    hits: list[FieldHit] = []
    for item in items:
        matched = match_field(item.label, item.context)
        if not matched:
            continue
        fieldname, score, how = matched
        if item.source in ("subtotal", "bare_total", "heading_total"):
            # read from layout, not from a printed label: an explicit label outranks it
            score, how = min(score, 0.9), item.source
        for col, parsed in enumerate(item.figures):
            if not parsed.ok or col >= len(loc.period_fys):
                continue
            hits.append(FieldHit(
                field=fieldname, fy=loc.period_fys[col], value=parsed.value, column=col,
                page=item.page, statement=loc.statement, label=item.label,
                match_how=how, match_score=score,
                parse_confidence=parsed.confidence, parse_flags=list(parsed.flags),
                unit=loc.unit, printed=parsed.printed,
            ))
    return hits


#: fields that may also be read off a statement other than their own
_CROSS_STATEMENT_OK = {("cash_and_equivalents", "cash_flow")}


def best_hits(hits: Iterable[FieldHit]) -> dict[tuple[str, int], FieldHit]:
    """Keep one hit per (field, fy): the most trustworthy, then the earliest page.

    Statements repeat figures (a total restated in the notes, a running header),
    so a rule is needed. Preferring the expected statement stops the cash-flow
    statement's closing cash line from overriding the balance sheet's.

    A figure from the wrong statement is dropped altogether, except closing cash
    from the cash flow statement: on real reports the cash flow's "Interest
    expense" adjustment became finance costs and "Long term borrowings taken /
    (repaid)" became long-term borrowings whenever the statement itself had not
    been found - a movement filed as a balance.
    """
    chosen: dict[tuple[str, int], FieldHit] = {}
    for hit in hits:
        expected = FIELD_STATEMENT.get(hit.field)
        if hit.statement != expected and (hit.field, hit.statement) not in _CROSS_STATEMENT_OK:
            continue
        key = (hit.field, hit.fy)
        current = chosen.get(key)
        if current is None or _hit_rank(hit) > _hit_rank(current):
            chosen[key] = hit
    return chosen


def _hit_rank(hit: FieldHit) -> tuple:
    expected = FIELD_STATEMENT.get(hit.field)
    return (
        1 if hit.statement == expected else 0,
        hit.match_score,
        hit.parse_confidence,
        -hit.page,
    )


# --------------------------------------------------------------------------- table extraction
def extract_table_rows(pdf_path: Path, page_numbers: Iterable[int],
                       flavours: Iterable[str] = ("pdfplumber", "camelot")) -> list[tuple[int, list[str]]]:
    """Read ruled tables off the statement pages, as ``(page_number, cells)``.

    A ruled table gives column geometry, which beats inferring columns from
    whitespace. Every dependency here is optional and every failure is
    swallowed: this path is an accuracy improvement, and the text walk in
    :func:`parse_statement_lines` is what must always work -- a scanned
    statement has no table structure to find.
    """
    rows: list[tuple[int, list[str]]] = []
    pages = sorted(set(int(p) for p in page_numbers))
    if not pages:
        return rows

    if "pdfplumber" in flavours:
        try:
            import pdfplumber
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_no in pages:
                    if page_no > len(pdf.pages):
                        continue
                    for table in pdf.pages[page_no - 1].extract_tables() or []:
                        for cells in table:
                            rows.append((page_no, [("" if c is None else str(c)) for c in cells]))
        except Exception as exc:                                  # noqa: BLE001
            log.debug("pdfplumber tables unavailable for %s: %s", pdf_path, exc)

    if rows or "camelot" not in flavours:
        return rows

    try:
        import camelot
        tables = camelot.read_pdf(str(pdf_path), pages=",".join(str(p) for p in pages),
                                  flavor="lattice")
        for table in tables:
            page_no = int(table.page)
            for cells in table.df.values.tolist():
                rows.append((page_no, [str(c) for c in cells]))
    except Exception as exc:                                      # noqa: BLE001
        log.debug("camelot tables unavailable for %s: %s", pdf_path, exc)
    return rows


def rows_to_line_items(rows: Iterable[tuple[int, list[str]]], loc: StatementLocation,
                       source: str = "pdfplumber") -> list[LineItem]:
    """Turn extracted table rows into LineItems, reusing the text-path logic.

    Column *position* is what says which year a figure belongs to, so empty
    cells are kept rather than squeezed out: a blank this year and a figure last
    year must stay in their own columns. Only columns that are empty in every
    row are dropped, and those are extractor padding rather than real periods.
    """
    n_periods = max(1, len(loc.period_fys) or 2)
    rows = [(page_no, [re.sub(r"\s+", " ", c).strip() for c in cells]) for page_no, cells in rows]
    rows = [(page_no, cells) for page_no, cells in rows if any(cells)]
    if not rows:
        return []

    width = max(len(cells) for _, cells in rows)
    rows = [(page_no, cells + [""] * (width - len(cells))) for page_no, cells in rows]
    # An always-empty column is extractor padding -- but only once there are
    # enough rows to tell padding from a period that happens to be blank here.
    keep = (list(range(width)) if len(rows) < 3
            else [i for i in range(width) if any(cells[i] for _, cells in rows)])

    items: list[LineItem] = []
    context: str | None = None
    for page_no, cells in rows:
        kept = [cells[i] for i in keep]
        if not kept:
            continue
        label, money = kept[0], kept[1:]
        ctx_here = detect_context(label)
        if ctx_here:
            context = ctx_here
        if not any(money) or not label:
            continue
        # More columns than periods means a note column is still in front of the
        # money. Drop from the left, note-like columns first.
        while len(money) > n_periods and money and is_note_reference(money[0]):
            money = money[1:]
        figure_cells = money[-n_periods:] if len(money) >= n_periods else money
        figures = [parse_number(c, loc.unit) for c in figure_cells]
        if not any(f.ok for f in figures):
            continue
        items.append(LineItem(label=label, figures=figures, page=page_no,
                              context=context, raw=" | ".join(kept), source=source))
    return items
