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


def detect_context(label: str) -> str | None:
    cand = re.sub(r"\s+", " ", label).strip().strip(":-– ")
    if len(cand) > 60:
        return None
    for name, rx in _CONTEXT_COMPILED:
        if rx.search(cand):
            return name
    return None


@dataclass
class LineItem:
    label: str
    figures: list[ParsedNumber]
    page: int
    context: str | None
    raw: str
    source: str = "text"          # "text" | "pdfplumber" | "camelot"


def parse_statement_lines(pages: list[dict[str, Any]], loc: StatementLocation) -> list[LineItem]:
    """Walk the statement's pages and turn every money line into a LineItem."""
    n_periods = max(1, len(loc.period_fys) or 2)
    items: list[LineItem] = []
    context: str | None = None
    for page in pages:
        if not (loc.start_page <= page["page"] <= loc.end_page):
            continue
        for raw_line in page["text"].split("\n"):
            line = raw_line.rstrip()
            if not line.strip():
                continue
            label, figure_tokens = split_label_and_figures(line, n_periods)
            if not figure_tokens:
                ctx = detect_context(line)
                if ctx:
                    context = ctx
                continue
            if not label:
                continue
            ctx_here = detect_context(label)
            if ctx_here:
                context = ctx_here
            figures = [parse_number(tok, loc.unit) for tok in figure_tokens]
            if not any(f.ok for f in figures):
                continue
            items.append(LineItem(label=label, figures=figures, page=page["page"],
                                  context=context, raw=line.strip()))
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
    ("cash_and_equivalents", None, r"cash\s+and\s+cash\s+equivalents\b", 100),
    ("cash_and_equivalents", None, r"cash\s+and\s+bank\s+balances?\b", 60),
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
    ("pbt", None, r"(?:profit|loss)\s*/?\s*\(?(?:loss|profit)?\)?\s*before\s+tax(?:ation)?\b", 100),
    ("pbt", None, r"(?:profit|loss)[^\n]{0,40}before\s+(?:exceptional[^\n]{0,30})?tax(?:ation)?\b", 80),
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
    s = label.lower().replace("&", "and")
    s = re.sub(r"\(.*?\)", " ", s)                       # "(refer note 7)"
    s = re.sub(r"[^a-z0-9\s/-]", " ", s)
    s = re.sub(r"\bamortisation\b", "amortization", s)
    return re.sub(r"\s+", " ", s).strip()


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

    for fieldname, spellings in FIELD_SYNONYMS.items():
        for spelling in spellings:
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


def best_hits(hits: Iterable[FieldHit]) -> dict[tuple[str, int], FieldHit]:
    """Keep one hit per (field, fy): the most trustworthy, then the earliest page.

    Statements repeat figures (a total restated in the notes, a running header),
    so a rule is needed. Preferring the expected statement stops the cash-flow
    statement's closing cash line from overriding the balance sheet's.
    """
    chosen: dict[tuple[str, int], FieldHit] = {}
    for hit in hits:
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
