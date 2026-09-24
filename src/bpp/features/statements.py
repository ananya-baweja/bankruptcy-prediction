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

from bpp.features.numbers import (TO_CRORE, ParsedNumber, detect_unit, find_cell_spans, is_note_reference,
                                  parse_number, split_label_and_figures)

log = logging.getLogger(__name__)

STATEMENTS = ["balance_sheet", "profit_and_loss", "cash_flow"]

# --------------------------------------------------------------------------- locating
_SCOPE = r"(?P<scope>standalone|consolidated|separate)?\s*"

STATEMENT_PATTERNS: list[tuple[str, str]] = [
    ("balance_sheet", _SCOPE + r"balance\s+sheet\s+as\s+(?:at|on)\b"),
    ("balance_sheet", _SCOPE + r"(?:audited\s+)?balance\s+sheet$"),     # the date on the next line
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


_MONTHS = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
           r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_TAIL_DATE = re.compile(
    rf"^[\s,.:;(\-]*(?:\d{{1,2}}\s*(?:st|nd|rd|th|[\"'”’`]+)?\s*[.,/-]?\s*{_MONTHS}\.?\s*[.,/-]?\s*,?\s*\d{{2,4}}"
    rf"|{_MONTHS}\.?\s*\d{{1,2}}(?:st|nd|rd|th)?\s*,?\s*\d{{4}}"
    rf"|\d{{1,2}}\s*[./-]\s*\d{{1,2}}\s*[./-]\s*\d{{2,4}})", re.I)
_TAIL_REST = re.compile(
    r"^[\s,.:;()\[\]\-–|`₹*]*(?:(?:all|amounts?|figures?|fig|rs|rupees|inr|in|lakhs?|lacs?|crores?|millions?"
    r"|thousands?|hundreds?|unless|otherwise|stated|except|per|share|data|and|standalone|audited)\b"
    r"[\s,.:;()\[\]\-–|`₹*]*)*$",
    re.I)


#: what a balance-sheet page says; a directors' report's "Balance carried forward to / Balance
#: Sheet" line wrapped into a bare heading and won as the earliest page (a real report)
_BALANCE_SHEET_WORDS = [re.compile(p, re.I) for p in (
    r"equity\s+and\s+liabilit", r"share\s*holders?['’`]?\s*funds?", r"non[\s-]*current\s+assets",
    r"non[\s-]*current\s+liabilit", r"(?<!non-)(?<!non )\bcurrent\s+assets", r"(?<!non-)(?<!non )\bcurrent\s+liabilit",
    r"total\s+assets", r"other\s+equity|reserves\s*(?:and|&)\s*surplus", r"property,?\s+plant|fixed\s+assets")]


def _reads_like_balance_sheet(text: str) -> bool:
    return sum(1 for p in _BALANCE_SHEET_WORDS if p.search(text)) >= 3


_BARE_HEADING = re.compile(r"(?:(?:standalone|consolidated|separate)\s+)?(?:audited\s+)?balance\s+sheet", re.I)
_CONTENTS = re.compile(r"\bcontents\b|\bindex\b", re.I)

#: words that make a heading's tail a sentence ("...which comprise the Balance Sheet as at")
_SENTENCE = re.compile(r"\b(?:included|which|comprise\w*|referred|annexed|forming|these|dealt|read\s+with"
                       r"|and\s+the|they|been)\b", re.I)
_PERIOD_PREAMBLE = re.compile(r"^(?:for\s+the\s+(?:financial\s+)?(?:year|period)\s+end(?:ed|ing)?(?:\s+on)?)\s*",
                              re.I)


def _heading_tail_ok(tail: str, heading: str = "") -> bool:
    """What follows a statement's name must be a date and a unit caption, not a sentence.

    A date plus caption may run long ("as at 31 March 2019 (All amounts in lacs,
    unless stated otherwise)" was rejected by the length rule on a real report);
    anything else keeps the old length limit.
    """
    t = tail.strip()
    if not t:
        return True
    if _SENTENCE.search(t):
        return False
    # a contents line: the name, maybe its date, then a page number ("Cash Flow
    # Statement 144", "Balance Sheet as at 31 March, 2020 67") - chosen over the
    # real statements on real reports because address pages count as figures
    if re.fullmatch(r"\d{1,3}", t):
        as_at = bool(re.search(r"\bas\s+(?:at|on)$", heading.strip(), re.I))
        return as_at and bool(re.fullmatch(r"[1-9]|[12]\d|3[01]", t))    # "as at 31" / "March, 2019"
    core = _PERIOD_PREAMBLE.sub("", t)
    m = _TAIL_DATE.match(core)
    if m and re.fullmatch(r"\s*\d{1,3}\s*", core[m.end():]):
        return False
    if m and _TAIL_REST.match(core[m.end():]):
        return True
    return len(t) <= _MAX_HEADING_TAIL


def _heading_hits(pages: list[dict[str, Any]]) -> list[tuple[str, str, int, str]]:
    """(statement, scope, page_number, heading_line) for every statement heading."""
    hits: list[tuple[str, str, int, str]] = []
    for idx, page in enumerate(pages):
        for line in page["text"].split("\n"):
            candidate = re.sub(r"\s+", " ", line).strip().strip("*#|:-")
            candidate = re.sub(r"^[A-Z]{1,3}\s+(?=(?:standalone\s+)?(?:balance|statement|profit|cash)\b)", "",
                               candidate, flags=re.I)
            if not candidate or len(candidate) > 160:
                continue
            for name, rx in _STATEMENT_COMPILED:
                m = rx.match(candidate)
                if not m:
                    continue
                if not _heading_tail_ok(candidate[m.end():], m.group(0)):
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
        # A heading that is only the statement's name ("Balance Sheet") is also how
        # a contents page and a running header read. It stands only on a page that
        # is not a contents page and carries figures like the other candidates do.
        best_lines = max((money_lines.get(h[2], 0) for h in candidates), default=0)
        substantive = [h for h in substantive
                       if not _BARE_HEADING.fullmatch(h[3])
                       or (not _CONTENTS.search(by_page[h[2]]["text"])
                           and money_lines.get(h[2], 0) >= 0.5 * best_lines
                           and _reads_like_balance_sheet(by_page[h[2]]["text"]))]
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
        _read_unit(loc, by_page[start_page]["text"], fcfg["default_unit"])
        loc.period_fys, loc.period_confidence = detect_period_columns(head_text, report_fy)
        if name == "balance_sheet" and _has_opening_column(by_page[start_page]["text"], loc.period_fys):
            loc.period_fys = loc.period_fys + [OPENING_COLUMN]
            loc.flags.append("opening_balance_sheet_column_skipped")
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

    _units_from_rest_of_report(out, pages)
    for loc in out.values():
        if loc.unit_confidence < 0.5:
            loc.flags.append("unit_not_stated_assumed_" + loc.unit)
        elif loc.unit_confidence < 0.9:
            loc.flags.append("unit_ambiguous_read_as_" + loc.unit)
    return out


def _read_unit(loc: StatementLocation, text: str, default: str) -> None:
    """The unit printed on the statement's own first page, wherever the text layer put it."""
    head_text = text[:1500]
    loc.unit, loc.unit_confidence = detect_unit(head_text, default)
    loc.unit_source = _unit_source_line(head_text)
    # a bare "Rs. Rs." column header yields to a scale caption printed elsewhere on the page
    header_only = loc.unit == "rupee" and loc.unit_confidence < 0.9
    if loc.unit_confidence >= 0.5 and not header_only:
        return
    # The caption is printed top right, but many PDFs' text layers put it after the
    # signature block ("(Rs in Lakh)" as the page's last line) or inside the heading line
    # ("Balance sheet as at 31 March 2019 (` in Lakhs)"); read a caption anywhere on the
    # page before assuming the default (lakh statements were read as crore, 100x too
    # large, on real reports)
    caption = _caption_lines(text)
    if caption:
        unit, conf = detect_unit(caption, default)
        if conf >= 0.5 and not (header_only and unit == "rupee"):
            loc.unit, loc.unit_confidence = unit, min(conf, 0.9)
            loc.unit_source = _unit_source_line(caption) or caption.split("\n")[0][:120]
            loc.flags.append("unit_caption_found_below_statement")
            return
    if header_only:
        return
    decoded = _decoded_caption_unit(text)
    if decoded:
        loc.unit, loc.unit_confidence, loc.unit_source = decoded[0], 0.8, decoded[1]
        loc.flags.append("unit_caption_decoded_from_shifted_font")


def _units_from_rest_of_report(out: dict[str, StatementLocation], pages: list[dict[str, Any]]) -> None:
    """A statement with no caption of its own takes its unit from the rest of the report.

    A report prints its statements in one unit, but often captions only some of them
    (the profit and loss says "(Amount in Lakhs)", the balance sheet nothing). Failing
    that, the accounting-policy note says it ("presented in lakhs of Indian rupees",
    "rounded to the nearest lakhs"). Both were found on real reports read as crore.
    """
    captioned = [loc for loc in out.values() if loc.unit_confidence >= 0.5]
    for loc in out.values():
        if loc.unit_confidence >= 0.5:
            continue
        same = [c for c in captioned if c.scope == loc.scope]
        if same:
            src = min(same, key=lambda c: abs(c.start_page - loc.start_page))
            loc.unit, loc.unit_confidence = src.unit, min(src.unit_confidence, 0.75)
            loc.unit_source = src.unit_source
            loc.flags.append(f"unit_from_{src.statement}")
            continue
        found = report_unit(pages, loc.start_page)
        if found:
            loc.unit, loc.unit_confidence, loc.unit_source, how = found
            loc.flags.append(how)


_SCALE_WORD = r"(?:\b(?:crores?|lakhs?|lacs?|hundreds?|millions?|thousands?|billions?)\b|['’]000)"

#: a line that is only a unit caption: "(Rs in Lakh)", "(All amounts are in Rs. Lakhs, unless
#: otherwise stated)", "(Amount in INR lakhs)", "(Rupees in crores)"
_CAPTION_LINE = re.compile(r"^[^\n\d]{0,70}?" + _SCALE_WORD + r"[^\n\d]{0,80}\)?\s*$", re.I)
#: a bracketed caption inside a longer line: "Balance sheet as at 31 March 2019 ( ` in Lakhs)"
_CAPTION_GROUP = re.compile(r"\([^()\d\n]{0,60}?" + _SCALE_WORD + r"[^()\d\n]{0,90}\)?", re.I)
#: a bracketed caption that names the currency only: "(Amount in `)", "(Amount in Rs.)", "(In Rs.)"
_RUPEE_CAPTION = re.compile(
    r"\(\s*(?:(?:all\s+)?(?:amounts?|figures?)\s+(?:are\s+)?)?in\s+(?:rs\.?|₹|`|inr|rupees|indian\s+rupees)\s*\)",
    re.I)


def _caption_lines(text: str) -> str:
    out: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if len(line) <= 130 and _CAPTION_LINE.match(line):
            out.append(line)
            continue
        out += [m.group(0) for m in _CAPTION_GROUP.finditer(line)]
        out += [m.group(0) for m in _RUPEE_CAPTION.finditer(line)]
    return "\n".join(out)


_UNIT_NAMES = {"lakh": "lakh", "lakhs": "lakh", "lac": "lakh", "lacs": "lakh", "crore": "crore",
               "crores": "crore", "thousand": "thousand", "thousands": "thousand", "hundred": "hundred",
               "hundreds": "hundred", "million": "million", "millions": "million", "rupee": "rupee",
               "rupees": "rupee"}

#: the scale after "in" once a shifted caption is decoded: "inLakhs", "inINRLakhs"
_SHIFTED_UNIT = re.compile(r"in\s*(?:inr|rs\.?|₹|`)?\s*(lakhs?|lacs?|crores?|thousands?|millions?|hundreds?)", re.I)


def _decoded_caption_unit(text: str) -> tuple[str, str] | None:
    """A caption set in a font whose text layer is shifted 29 code points down.

    Subset fonts without a Unicode map come out as glyph numbers: "(`LQ/DNKV" is
    "(`in Lakhs" and "$PRXQWLQ,15/DNKV" is "Amount in INR Lakhs" (two real reports).
    Only short lines with no lower-case letters are tried, and only a decoded
    "in <scale>" counts.
    """
    for line in text.split("\n"):
        line = line.strip()
        if not 4 <= len(line) <= 40 or re.search(r"[a-z]", line):
            continue
        decoded = "".join(chr(ord(c) + 29) if "!" <= c <= "]" else c for c in line)
        m = _SHIFTED_UNIT.search(decoded)
        if m:
            return _UNIT_NAMES[m.group(1).lower()], f"{line} (decoded: {decoded})"
    return None


#: the accounting-policy note's statement of the unit
_POLICY_UNIT = re.compile(
    r"\brounded\s+(?:off\s+)?(?:to|in)\s+(?:the\s+)?(?:nearest\s+)?(?:of\s+)?(?:the\s+)?"
    r"(?:(?:₹|rs\.?|inr|indian\s+rupees?|rupees?)\s+)?(?:in\s+)?(?P<u1>lakhs?|lacs?|crores?|thousands?|hundreds?"
    r"|millions?|rupees?)\b"
    r"|\b(?:presented|expressed|stated|reported)\s+in\s+(?:(?:₹|rs\.?|inr|indian\s+rupees?|rupees?)\s+(?:in\s+)?)?"
    r"(?P<u2>lakhs?|lacs?|crores?|thousands?|millions?)\b", re.I)
#: a table caption in the notes: "Rs. In Lakhs", "(₹ in crore)", "(Amount in Lakhs)"
_NOTE_CAPTION = re.compile(r"(?:₹|`|\brs\b\.?|\binr\b|\brupees\b|\bamounts?\b)\s*(?:in\s+)?"
                           r"\b(lakhs?|lacs?|crores?|thousands?|millions?)\b", re.I)


def report_unit(pages: list[dict[str, Any]], start_page: int, ahead: int = 60,
                min_captions: int = 3) -> tuple[str, float, str, str] | None:
    """The unit the notes after a statement use: ``(unit, confidence, source, flag)`` or None.

    The accounting-policy sentence decides when there is one; otherwise the notes'
    own table captions, when at least ``min_captions`` of them agree (80%).
    """
    policy: dict[str, int] = {}
    first: dict[str, str] = {}
    notes: dict[str, int] = {}
    for page in pages:
        if not start_page - 1 <= page["page"] <= start_page + ahead:
            continue
        text = re.sub(r"\s+", " ", page["text"])
        for m in _POLICY_UNIT.finditer(text):
            unit = _UNIT_NAMES[(m.group("u1") or m.group("u2")).lower()]
            policy[unit] = policy.get(unit, 0) + 1
            first.setdefault(unit, f"p{page['page']}: {m.group(0)}")
        for m in _NOTE_CAPTION.finditer(text):
            unit = _UNIT_NAMES[m.group(1).lower()]
            notes[unit] = notes.get(unit, 0) + 1
    if policy:
        unit, n = max(policy.items(), key=lambda kv: kv[1])
        if n >= 2 / 3 * sum(policy.values()):
            return unit, 0.75, first[unit][:120], "unit_from_accounting_policy_note"
    if notes:
        unit, n = max(notes.items(), key=lambda kv: kv[1])
        if n >= min_captions and n >= 0.8 * sum(notes.values()):
            return unit, 0.7, f"{n} note captions in {unit}", "unit_from_note_captions"
    return None


def _unit_source_line(text: str) -> str:
    for line in text.split("\n"):
        if re.search(r"crore|lakh|lac|million|thousand|hundred|'000|rupees", line, re.I):
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
#: the column of a balance sheet that is read but filed under no year (see _has_opening_column)
OPENING_COLUMN = 0
#: "As at 1st April, 2016", "April 1, 2016", "01.04.2016": the Ind AS transition date
_OPENING_DAY = re.compile(r"\b0?1\s*(?:st)?[\s-]*(?:of\s+)?apr(?:il)?\b|\bapr(?:il)?[\s-]*0?1\s*(?:st)?\b|\b0?1[./-]0?4[./-]",
                          re.I)
_OPENING_YEAR = re.compile(r"\s*[,.'’-]?\s*((?:19|20)?\d{2})\b")
#: a sentence about the transition is not a column header
_TRANSITION_PROSE = re.compile(r"adopt|transition|with\s+effect|w\.\s*e\.\s*f|effective|\bfrom\b", re.I)


def _money_cell(token: str, kind: str) -> bool:
    return kind == "nil" or "," in token or "(" in token or bool(re.search(r"\.\d{1,2}\)?$", token))


def _has_opening_column(text: str, period_fys: list[int]) -> bool:
    """A balance sheet with a third column: the opening balance sheet of the Ind AS transition.

    The first Ind AS reports (FY2017 and FY2018) print "As at 31 March 2018 | 31 March
    2017 | 1 April 2016". Read as two columns, the last two figures were taken, so every
    line filed last year's figure as this year's and the opening balance as last year's
    (76 reports of the full cohort). The opening date must be in the header, a year before
    the comparative year, and most figure lines must carry three money cells.
    """
    if len(period_fys) != 2 or period_fys[0] - period_fys[1] != 1:
        return False
    opening_year = str(period_fys[1] - 1)
    # the header lines joined: the day, the month and the year are often split over
    # lines ("As at 1st" / "April, 2015"), or the dates are on one line and the years
    # on the next ("As at 31 March, As at 31 March, As at 1 April," / "2018 2017 2016")
    header = " ".join(ln.strip() for ln in text[:1500].split("\n")
                      if len(ln) <= 160 and not _NOT_A_COLUMN_HEADER.search(ln)
                      and not _TRANSITION_PROSE.search(ln))
    found = False
    for m in _OPENING_DAY.finditer(header):
        year = _OPENING_YEAR.match(header, m.end())
        if year and len(year.group(1)) == 4 and year.group(1) != opening_year:
            nearby = re.findall(r"\b(?:19|20)\d{2}\b", header[m.end():m.end() + 40])
            found = opening_year in nearby[:3]          # "1 April, 2018 2017 2016"
        else:
            found = bool(year) and year.group(1)[-2:] == opening_year[-2:]
        if found:
            break
    if not found:
        return False
    three = two = 0
    for line in text.split("\n"):
        trailing = 0
        for token, kind, _, _ in reversed(find_cell_spans(line)):
            if not _money_cell(token, kind):
                break
            trailing += 1
        if trailing >= 3:
            three += 1
        elif trailing == 2:
            two += 1
    return three >= 4 and three >= two


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
    ("equity", r"^equity\b(?!\s+and\s+liabilities)|share\s*ho[il1]ders?['’]?\s*funds?\b"),
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
_ROMAN = r"(?=[ivxIVX])[xX]{0,3}(?:[iI][xX]|[iI][vV]|[vV]?[iI]{0,3})"
_ENUMERATOR = re.compile(
    rf"^\s*(?:\(\s*(?:{_ROMAN}|[A-Za-z]|\d{{1,2}})\s*\)"          # (iii) (a) (2)
    rf"|(?:{_ROMAN}|[A-Za-z]|\d{{1,2}})\s*[.)]"                     # iii) a. 2.
    rf"|(?:{_ROMAN}|[A-Ha-h]|\d{{1,2}})(?=\s+[A-Za-z]))\s*")       # III CURRENT / a Inventories / 2 Current


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
_GRAND_TOTAL = re.compile(r"^total\s+(?:assets|equity\s+and\s+liabilit(?:y|ies)|liabilit(?:y|ies)\s+and\s+equity)\b")
#: signature-block lines under a statement, which carry numbers but are not rows
_FOOTER = re.compile(r"^(?:din|m\s*no|membership|firm\s*reg|frn|date|place|pan|cin|udin|icai)\b")
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


def _closes_side(lines: list[str], i: int, context: str | None, n_periods: int) -> bool:
    """Whether a bare "Total" at line ``i - 1`` ends its side of the balance sheet.

    The next thing printed is the other side's heading, or no more figures at all:
    then it is the side's grand total even if it equals the rows above it (which
    happens when a section heading was lost and its rows ran on).
    """
    for nxt in lines[i:]:
        if not nxt.strip():
            continue
        ctx = detect_context(nxt)
        if ctx:
            return (context in _ASSET_SIDE) != (ctx in _ASSET_SIDE)
        _, toks = split_label_and_figures(nxt, n_periods)
        if any(parse_number(t).ok for t in toks):
            return False
    return True


def parse_statement_lines(pages: list[dict[str, Any]], loc: StatementLocation) -> list[LineItem]:
    """Walk the statement's pages and turn every money line into a LineItem.

    Besides labelled lines this reads four layouts found on real reports:

    * a row whose figures sit on the lines after its label (joined to the label);
    * a section total printed with no label, or as a bare ``Total``: taken as the
      section's total only when it equals the sum of the section's rows, so a
      grand total or a stray figure is never mistaken for one;
    * a section heading that carries the section total on its own line;
    * sections printed with no total at all (the pre-Ind AS format prints only
      each side's grand total): the section's rows are summed, and the sums are
      kept only when they reproduce the balance sheet - non-current + current
      assets = total assets, equity + non-current + current liabilities = total
      equity and liabilities - column by column.
    """
    n_periods = max(1, len(loc.period_fys) or 2)
    items: list[LineItem] = []
    context: str | None = None
    sums: dict[str, list[float]] = {}          # per section, per column (as printed)
    counts: dict[str, int] = {}
    subtotals: dict[str, LineItem] = {}
    candidates: dict[str, list[tuple[LineItem, int]]] = {}   # unlabelled totals, with the row count then
    last_page = loc.start_page

    def add_row(figures: list[ParsedNumber]) -> None:
        if context not in _SUBTOTAL_FIELD:
            return
        acc = sums.setdefault(context, [0.0] * n_periods)
        for col in range(min(n_periods, len(figures))):
            f = figures[col]
            if f.ok and f.printed is not None:
                acc[col] += f.printed
        counts[context] = counts.get(context, 0) + 1

    for page in pages:
        if not (loc.start_page <= page["page"] <= loc.end_page):
            continue
        last_page = page["page"]
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
                    context = ctx
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
                n = counts.get(context, 0)
                closes_side = bool(label) and _closes_side(lines, i, context, n_periods)
                if context in _SUBTOTAL_FIELD and n >= 2 and not closes_side \
                        and _adds_up(first, sums[context][0], n):
                    candidates.setdefault(context, []).append((LineItem(
                        label="total " + context.replace("_", " "), figures=figures, page=page["page"],
                        context=context, raw=raw, source="subtotal"), n))
                elif label and context in _ASSET_SIDE:
                    items.append(LineItem(label="total assets", figures=figures, page=page["page"],
                                          context=context, raw=raw, source="bare_total"))
                    context = None               # the side is closed: nothing below belongs to it
                elif label and context in _LIABILITY_SIDE:
                    items.append(LineItem(label="total equity and liabilities", figures=figures,
                                          page=page["page"], context=context, raw=raw, source="bare_total"))
                    context = None
                continue
            if _FOOTER.match(norm):
                continue                         # signatures: "DIN 01704145", "Membership No. 409391"

            heading = _SECTION_HEADINGS.get(_heading_key(norm))
            ctx_here = detect_context(label)
            if ctx_here:
                context = ctx_here
            if not any(f.ok for f in figures):
                if ctx_here is None and context in _SUBTOTAL_FIELD:
                    counts[context] = counts.get(context, 0) + 1   # an explicitly empty row
                continue
            if heading:
                # "Current liabilities 8,002.68 8,354.75": the heading carries the total
                items.append(LineItem(label="total " + _heading_key(norm), figures=figures, page=page["page"],
                                      context=context, raw=raw, source="heading_total"))
                subtotals.setdefault(_HEADING_CONTEXT.get(_heading_key(norm), ""), items[-1])
                continue
            items.append(LineItem(label=label, figures=figures, page=page["page"],
                                  context=context, raw=raw))
            if not norm.startswith("total"):
                add_row(figures)
            elif _GRAND_TOTAL.match(norm):
                context = None                   # "Total assets" closes its side of the sheet
    subtotals.pop("", None)
    # An unlabelled total counts only if it closes its section - no row after it.
    # "(a) Fixed assets" followed by its two rows and their unlabelled sum adds up
    # too, but more non-current rows follow it.
    grand = {}
    for it in items:
        hit = match_field(it.label, it.context)
        if hit and hit[0] in ("total_assets", "total_equity_and_liabilities") and it.figures and it.figures[0].ok:
            grand.setdefault(hit[0], it.figures[0].printed)
    siblings = {"non_current_assets": ("current_assets",), "current_assets": ("non_current_assets",),
                "equity": ("non_current_liabilities", "current_liabilities"),
                "non_current_liabilities": ("equity", "current_liabilities"),
                "current_liabilities": ("equity", "non_current_liabilities")}

    def is_grand_total(item: LineItem, ctx: str) -> bool:
        # a missed section heading leaves the next section's rows in this one, and
        # the side's grand total then "adds up" to it (real report: "2 Current assets")
        g = grand.get("total_assets" if ctx in ("non_current_assets", "current_assets")
                      else "total_equity_and_liabilities")
        v = item.figures[0].printed if item.figures and item.figures[0].ok else None
        others = any(abs(sums.get(o, [0.0])[0]) > 0 for o in siblings[ctx])
        return g is not None and v is not None and others and abs(v - g) <= 0.001 * max(abs(g), 1e-9)

    for ctx, cands in candidates.items():
        closing = [item for item, n in cands if n == counts.get(ctx, 0) and not is_grand_total(item, ctx)]
        if closing and ctx not in subtotals:
            subtotals[ctx] = closing[-1]
            items.append(closing[-1])
    # printed totals ("Total current assets 1,234") count as the section's total too
    for it in items:
        hit = match_field(it.label, it.context)
        ctx = _FIELD_SECTION.get(hit[0]) if hit and hit[2] == "pattern" else None
        if ctx and it.source == "text":
            subtotals.setdefault(ctx, it)
    items.extend(_computed_section_totals(items, subtotals, sums, counts, n_periods, loc, last_page))
    return items


_FIELD_SECTION = {"current_assets": "current_assets", "non_current_assets": "non_current_assets",
                  "current_liabilities": "current_liabilities",
                  "non_current_liabilities": "non_current_liabilities", "total_equity": "equity"}


_HEADING_CONTEXT = {"non current assets": "non_current_assets", "current assets": "current_assets",
                    "non current liabilities": "non_current_liabilities",
                    "current liabilities": "current_liabilities", "equity": "equity",
                    "shareholders funds": "equity", "shareholder s funds": "equity"}


def _computed_section_totals(items: list[LineItem], subtotals: dict[str, LineItem],
                             sums: dict[str, list[float]], counts: dict[str, int],
                             n_periods: int, loc: StatementLocation, page: int) -> list[LineItem]:
    """Section totals built by adding the section's rows, kept only where they balance."""
    missing = [ctx for ctx in _SUBTOTAL_FIELD if ctx not in subtotals and counts.get(ctx, 0) >= 1]
    if not missing:
        return []

    def printed(item: LineItem | None) -> list[float | None]:
        if item is None:
            return [None] * n_periods
        vals = [f.printed if f.ok else None for f in item.figures[:n_periods]]
        return vals + [None] * (n_periods - len(vals))

    grand: dict[str, LineItem] = {}
    for it in items:
        hit = match_field(it.label, it.context)
        if hit and hit[0] in ("total_assets", "total_equity_and_liabilities"):
            grand.setdefault(hit[0], it)
    ta, tel = printed(grand.get("total_assets")), printed(grand.get("total_equity_and_liabilities"))

    section: dict[str, list[float | None]] = {}
    for ctx in _SUBTOTAL_FIELD:
        if ctx in subtotals:
            section[ctx] = printed(subtotals[ctx])
        elif ctx in missing:
            section[ctx] = list(sums.get(ctx, [0.0] * n_periods))
    sides = {"assets": ("non_current_assets", "current_assets"),
             "liabilities": ("equity", "non_current_liabilities", "current_liabilities")}
    decimals = any("." in f.raw for it in items for f in it.figures if f.ok)
    half_step = 0.005 if decimals else 0.5
    out: list[LineItem] = []
    keep: dict[str, list[bool]] = {ctx: [False] * n_periods for ctx in missing}
    for col in range(n_periods):
        side_sum = {}
        for side, parts in sides.items():
            vals = [section.get(ctx, [None] * n_periods)[col] for ctx in parts]
            side_sum[side] = None if any(v is None for v in vals) else sum(vals)
        targets = {"assets": ta[col] if ta[col] is not None else tel[col],
                   "liabilities": tel[col] if tel[col] is not None else ta[col]}
        for side, parts in sides.items():
            target = targets[side]
            if target is None:            # no grand total printed: the two sides must agree
                other = side_sum["liabilities" if side == "assets" else "assets"]
                target = other
            got = side_sum[side]
            if got is None or target is None:
                continue
            # A summed total is a number we build, so only rounding may separate it
            # from the printed total: half a unit of the last printed digit per row.
            # A relative tolerance would pass a sheet missing a small row, and a
            # current-liabilities total short by 30% (seen in testing).
            n_rows = sum(counts.get(ctx, 0) for ctx in parts) + 1
            if abs(got - target) <= max(half_step * n_rows, 1e-6 * abs(target)):
                for ctx in parts:
                    if ctx in keep:
                        keep[ctx][col] = True
    factor = TO_CRORE.get(loc.unit, 1.0)
    for ctx, cols in keep.items():
        if not any(cols):
            continue
        figures = []
        for col in range(n_periods):
            v = section[ctx][col]
            if cols[col] and v is not None:
                figures.append(ParsedNumber(value=round(v * factor, 9), raw="sum of rows", printed=v,
                                            unit=loc.unit, confidence=0.9, flags=["computed_section_sum"]))
            else:
                figures.append(ParsedNumber(raw="", unit=loc.unit, is_nil=True))
        out.append(LineItem(label="total " + ctx.replace("_", " "), figures=figures, page=page,
                            context=ctx, raw="sum of the section's rows", source="section_sum"))
    return out


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
    s = strip_enumerator(label).lower().replace("&", "and").replace("'", "").replace("’", "")
    s = re.sub(r"^sub[\s-]*total\s*[-–:]*\s*", "total ", s)     # "Sub total-Current Liabilities"
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
        if item.source in ("subtotal", "bare_total", "heading_total", "section_sum"):
            # read from layout, not from a printed label: an explicit label outranks it
            score, how = min(score, 0.9), item.source
        for col, parsed in enumerate(item.figures):
            if not parsed.ok or col >= len(loc.period_fys) or loc.period_fys[col] == OPENING_COLUMN:
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
