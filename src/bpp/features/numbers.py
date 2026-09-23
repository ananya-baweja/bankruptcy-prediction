"""Read the numbers printed in Indian financial statements.

Indian reports state their scale in a header line -- "(Rs. in crore)", "(₹ in
lakhs)", "(All amounts in INR million)" -- and every figure in the statement is
in that scale. Getting the scale wrong is the most damaging silent error in
Phase 3: the figure still looks plausible, every ratio still computes, and the
firm simply appears 100x larger. So the scale is read explicitly, kept next to
the value, and cross-checked later in :mod:`bpp.features.financials`.

Everything is stored in **₹ crore**, the unit the rest of the project already
uses (see ``total_assets`` in ``data/manual/firm_financials.csv`` and
``docs/data_dictionary.md``).

Also handled: brackets for negatives, Indian digit grouping (12,34,567) as well
as Western (1,234,567), currency marks, footnote markers, "Nil"/"NA"/em-dash,
and the letter-for-digit substitutions Tesseract makes on scanned statements.
Where a cell is damaged beyond safe repair the parser refuses it and says why,
rather than guessing -- a wrong number is worse than a missing one, because a
missing one is recorded as missing and a wrong one is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- units
#: multiplier converting a figure *as printed* into ₹ crore
TO_CRORE: dict[str, float] = {
    "crore": 1.0,
    "lakh": 1e-2,
    "million": 1e-1,
    "billion": 1e2,
    "thousand": 1e-4,
    "rupee": 1e-7,
}

# Longest / most specific first: "lakhs" must not be shadowed, and "crore"
# must be tried before the bare-rupee pattern.
_UNIT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("crore", re.compile(r"\bcrores?\b|\bcr\.?\b|\bkoti\b", re.I)),
    ("lakh", re.compile(r"\blakh?s?\b|\blacs?\b|\blakhs\b", re.I)),
    ("billion", re.compile(r"\bbillions?\b|\bbn\b", re.I)),
    ("million", re.compile(r"\bmillions?\b|\bmn\b|\bmio\b", re.I)),
    ("thousand", re.compile(r"\bthousands?\b|'000|’000|\bin\s+000'?s?\b", re.I)),
    ("rupee", re.compile(r"\bin\s+rupees\b|\bin\s+absolute\s+(?:terms|figures)\b|"
                         r"\bamounts?\s+in\s+(?:rs\.?|₹)\s*\)|\bin\s+units\b", re.I)),
]

#: cells that mean "no figure", not "zero"
_NIL_TOKENS = frozenset({
    "", "-", "--", "---", "–", "—", "―", "−",
    "nil", "nill", "null", "na", "n.a.", "n/a", "none", "not applicable", "*", "**", "#", ".",
})

# currency marks, footnote markers and stray furniture.
# Deliberately does NOT remove internal whitespace: when a table extractor merges
# two money columns into one cell, "500 400" must be caught as two numbers rather
# than silently become 500400.
_STRIP = re.compile(
    r"(?:₹|\brs\.?|\binr\b|\brupees\b|\$|\busd\b)"     # currency
    r"|\[[^\]]*\]"                                           # [3]
    r"|[*#†‡^~ ]",                            # footnote marks, nbsp
    re.I,
)

#: two numbers left in one cell, e.g. a merged column pair
_TWO_NUMBERS = re.compile(r"\d[\d,.]*\s+[\d,.]*\d")

_WESTERN = re.compile(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?")
_INDIAN = re.compile(r"\d{1,2}(?:,\d{2})+,\d{3}(?:\.\d+)?")
_PLAIN = re.compile(r"\d+(?:\.\d+)?")

# Tesseract's usual letter-for-digit slips inside an otherwise numeric cell
_OCR_SUBS = str.maketrans({"O": "0", "o": "0", "D": "0", "I": "1", "l": "1", "|": "1",
                           "S": "5", "s": "5", "B": "8", "Z": "2", "g": "9"})


@dataclass
class ParsedNumber:
    """One figure, with everything needed to audit it months later."""

    value: float | None = None          # ₹ crore; None when the cell is blank or unusable
    raw: str = ""                       # the cell exactly as extracted
    printed: float | None = None        # the figure as printed, before unit scaling
    unit: str = "crore"
    negative_style: str | None = None   # "bracket" | "minus" | None
    is_nil: bool = False
    confidence: float = 1.0
    flags: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.value is not None

    def flag(self, name: str, penalty: float = 0.0) -> None:
        if name not in self.flags:
            self.flags.append(name)
        if penalty:
            self.confidence = max(0.0, round(self.confidence - penalty, 4))


#: a scale is only believed when it reads like a caption -- "(Rs. in lakhs)",
#: "All amounts in crore" -- rather than a word that happens to appear on the page
_SCALE_QUALIFIER = re.compile(
    r"(?:\bin\b|\bamounts?\b|\bfigures?\b|\ball\b|\bstated\b|₹|\brs\.?|\binr\b|\()"
    r"[^()\n]{0,25}$", re.I)


def detect_unit(text: str, default: str = "crore") -> tuple[str, float]:
    """Read the scale out of a header or caption.

    Returns ``(unit, confidence)``. Two traps are guarded against, because a
    wrong scale is the error that still looks plausible -- every ratio computes
    and the firm is simply 100x too large:

    * a unit word appearing in prose ("turnover crossed Rs. 500 crore") must not
      beat the caption that actually governs the statement, so the earliest
      caption-like mention wins rather than the first pattern in the list;
    * short forms ("Cr", "Mn") are only believed with a qualifier in front,
      because the "Cr" of a Dr/Cr column header is not a crore.

    When two different scales are both caption-like, confidence drops instead of
    one silently winning.
    """
    if not text:
        return default, 0.20
    hay = text.replace("’", "'")

    found: list[tuple[int, str, bool]] = []
    for name, pattern in _UNIT_PATTERNS:
        for match in pattern.finditer(hay):
            token = match.group(0).strip(". ")
            qualified = bool(_SCALE_QUALIFIER.search(hay[max(0, match.start() - 30):match.start()]))
            if len(token) <= 3 and not qualified:
                continue                       # "Cr", "Mn", "Bn" on their own prove nothing
            found.append((match.start(), name, qualified))
    if not found:
        return default, 0.20

    found.sort()
    qualified = [f for f in found if f[2]]
    pool = qualified or found
    units = {name for _, name, _ in pool}
    if not qualified:
        return pool[0][1], 0.35
    return pool[0][1], 0.95 if len(units) == 1 else 0.55


def parse_number(raw: object, unit: str = "crore") -> ParsedNumber:
    """Parse one statement cell into ₹ crore."""
    original = "" if raw is None else str(raw)
    out = ParsedNumber(raw=original, unit=unit)

    s = original.strip()
    if s.lower() in _NIL_TOKENS:
        out.is_nil = True
        return out

    negative = False
    # Brackets mean negative in every Indian statement; check before stripping.
    if re.search(r"\(\s*[\d.,OoIlSB|]+\s*\)", s):
        negative = True
        out.negative_style = "bracket"
        s = s.replace("(", "").replace(")", "")
    # A minus must touch its digits. "- 12.34" is a nil dash next to the *next*
    # column's figure, and reading it as -12.34 inverts the sign of a real number.
    elif re.match(r"^[-−–]\d", s) or re.search(r"\d[-−]$", s):
        negative = True
        out.negative_style = "minus"
        s = re.sub(r"[-−–]", "", s)

    s = _STRIP.sub("", s).strip()
    if not s or s.lower() in _NIL_TOKENS:
        out.is_nil = True
        return out

    # Two numbers in one cell: a merged column pair. Refuse -- concatenating them
    # would produce a plausible-looking figure that is off by orders of magnitude.
    if _TWO_NUMBERS.search(s):
        out.flag("multiple_numbers_in_cell", 1.0)
        return out
    s = re.sub(r"\s+", "", s)
    if not s:
        out.is_nil = True
        return out

    # OCR letter/digit slips. Only substitute a letter sitting *between* digits:
    # a trailing letter is a unit suffix ("1.5 B"), and turning it into a digit
    # would invent a number rather than read one.
    if re.search(r"[OoDIlSsBZg|]", s) and re.search(r"\d", s):
        fixed = "".join(
            ch.translate(_OCR_SUBS)
            if (0 < i < len(s) - 1 and s[i - 1] in "0123456789,." and s[i + 1] in "0123456789,.")
            else ch
            for i, ch in enumerate(s))
        if fixed != s:
            s = fixed
            out.flag("ocr_char_substitution", 0.25)

    if not re.fullmatch(r"[\d.,]+", s):
        out.flag("unparseable", 1.0)
        return out

    # Digit grouping tells us whether a comma was lost or invented by OCR.
    if "," in s:
        if _WESTERN.fullmatch(s):
            pass
        elif _INDIAN.fullmatch(s):
            out.flag("indian_grouping")
        else:
            out.flag("irregular_digit_grouping", 0.30)

    body = s.replace(",", "")
    if body.count(".") > 1:
        out.flag("multiple_decimal_points", 1.0)      # "12.34.56" -- refuse, do not guess
        return out
    if not _PLAIN.fullmatch(body):
        out.flag("unparseable", 1.0)
        return out

    try:
        printed = float(body)
    except ValueError:                                 # pragma: no cover - guarded above
        out.flag("unparseable", 1.0)
        return out

    # Statements print 0-2 decimals. More usually means a lost thousands comma.
    if "." in body and len(body.split(".", 1)[1]) > 2:
        out.flag("unexpected_decimal_precision", 0.20)

    if negative:
        printed = -printed

    out.printed = printed
    # round to paise so unit scaling does not leave binary-float residue
    out.value = round(printed * TO_CRORE.get(unit, 1.0), 9)
    return out


# A cell is either a figure or an explicit nil. Both have to be tokenised,
# because a statement line is positional: "Inventories 8 - -" has its money
# columns empty, and if the nils are invisible the note reference 8 slides into
# the money window and becomes ₹8 crore of inventory.
# The number alternative comes first so "-12.34" is a negative, while a lone
# dash (not touching a digit or a word) is a nil.
_CELL_TOKEN = re.compile(
    r"(?P<num>"
    r"\(\s*(?:₹|Rs\.?)?\s*\d[\d,]*(?:\.\d+)?\s*\)"        # (1,230.75)
    r"|[-−]?\d[\d,]*(?:\.\d+)?"                            # 1,204.55 / -88.12
    r")"
    r"|(?P<nil>"
    r"(?<![\w-])(?:-{1,3}|–|—|―)(?![\w-])"       # a standalone dash
    r"|\bnil\b|\bn\.a\.|\bn/a\b"
    r")",
    re.I,
)


def find_cell_spans(line: str) -> list[tuple[str, str, int, int]]:
    """Every money-column cell, as ``(token, kind, start, end)``.

    ``kind`` is ``"num"`` or ``"nil"``. Positions matter because one figure can
    be a substring of another (``10.00`` inside ``110.00``), so the label/figure
    split cannot be done by searching for the token text.
    """
    out: list[tuple[str, str, int, int]] = []
    for m in _CELL_TOKEN.finditer(line):
        kind = "num" if m.group("num") else "nil"
        out.append((m.group(0).strip(), kind, m.start(), m.end()))
    return out


def find_number_spans(line: str) -> list[tuple[str, int, int]]:
    """Numeric cells only, as ``(token, start, end)``."""
    return [(tok, start, end) for tok, kind, start, end in find_cell_spans(line) if kind == "num"]


def find_numbers(line: str) -> list[str]:
    """Pull the numeric cells out of a statement line, left to right.

    ``"Cash and cash equivalents 9 88.12 210.45"`` ->
    ``["9", "88.12", "210.45"]``. Brackets are kept with their number so the
    sign survives, and thousands separators are not split into two numbers.
    """
    return [tok for tok, _, _ in find_number_spans(line)]


def split_label_and_figures(line: str, n_periods: int = 2) -> tuple[str, list[str]]:
    """Split a statement line into its label and its ``n_periods`` money columns.

    Position, not appearance, decides what is money. A Schedule III line reads
    ``<label> <note ref> <this year> <last year>``, so the money is the *last*
    ``n_periods`` tokens and anything before them is a note reference. Judging
    tokens on their own shape cannot work: ``88.12`` is a plausible note
    reference (2.14-style) and a plausible figure in ₹ crore, and only its
    position distinguishes them.

    An explicitly empty column (a dash, ``Nil``) counts as a column and is
    returned as its own token, so the columns stay aligned with the years. Without
    that, ``"Borrowings 14 500.00 -"`` would read the note reference 14 as this
    year's figure and file last year's column against this year.

    Returns ``(label, figure_tokens)``. When the line carries fewer cells than
    ``n_periods`` every cell found is returned, so the caller can see the
    shortfall rather than receive silently padded columns.
    """
    spans = find_cell_spans(line)
    if not spans:
        return _tidy_label(line), []
    figures = spans[-n_periods:] if len(spans) >= n_periods else spans
    label = line[:figures[0][2]]
    # a note reference sits between the label and the money -- drop it
    label = re.sub(r"[\s.:…]*\b\d{1,2}(?:\.\d{1,2})?\s*$", "", label)
    return _tidy_label(label), [tok for tok, _, _, _ in figures]


def _tidy_label(text: str) -> str:
    text = re.sub(r"\.{2,}", " ", text)            # dotted leaders
    return re.sub(r"\s+", " ", text).strip(" .:…-–—|")


def is_note_reference(token: str) -> bool:
    """True for a bare Schedule III note reference such as ``7`` or ``2.14``.

    Only a fallback for lines whose column count could not be established --
    :func:`split_label_and_figures` is the reliable route. Commas, brackets and
    signs never appear in a note reference, so those are always figures.
    """
    t = token.strip()
    if "," in t or "(" in t or t.startswith(("-", "−")):
        return False
    return bool(re.fullmatch(r"\d{1,2}(?:\.\d{1,2})?", t))
