import pytest

from bpp.features.numbers import (detect_unit, find_numbers, is_note_reference, parse_number,
                                  split_label_and_figures)


@pytest.mark.parametrize("raw,unit,expected", [
    ("1,204.55", "crore", 1204.55),
    ("(1,230.75)", "crore", -1230.75),          # brackets are how statements write a minus
    ("12,34,567.89", "lakh", 1234567.89),       # Indian digit grouping
    ("1,234,567.89", "rupee", 1234567.89),      # Western digit grouping
    ("Rs. 88.12", "crore", 88.12),
    ("₹ 4,120.00", "crore", 4120.00),
    ("450.00*", "crore", 450.00),               # footnote marker
    ("-2,310.50", "crore", -2310.50),
])
def test_figures_are_read_as_printed(raw, unit, expected):
    assert parse_number(raw, unit).printed == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "-", "—", "Nil", "NIL", "N.A.", "n/a", "–"])
def test_nil_is_not_zero(raw):
    parsed = parse_number(raw, "crore")
    assert parsed.value is None and parsed.is_nil


@pytest.mark.parametrize("printed,unit,expected_crore", [
    (100.0, "crore", 100.0),
    (100.0, "lakh", 1.0),           # 100 lakh = 1 crore
    (100.0, "million", 10.0),
    (100.0, "thousand", 0.01),
    (10000000.0, "rupee", 1.0),
    (1.0, "billion", 100.0),
])
def test_every_unit_normalises_to_crore(printed, unit, expected_crore):
    assert parse_number(str(printed), unit).value == pytest.approx(expected_crore)


@pytest.mark.parametrize("header,expected", [
    ("(All amounts in Rs. crores, unless otherwise stated)", "crore"),
    ("(Rs. in lakhs)", "lakh"),
    ("Amount in ₹ lacs", "lakh"),
    ("(All figures in INR million)", "million"),
    ("(Amounts in ₹'000)", "thousand"),
    ("(All amounts in Rupees)", "rupee"),
])
def test_unit_is_read_from_the_header(header, expected):
    unit, confidence = detect_unit(header)
    assert unit == expected and confidence > 0.5


def test_an_unstated_unit_is_low_confidence():
    """An assumed scale must stay visible: it is the error that still looks plausible."""
    unit, confidence = detect_unit("Balance Sheet as at 31 March 2019")
    assert unit == "crore" and confidence < 0.5


@pytest.mark.parametrize("raw,flag", [
    ("12.34.56", "multiple_decimal_points"),      # refuse rather than guess
    ("abc", "unparseable"),
])
def test_damaged_cells_are_refused(raw, flag):
    parsed = parse_number(raw, "crore")
    assert parsed.value is None and flag in parsed.flags


@pytest.mark.parametrize("raw,flag,still_parses", [
    ("4,88O.11", "ocr_char_substitution", True),        # OCR read 0 as the letter O
    ("1,2045.50", "irregular_digit_grouping", True),    # a comma in the wrong place
    ("2,990.315", "unexpected_decimal_precision", True),
])
def test_suspect_cells_parse_but_lose_confidence(raw, flag, still_parses):
    parsed = parse_number(raw, "crore")
    assert (parsed.value is not None) is still_parses
    assert flag in parsed.flags and parsed.confidence < 1.0


@pytest.mark.parametrize("line,n_periods,label,figures", [
    ("Cash and cash equivalents 9 88.12 210.45", 2, "Cash and cash equivalents", ["88.12", "210.45"]),
    ("Other equity 14 (1,230.75) 320.40", 2, "Other equity", ["(1,230.75)", "320.40"]),
    ("Total assets 7,856.15 8,310.92", 2, "Total assets", ["7,856.15", "8,310.92"]),
    ("Finance costs .......... 25 610.00 480.00", 2, "Finance costs", ["610.00", "480.00"]),
    ("Total equity and liabilities", 2, "Total equity and liabilities", []),
])
def test_label_and_money_columns_are_split_by_position(line, n_periods, label, figures):
    assert split_label_and_figures(line, n_periods) == (label, figures)


def test_a_figure_that_contains_another_figure_is_not_confused():
    """``10.00`` is a substring of ``110.00``; only position can tell them apart."""
    label, figures = split_label_and_figures("Revenue from operations 2.14 10.00 110.00", 2)
    assert label == "Revenue from operations"
    assert figures == ["10.00", "110.00"]


def test_note_column_is_dropped_not_treated_as_money():
    label, figures = split_label_and_figures("Borrowings 15 4,120.00 3,900.00", 2)
    assert label == "Borrowings" and figures == ["4,120.00", "3,900.00"]


@pytest.mark.parametrize("token,expected", [
    ("7", True), ("2.14", True), ("15", True),
    ("1,230.75", False), ("(88.12)", False), ("-5", False), ("110", False),
])
def test_note_references(token, expected):
    assert is_note_reference(token) is expected


def test_thousands_separator_is_not_two_numbers():
    assert find_numbers("Total assets 7,856.15") == ["7,856.15"]
