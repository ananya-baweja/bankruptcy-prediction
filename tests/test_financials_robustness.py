"""Regression tests for ways a statement can be read wrongly but plausibly.

Every case here produced a wrong number that looked right. They are kept
separate from the happy-path tests because a clean fixture cannot catch them:
real statements have blank cells, prose that mentions a scale, auditor's reports
that quote the balance sheet's title, and merged table columns.
"""

import pandas as pd
import pytest

from bpp.features.numbers import detect_unit, parse_number, split_label_and_figures
from bpp.features.ratios import ALTMAN_DISTRESS, ALTMAN_SAFE, compute_ratios
from bpp.features.statements import (StatementLocation, detect_period_columns, locate_statements,
                                     match_field, rows_to_line_items)


def _location(period_fys=(2019, 2018)):
    return StatementLocation("balance_sheet", "standalone", 1, 1, "heading",
                             period_fys=list(period_fys))


# --------------------------------------------------------------------------- blank columns
@pytest.mark.parametrize("line,expected", [
    # a blank column must stay a column, or the note reference becomes money
    ("Inventories 8 - -", ["-", "-"]),
    ("Borrowings 14 500.00 -", ["500.00", "-"]),
    ("Deferred tax assets 5 - 12.34", ["-", "12.34"]),
    ("Trade receivables 6 Nil 45.00", ["Nil", "45.00"]),
])
def test_an_empty_money_column_still_counts_as_a_column(line, expected):
    assert split_label_and_figures(line, 2)[1] == expected


def test_a_note_reference_is_never_read_as_this_years_figure():
    _, figures = split_label_and_figures("Inventories 8 - -", 2)
    assert all(parse_number(token).is_nil for token in figures)


def test_a_nil_dash_does_not_flip_the_next_columns_sign():
    """``- 12.34`` is a blank column beside a positive figure, not minus 12.34."""
    _, figures = split_label_and_figures("Deferred tax assets 5 - 12.34", 2)
    assert parse_number(figures[1]).printed == pytest.approx(12.34)


def test_a_hyphenated_label_is_not_mistaken_for_a_blank_column():
    assert split_label_and_figures("Non-current assets", 2) == ("Non-current assets", [])


def test_a_table_row_with_a_trailing_blank_keeps_its_columns():
    items = rows_to_line_items([(1, ["Cash and cash equivalents", "9", "88.12", ""])], _location())
    figures = items[0].figures
    assert figures[0].printed == pytest.approx(88.12)
    assert figures[1].is_nil


# --------------------------------------------------------------------------- merged cells
@pytest.mark.parametrize("cell", ["500 400", "12 500", "1,234.56 1,100.00"])
def test_two_numbers_in_one_cell_are_refused(cell):
    """A merged column pair must not be concatenated into one plausible figure."""
    parsed = parse_number(cell)
    assert parsed.value is None and "multiple_numbers_in_cell" in parsed.flags


@pytest.mark.parametrize("cell", ["2,345.00 -", "1.5 B", "- 12.34"])
def test_ambiguous_cells_are_refused_rather_than_guessed(cell):
    assert parse_number(cell).value is None


def test_a_trailing_letter_is_not_turned_into_a_digit():
    """The OCR fix must not read ``1.5 B`` as 1.58."""
    assert parse_number("1.5 B").printed != pytest.approx(1.58)


# --------------------------------------------------------------------------- units
def test_a_scale_mentioned_in_prose_does_not_beat_the_caption():
    unit, confidence = detect_unit(
        "(All amounts in Rs. lakhs unless otherwise stated)\n"
        "Turnover crossed Rs. 500 crore this year.")
    assert unit == "lakh"
    assert confidence < 0.9          # two scales seen, so confidence drops


def test_a_dr_cr_column_header_is_not_a_crore():
    unit, confidence = detect_unit("Dr Cr\nBalance Sheet\n(Rs in Lakhs)")
    assert unit == "lakh" and confidence > 0.9


def test_an_unstated_scale_stays_low_confidence():
    assert detect_unit("Balance Sheet as at 31 March 2019")[1] < 0.5


# --------------------------------------------------------------------------- period columns
def test_a_regrouping_note_is_not_read_as_column_headers():
    """Three years in a sentence would shift every line by one column."""
    years, _ = detect_period_columns(
        "Balance Sheet as at 31 March 2019 (previous year figures as at "
        "31 March 2018 and 1 April 2017 regrouped)", report_fy=2019)
    assert len(years) == 2


@pytest.mark.parametrize("header", [
    "Particulars Note FY 2018-19 FY 2017-18",
    "Particulars Note 2018-19 2017-18",
    "Particulars Note As at 31 March 2019 As at 31 March 2018",
])
def test_fiscal_year_spellings(header):
    assert detect_period_columns(header, report_fy=2019)[0] == [2019, 2018]


def test_ascending_columns_lower_the_confidence():
    """Indian statements print the current year first; ascending means a misread."""
    _, confidence = detect_periods = detect_period_columns(
        "Particulars Note Year ended 31 March 2018 Year ended 31 March 2019", report_fy=2019)
    assert confidence < 0.5


# --------------------------------------------------------------------------- locating
def test_the_auditors_report_does_not_hijack_the_balance_sheet(cfg):
    """The audit opinion quotes the balance sheet's title in a sentence. The real
    statement is the page carrying the figures, not the page describing it."""
    audit = ("INDEPENDENT AUDITOR'S REPORT\nWe have audited the standalone financial statements\n"
             "Balance Sheet as at March 31, 2019, and the Statement of Profit and Loss, "
             "the Cash Flow Statement\nfor the year then ended.")
    real = ("STANDALONE BALANCE SHEET AS AT 31ST MARCH, 2019\n(Rs. in crore)\n"
            "Particulars Note As at 31 March 2019 As at 31 March 2018\n"
            + "\n".join(f"Line item {i} {i}00.00 {i}10.00" for i in range(1, 12)))
    pages = [{"page": 40, "method": "text", "text": audit},
             {"page": 48, "method": "text", "text": real}]
    assert locate_statements(pages, cfg, 2019)["balance_sheet"].start_page == 48


# --------------------------------------------------------------------------- field mapping
@pytest.mark.parametrize("label,expected", [
    # "Total liabilities" is not "Total equity and liabilities"
    ("Total liabilities", "total_liabilities"),
    ("Total equity and liabilities", "total_equity_and_liabilities"),
    # a bare "sales" prefix would swallow these costs
    ("Sales tax", None),
    ("Sales promotion expenses", None),
    ("Sales and distribution expenses", None),
    # older Indian wording
    ("Profit/(loss) before taxation", "pbt"),
    ("Loss before taxation", "pbt"),
])
def test_labels_that_used_to_map_to_the_wrong_field(label, expected):
    matched = match_field(label)
    assert (matched[0] if matched else None) == expected


def test_total_liabilities_does_not_break_the_balance_check(cfg):
    from bpp.features.financials import validate_year
    row = pd.Series({"total_assets": 1000.0, "total_liabilities": 700.0,
                     "total_equity_and_liabilities": 1000.0, "total_equity": 300.0,
                     "non_current_liabilities": 400.0, "current_liabilities": 300.0})
    assert validate_year(row, cfg)["validation_flags"] == ""


# --------------------------------------------------------------------------- ratios
def test_partial_borrowings_are_flagged_not_passed_off_as_complete():
    result = compute_ratios({"total_assets": 1000.0, "total_equity": 200.0,
                             "borrowings_long_term": 400.0})
    assert result.indicators["borrowings_partial"] is True


def test_complete_borrowings_are_not_flagged():
    result = compute_ratios({"total_assets": 1000.0, "total_equity": 200.0,
                             "borrowings_long_term": 400.0, "borrowings_short_term": 100.0})
    assert result.indicators["borrowings_partial"] is False


def test_a_reported_total_liabilities_is_preferred_for_altman_x4():
    from bpp.features.ratios import derive
    assert derive({"total_assets": 1000.0, "total_equity": 300.0,
                   "total_liabilities": 650.0})["total_liabilities"] == pytest.approx(650.0)


def test_the_altman_zone_is_not_read_off_the_shifted_scale():
    failing = dict(total_assets=1000.0, current_assets=300.0, current_liabilities=330.0,
                   total_equity=120.0, other_equity=10.0, pbt=-30.0, finance_costs=35.0)
    result = compute_ratios(failing)
    assert result.values["altman_z_em"] > ALTMAN_SAFE          # on the wrong scale: "safe"
    assert result.derived["altman_z_dprime"] < ALTMAN_DISTRESS
    assert result.altman_zone == "distress"


# --------------------------------------------------------------------------- pipeline
def test_the_xbrl_supplement_still_applies_when_no_report_yielded_anything(cfg, paths):
    """An all-scanned batch is exactly when the supplement is the only source."""
    from bpp.features.financials import resolve_figures

    xbrl = pd.DataFrame([{"firm_id": "BSE900101", "fy": 2019, "field": "total_assets",
                          "value_cr": 900.0},
                         {"firm_id": "BSE900101", "fy": 2019, "field": "total_equity",
                          "value_cr": 100.0}])
    resolved, _ = resolve_figures(pd.DataFrame(), xbrl, cfg)
    assert len(resolved) == 2
    assert set(resolved["source"]) == {"manual_xbrl"}
