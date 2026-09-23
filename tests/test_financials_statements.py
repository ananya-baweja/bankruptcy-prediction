import pytest

from bpp.features.financials import extract_document, validate_year
from bpp.features.statements import (best_hits, detect_context, detect_period_columns,
                                     locate_statements, map_statement, match_field,
                                     parse_statement_lines)
from bpp.synthetic import statement_pages, year_figures

import pandas as pd


def _pages(texts, first_page=10, method="text"):
    return [{"page": first_page + i, "method": method, "text": t} for i, t in enumerate(texts)]


@pytest.fixture
def report_pages():
    """A distressed firm's standalone statements, followed by a consolidated set."""
    return _pages(statement_pages("Zenthra Infraprojects Limited", 2019, True, seed=1))


# --------------------------------------------------------------------------- locating
def test_statements_are_located(cfg, report_pages):
    found = locate_statements(report_pages, cfg, report_fy=2019)
    assert set(found) == {"balance_sheet", "profit_and_loss", "cash_flow"}
    assert found["balance_sheet"].start_page == 10


def test_standalone_is_preferred_over_consolidated(cfg, report_pages):
    """CARO and the auditor's report already prefer standalone (decisions_log, 2026-09-16).
    Consolidated statements include subsidiaries, so they are the wrong denominator."""
    found = locate_statements(report_pages, cfg, report_fy=2019)
    assert found["balance_sheet"].scope == "standalone"
    assert "consolidated_only" not in found["balance_sheet"].flags


def test_consolidated_is_used_only_when_nothing_else_exists(cfg):
    only_consolidated = _pages([
        "CONSOLIDATED BALANCE SHEET AS AT 31ST MARCH, 2019\n(Rs. in crore)\n"
        "Particulars As at 31 March 2019 As at 31 March 2018\nTotal assets 900.00 950.00"])
    found = locate_statements(only_consolidated, cfg, report_fy=2019)
    assert found["balance_sheet"].scope == "consolidated"
    assert "consolidated_only" in found["balance_sheet"].flags


def test_unit_and_periods_are_read_from_the_header(cfg, report_pages):
    loc = locate_statements(report_pages, cfg, report_fy=2019)["balance_sheet"]
    assert loc.unit == "crore" and loc.unit_confidence > 0.5
    assert loc.period_fys == [2019, 2018]


@pytest.mark.parametrize("header,expected", [
    ("Particulars Note As at 31 March 2019 As at 31 March 2018", [2019, 2018]),
    ("Particulars Year ended 31 March 2022 Year ended 31 March 2021", [2022, 2021]),
])
def test_period_columns(header, expected):
    years, confidence = detect_period_columns(header, report_fy=None)
    assert years == expected and confidence > 0.5


def test_period_columns_fall_back_to_the_report_year():
    years, confidence = detect_period_columns("Particulars Note Amount", report_fy=2019)
    assert years == [2019, 2018] and confidence < 0.5


# --------------------------------------------------------------------------- reading lines
@pytest.mark.parametrize("line,expected", [
    ("Current assets", "current_assets"),
    ("Non-current liabilities", "non_current_liabilities"),
    ("EQUITY AND LIABILITIES", "equity_and_liabilities"),
    ("Equity", "equity"),
])
def test_schedule_iii_subheadings_are_recognised(line, expected):
    assert detect_context(line) == expected


def test_borrowings_is_disambiguated_by_its_subheading(cfg, report_pages):
    """Schedule III prints ``Borrowings`` under both non-current and current
    liabilities. Only the sub-heading above it says which is long-term debt."""
    loc = locate_statements(report_pages, cfg, report_fy=2019)["balance_sheet"]
    hits = best_hits(map_statement(parse_statement_lines(report_pages, loc), loc))
    figures = year_figures(2019, True, seed=1)
    assert hits[("borrowings_long_term", 2019)].value == pytest.approx(figures["borrowings_long"])
    assert hits[("borrowings_short_term", 2019)].value == pytest.approx(figures["borrowings_short"])


def test_a_similar_label_does_not_steal_a_field(cfg, report_pages):
    """``Bank balances other than cash and cash equivalents`` is a different line."""
    loc = locate_statements(report_pages, cfg, report_fy=2019)["balance_sheet"]
    hits = best_hits(map_statement(parse_statement_lines(report_pages, loc), loc))
    assert hits[("cash_and_equivalents", 2019)].value == pytest.approx(
        year_figures(2019, True, seed=1)["cash"])


@pytest.mark.parametrize("label,expected", [
    ("Total assets", "total_assets"),
    ("Total current liabilities", "current_liabilities"),
    ("Revenue from operations", "revenue"),
    ("Profit/(loss) before tax", "pbt"),
    ("Profit/(loss) for the year", "net_profit"),
    ("Depreciation and amortisation expense", "depreciation"),
    ("Finance costs", "finance_costs"),
    ("Other equity", "other_equity"),
    ("Total equity and liabilities", "total_equity_and_liabilities"),
])
def test_line_items_map_to_standard_fields(label, expected):
    matched = match_field(label)
    assert matched is not None and matched[0] == expected


def test_an_unknown_label_is_not_forced_into_a_field():
    assert match_field("Contingent liabilities and commitments") is None


# --------------------------------------------------------------------------- whole document
@pytest.mark.parametrize("distressed", [True, False])
@pytest.mark.parametrize("unit", ["crore", "lakh", "million"])
def test_every_unit_gives_the_same_figures(cfg, distressed, unit):
    """The same company printed in lakhs must come out identical to crores."""
    pages = _pages(statement_pages("Demo Limited", 2019, distressed, seed=2, unit=unit))
    payload = {"doc_id": "BSE900101_FY2019", "firm_id": "BSE900101", "fy": 2019, "pages": pages}
    hits, _ = extract_document(payload, cfg)
    values = {(h.field, h.fy): h.value for h in best_hits(hits).values()}
    expected = year_figures(2019, distressed, seed=2)
    assert values[("total_assets", 2019)] == pytest.approx(expected["total_assets"], rel=1e-6)
    assert values[("total_equity", 2019)] == pytest.approx(expected["total_equity"], rel=1e-6)
    assert values[("revenue", 2019)] == pytest.approx(expected["revenue"], rel=1e-6)


def test_both_year_columns_are_extracted(cfg, report_pages):
    payload = {"doc_id": "BSE900101_FY2019", "firm_id": "BSE900101", "fy": 2019, "pages": report_pages}
    hits, _ = extract_document(payload, cfg)
    values = {(h.field, h.fy): h.value for h in best_hits(hits).values()}
    assert values[("total_assets", 2019)] == pytest.approx(year_figures(2019, True, 1)["total_assets"])
    assert values[("total_assets", 2018)] == pytest.approx(year_figures(2018, True, 1)["total_assets"])


def test_the_extracted_balance_sheet_balances(cfg, report_pages):
    payload = {"doc_id": "BSE900101_FY2019", "firm_id": "BSE900101", "fy": 2019, "pages": report_pages}
    hits, _ = extract_document(payload, cfg)
    row = pd.Series({h.field: h.value for h in best_hits(hits).values() if h.fy == 2019})
    checks = validate_year(row, cfg)
    assert checks["balance_check"] == pytest.approx(0.0, abs=1e-6)
    assert checks["components_check"] == pytest.approx(0.0, abs=1e-6)
    assert checks["validation_flags"] == ""


def test_a_broken_balance_sheet_is_flagged(cfg):
    row = pd.Series({"total_assets": 1000.0, "total_equity_and_liabilities": 1400.0,
                     "total_equity": 200.0, "non_current_liabilities": 300.0,
                     "current_liabilities": 100.0})
    checks = validate_year(row, cfg)
    assert "balance_sheet_does_not_balance" in checks["validation_flags"]
    assert "equity_plus_liabilities_mismatch" in checks["validation_flags"]


def test_a_hundredfold_jump_is_read_as_a_unit_error(cfg):
    """A firm can double in a year. It cannot grow exactly hundredfold."""
    row = pd.Series({"total_assets": 780000.0})
    checks = validate_year(row, cfg, previous_assets=7800.0)
    assert "possible_unit_confusion" in checks["validation_flags"]


def test_genuine_growth_is_not_flagged_as_a_unit_error(cfg):
    row = pd.Series({"total_assets": 9400.0})
    checks = validate_year(row, cfg, previous_assets=7800.0)
    assert "possible_unit_confusion" not in checks["validation_flags"]


def test_ocr_pages_are_counted_against_confidence(cfg):
    pages = _pages(statement_pages("Demo Limited", 2019, True, seed=3), method="ocr")
    loc = locate_statements(pages, cfg, report_fy=2019)["balance_sheet"]
    assert loc.ocr_pages >= 1
