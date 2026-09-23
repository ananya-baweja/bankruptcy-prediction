"""Regression tests for failures found on the first real annual reports (pilot, 2026-09-23).

Each test reproduces, in miniature, a page layout seen in a real BSE report
where the Phase 3 reader returned a wrong number.
"""

from __future__ import annotations

import pytest

from bpp.features.financials import extract_document
from bpp.features.numbers import detect_unit, infer_unit_from_magnitude, rupee_column_header
from bpp.features.statements import FieldHit, best_hits, locate_statements


def _bs_page(page, unit_line, rows, heading="Balance Sheet as at 31st March, 2020"):
    lines = ["Blue Blends (India) Limited", heading, "Note As at As at", "No. 31-Mar-2020 31-Mar-2019"]
    if unit_line:
        lines.append(unit_line)
    lines += rows
    return {"page": page, "method": "text", "text": "\n".join(lines)}


RUPEE_ROWS = [
    "a) Property, Plant and Equipment 5 171,173,618 207,470,812",
    "Total Non-Current Assets 420,947,475 472,224,657",
    "a) Inventories 8 8,865,999 18,209,107",
    "iii) Cash and cash equivalents 199,491 336,018",
    "Total Current Assets 363,506,810 419,902,885",
    "Total Assets 784,454,285 892,127,542",
    "a) Equity Share Capital 11 216,512,130 216,512,130",
    "b) Other Equity 12 (560,702,726) (417,576,004)",
    "Total Equity (344,190,596) (201,063,874)",
    "Total Equity and Liabilities 784,454,285 892,127,542",
]


def test_a_rupees_column_header_is_a_unit():
    assert rupee_column_header("Balance Sheet\nNote As at As at\nRupees Rupees\n") is True
    assert detect_unit("Balance Sheet\nParticulars Amount (Rs.) Amount (Rs.)\n")[0] == "rupee"
    # a real caption still governs, and prose is not a header
    assert detect_unit("(Rs. in Lakhs)\nRupees Rupees")[0] == "lakh"
    assert rupee_column_header("The Company paid Rupees 5 lakh as fees") is False


def test_whole_numbers_in_the_hundreds_of_thousands_are_rupees():
    assert infer_unit_from_magnitude([784454285, 363506810, 8865999, 199491, 216512130]) == "rupee"
    assert infer_unit_from_magnitude([7844.54, 3635.07, 88.66, 1.99, 2165.12]) is None     # lakh or crore: no opinion
    assert infer_unit_from_magnitude([1, 2]) is None                                            # too few to judge


def test_a_statement_printed_in_rupees_without_a_caption_is_read_in_crore(cfg):
    pages = [_bs_page(63, "Rupees Rupees", RUPEE_ROWS)]
    hits, report = extract_document({"doc_id": "BSE1_FY2020", "fy": 2020, "pages": pages}, cfg)
    got = {(h.field, h.fy): h.value for h in best_hits(hits).values()}
    assert got[("total_assets", 2020)] == pytest.approx(78.4454285)
    assert report["statements"]["balance_sheet"]["unit"] == "rupee"


def test_rupees_are_inferred_when_nothing_on_the_page_names_a_unit(cfg):
    pages = [_bs_page(63, None, RUPEE_ROWS)]
    hits, report = extract_document({"doc_id": "BSE1_FY2020", "fy": 2020, "pages": pages}, cfg)
    got = {(h.field, h.fy): h.value for h in best_hits(hits).values()}
    assert got[("total_assets", 2020)] == pytest.approx(78.4454285)
    assert "unit_inferred_rupee_from_magnitude" in report["statements"]["balance_sheet"]["flags"]


def test_the_profit_and_loss_after_the_balance_sheet_beats_an_earlier_highlights_table(cfg):
    highlights = {"page": 11, "method": "text", "text": "\n".join(
        ["Profit and Loss Account for the year ended 31st March 2019", "Particulars 2018-19 2017-18"]
        + [f"Item {i} {100 + i}.00 {90 + i}.00" for i in range(8)])}
    bs = _bs_page(49, "(Rs. in lakhs)", [f"Row {i} {1000 + i}.00 {900 + i}.00" for i in range(8)],
                  heading="Balance Sheet as at 31st March, 2019")
    pl = {"page": 50, "method": "text", "text": "\n".join(
        ["Statement of Profit and Loss for the year ended 31st March 2019", "(Rs. in lakhs)",
         "Particulars 2018-19 2017-18"] + [f"Line {i} {200 + i}.00 {190 + i}.00" for i in range(8)])}
    locs = locate_statements([highlights, bs, pl], cfg, 2019)
    assert locs["profit_and_loss"].start_page == 50


def test_a_cash_flow_movement_never_becomes_a_balance_or_an_expense():
    def hit(field, statement, value):
        return FieldHit(field=field, fy=2019, value=value, column=0, page=51, statement=statement,
                        label="x", match_how="pattern", match_score=1.0, parse_confidence=1.0)
    chosen = best_hits([hit("finance_costs", "cash_flow", 0.1167),
                        hit("borrowings_long_term", "cash_flow", -0.46),
                        hit("cash_and_equivalents", "cash_flow", 0.05)])
    assert ("finance_costs", 2019) not in chosen
    assert ("borrowings_long_term", 2019) not in chosen
    assert ("cash_and_equivalents", 2019) in chosen        # closing cash is a balance: allowed


def test_a_bracketed_finance_cost_is_an_expense_not_income(cfg):
    pl = {"page": 50, "method": "text", "text": "\n".join([
        "Statement of Profit and Loss for the year ended 31st March 2024", "(Rs. in lakhs)",
        "Particulars Note 2023-24 2022-23",
        "Revenue from operations 1,200.00 1,100.00",
        "Other income 10.00 12.00",
        "Total income 1,210.00 1,112.00",
        "(e) Finance Cost (0.43) (0.51)",
        "Depreciation and amortisation expense (20.00) (18.00)",
        "Profit before tax 40.00 35.00",
        "Profit for the year 30.00 26.00"])}
    hits, _ = extract_document({"doc_id": "BSE1_FY2024", "fy": 2024, "pages": [pl]}, cfg)
    got = {(h.field, h.fy): h for h in best_hits(hits).values()}
    assert got[("finance_costs", 2024)].value == pytest.approx(0.0043)
    assert "expense_sign_normalised" in got[("finance_costs", 2024)].parse_flags
    assert got[("depreciation", 2024)].value == pytest.approx(0.20)


# --------------------------------------------------------------------------- XBRL filed in the wrong unit
from bpp.features.financials import SOURCE_EXCHANGE_XBRL, SOURCE_PRIMARY, resolve_figures  # noqa: E402
import pandas as pd  # noqa: E402


def _pdf(value, unit_confidence):
    return pd.DataFrame([{
        "firm_id": "BSE1", "fy": 2019, "field": "total_assets", "value_cr": value, "source": SOURCE_PRIMARY,
        "source_doc_id": "BSE1_FY2019", "page": 57, "statement": "balance_sheet", "statement_scope": "standalone",
        "label": "TOTAL ASSETS", "match_how": "pattern", "match_score": 1.0, "unit": "lakh",
        "unit_confidence": unit_confidence, "printed": value * 100, "parse_confidence": 1.0,
        "parse_flags": "", "ocr_pages": 0}])


def _xbrl(value):
    return pd.DataFrame([{"firm_id": "BSE1", "fy": 2019, "field": "total_assets", "value_cr": value},
                         {"firm_id": "BSE1", "fy": 2019, "field": "revenue", "value_cr": value / 10}])


def test_an_xbrl_filing_10e5_off_is_not_believed_when_the_report_states_its_unit(cfg):
    cfg = {**cfg, "financials": {**cfg["financials"], "xbrl_priority": "first"}}
    resolved, _ = resolve_figures(_pdf(7.547497, 0.95), pd.DataFrame(), cfg, _xbrl(753310.8))
    got = resolved.set_index("field")
    assert got.loc["total_assets", "source"] == SOURCE_PRIMARY
    assert got.loc["total_assets", "value_cr"] == pytest.approx(7.547497)
    assert got.loc["total_assets", "unit_check"] == "xbrl_unit_suspect"
    assert "revenue" not in got.index                     # nothing else from that filing either


def test_when_the_report_unit_was_only_assumed_the_xbrl_still_wins(cfg):
    cfg = {**cfg, "financials": {**cfg["financials"], "xbrl_priority": "first"}}
    resolved, _ = resolve_figures(_pdf(4664.72, 0.2), pd.DataFrame(), cfg, _xbrl(46.6471))
    got = resolved.set_index("field")
    assert got.loc["total_assets", "source"] == SOURCE_EXCHANGE_XBRL
    assert got.loc["total_assets", "value_cr"] == pytest.approx(46.6471)
    assert got.loc["total_assets", "unit_check"] == "pdf_unit_suspect"


def test_an_implausibly_large_xbrl_balance_sheet_is_never_used(cfg):
    cfg = {**cfg, "financials": {**cfg["financials"], "xbrl_priority": "first"}}
    resolved, _ = resolve_figures(pd.DataFrame(), pd.DataFrame(), cfg, _xbrl(4_624_894.0))
    assert resolved.empty or "total_assets" not in set(resolved["field"])


def test_pbt_is_the_line_after_exceptional_items():
    from bpp.features.statements import match_field
    assert match_field("Profit before exceptional items and tax") is None
    assert match_field("Profit before tax and exceptional items") is None
    assert match_field("Profit / (Loss) before tax")[0] == "pbt"
    assert match_field("Profit after exceptional items but before tax")[0] == "pbt"


# --------------------------------------------------------------------------- second round (full pilot)
from bpp.features.numbers import split_label_and_figures  # noqa: E402
from bpp.features.statements import detect_context, match_field, strip_enumerator  # noqa: E402

YASHRAJ_BS = [                      # section totals printed with no label (BSE530063, FY2022)
    "Standalone Balance Sheet as at March 31, 2022",
    "(All amounts in Indian Rupees Lakhs, except as otherwise stated)",
    "Notes March 31, 2022 March 31, 2021",
    "ASSETS",
    "Non-current assets",
    "(i) Property, plant and equipment 3 61.86 77.14",
    "(ii) Deferred Tax Assets 31 182.98 0.00",
    " 244.84 77.14",
    "Current assets",
    "(i) Inventories 11 - 64.69",
    "(ii) Financial assets",
    " - Trade receivables 12 164.89 185.00",
    " - Cash and cash equivalents 13 2.46 31.59",
    "(iii) Other current assets 10 68.76 54.38",
    " 236.11 335.66",
    "Assets available for Sale 4 200.00 200.00",
    "Total assets 680.95 612.80",
    "EQUITY AND LIABILITIES",
    "EQUITY",
    "(i) Equity share capital 15 1,700.00 1,700.00",
    "(ii) Other equity 16 (9,033.36) (9,448.14)",
    " (7,333.36) (7,748.14)",
    "LIABILITIES",
    "Non Current Liabilities",
    "(i) Financial liabilities",
    " - Borrowings 17 - -",
    "(ii) Provisions 19 36.84 31.01",
    " 36.84 31.01",
    "Current liabilities",
    "(i) Financial liabilities",
    " - Borrowings 17 6,676.13 6,873.20",
    " - Other financial liabilities 18 304.10 324.68",
    "(ii) Provisions 19 0.02 0.00",
    "(iii) Other current liabilities 20 997.22 1,034.95",
    " 7,977.47 8,232.83",
    "Total equity and liabilities 680.95 612.80",
]


def _read(rows, cfg, fy=2022):
    pages = [{"page": 64, "method": "text", "text": "\n".join(rows)}]
    hits, _ = extract_document({"doc_id": f"X_FY{fy}", "fy": fy, "pages": pages}, cfg)
    return {(h.field, h.fy): h for h in best_hits(hits).values()}


def test_other_current_liabilities_is_not_the_current_liabilities_total():
    assert match_field("(iii) Other current liabilities") is None
    assert match_field("Other current assets") is None
    # an OCR slip in the first word still matches
    assert match_field("Totai current liabilities")[0] == "current_liabilities"


def test_unlabelled_section_totals_are_read_when_they_add_up(cfg):
    got = _read(YASHRAJ_BS, cfg)
    assert got[("current_liabilities", 2022)].value == pytest.approx(79.7747)
    assert got[("current_liabilities", 2022)].match_how == "subtotal"
    assert got[("current_assets", 2022)].value == pytest.approx(2.3611)
    assert got[("non_current_assets", 2022)].value == pytest.approx(2.4484)
    assert got[("total_equity", 2022)].value == pytest.approx(-73.3336)
    assert got[("non_current_liabilities", 2022)].value == pytest.approx(0.3684)
    assert got[("borrowings_short_term", 2022)].value == pytest.approx(66.7613)
    # the grand total after "Assets available for sale" is not the current-assets total
    assert got[("total_assets", 2022)].value == pytest.approx(6.8095)


def test_an_unlabelled_line_that_does_not_add_up_is_ignored(cfg):
    rows = list(YASHRAJ_BS)
    rows[rows.index(" 7,977.47 8,232.83")] = " 9,999.00 8,232.83"
    got = _read(rows, cfg)
    assert ("current_liabilities", 2022) not in got


def test_a_labelled_total_beats_the_unlabelled_one(cfg):
    rows = list(YASHRAJ_BS)
    rows.insert(rows.index(" 7,977.47 8,232.83") + 1, "Total current liabilities 7,977.50 8,232.83")
    got = _read(rows, cfg)
    assert got[("current_liabilities", 2022)].value == pytest.approx(79.775)
    assert got[("current_liabilities", 2022)].match_how == "pattern"


INFRA_PL = [                        # figures on the lines after the label (BSE530777, FY2019)
    "STATEMENT OF PROFIT AND LOSS FOR THE PERIOD ENDED 31st March, 2019",
    "(Amount in Rs.)",
    "PARTICULARS",
    "NOTE",
    "For the Year Ended",
    "31st March, 2019 31st March, 2018",
    "INCOME",
    "Revenue from Operations 19 ",
    "27,053,549 ",
    "",
    "32,663,618 ",
    "Total Income ",
    "32,991,372 ",
    "",
    "42,990,899 ",
    "EXPENDITURE",
    "Finance Cost 24 ",
    "6,054,437 ",
    "",
    "6,815,964 ",
    "Depreciation and Amortisation Expense 2 ",
    "2,910,868 ",
    "",
    "2,755,357 ",
    "Profit/(Loss) before Tax and after Exceptional Item (5,064,449) (9,492,950)",
]


def test_figures_on_the_lines_after_the_label_are_joined_to_it(cfg):
    got = _read(INFRA_PL, cfg, fy=2019)
    assert got[("revenue", 2019)].value == pytest.approx(2.7053549)
    assert got[("revenue", 2018)].value == pytest.approx(3.2663618)
    assert got[("finance_costs", 2019)].value == pytest.approx(0.6054437)
    assert got[("depreciation", 2019)].value == pytest.approx(0.2910868)
    assert got[("total_income", 2019)].value == pytest.approx(3.2991372)
    assert got[("pbt", 2019)].value == pytest.approx(-0.5064449)


def test_a_note_reference_before_a_single_figure_is_not_this_years_figure():
    assert split_label_and_figures("a) Revenue from Operations 20 68,77,51,670") == (
        "a) Revenue from Operations", ["68,77,51,670"])
    assert split_label_and_figures("Revenue from Operations 19 ") == ("Revenue from Operations", [])
    assert split_label_and_figures(" - Cash and cash equivalents 13 2.46")[1] == ["2.46"]
    # a full line is positional as before, and two small figures are both figures
    assert split_label_and_figures("Share capital 15 1,700.00 1,700.00")[1] == ["1,700.00", "1,700.00"]
    assert split_label_and_figures("Inventories 12.50 10.25")[1] == ["12.50", "10.25"]
    assert split_label_and_figures("Exceptional items 5 -")[1] == ["5", "-"]


def test_enumerated_headings_set_the_context():
    assert detect_context("(2) Current Assets") == "current_assets"
    assert detect_context("2. CURRENT LIABILITIES") == "current_liabilities"
    assert detect_context("II CURRENT ASSETS") == "current_assets"
    assert detect_context("B EQUITY AND LIABILITIES") == "equity_and_liabilities"
    assert strip_enumerator("(a) (i) Borrowings") == "Borrowings"
    assert strip_enumerator("Cash and cash equivalents") == "Cash and cash equivalents"


def test_bare_total_lines_by_side_of_the_balance_sheet(cfg):
    rows = [
        "Balance Sheet as at 31st March, 2019", "(Rs. in Lakhs)", "Note 31.03.2019 31.03.2018",
        "(1) Non- Current Assets", "(a) Property, Plant and Equipment 2 120,000.00 150,000.00",
        "(2) Current Assets", "(a) Inventories 7 17,338.30 68,462.12",
        "TOTAL 137,338.30 218,462.12",
        "EQUITY AND LIABILITIES", "(1) Equity", "(a) Equity Share Capital 12 5,000.00 5,000.00",
        "(b) Other Equity 13 100,000.00 180,000.00",
        "(2) Current liabilities", "(a) Financial Liabilities",
        "(i) Borrowings 17 20,000.00 20,000.00",
        "(b) Other current liabilities 19 12,338.30 13,462.12",
        "Total Equity and Liabilities 137,338.30 218,462.12",
    ]
    got = _read(rows, cfg, fy=2019)
    assert got[("total_assets", 2019)].value == pytest.approx(1373.383)
    assert got[("borrowings_short_term", 2019)].value == pytest.approx(200.0)
    assert ("current_liabilities", 2019) not in got      # "Other current liabilities" is a component


def test_the_opening_cash_balance_is_not_cash(cfg):
    assert match_field("Cash and Cash Equivalents at the Beginning of the Year") is None
    assert match_field("Net Increase / (Decrease) In Cash and Cash Equivalent") is None
    assert match_field("Cash and cash equivalents at the end of the year")[0] == "cash_and_equivalents"
    assert match_field("(ii) Cash and cash equivalents")[0] == "cash_and_equivalents"


def test_ocr_slips_in_figures():
    from bpp.features.numbers import parse_number
    # the opening bracket lost by OCR
    assert split_label_and_figures("(b) Other equity 77,331,519) 25,760,299")[1] == ["77,331,519)", "25,760,299"]
    n = parse_number("77,331,519)", "rupee")
    assert n.ok and n.printed == -77331519 and "ocr_lost_open_bracket" in n.flags
    # a space before a thousands comma
    assert split_label_and_figures("(i) Long term borrowings 12 6 ,24,79,590 5,00,00,000") == (
        "(i) Long term borrowings", ["6,24,79,590", "5,00,00,000"])
    # a formula reference is not a figure
    assert split_label_and_figures("3 Profit Before Exceptional Item and Tax (1-2) 9,58,44,849 7,96,23,379")[1] == [
        "9,58,44,849", "7,96,23,379"]
    assert split_label_and_figures("Profit before tax (III-IV)")[1] == []


# --------------------------------------------------------------------------- XBRL vs report
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bpp.features.financials import (SOURCE_EXCHANGE_XBRL, _drop_impossible_components,  # noqa: E402
                                     arbitrate_by_identities, xbrl_stale_copies)


def test_an_xbrl_filing_repeating_last_years_profit_and_loss_is_caught():
    rows = []
    for fy, ta in ((2019, 1557.7), (2020, 1645.7)):
        rows += [("BSE1", fy, "revenue", 215.6697), ("BSE1", fy, "pbt", 1.5225),
                 ("BSE1", fy, "net_profit", 1.4304), ("BSE1", fy, "total_assets", ta)]
    rows += [("BSE2", 2019, "revenue", 10.0), ("BSE2", 2020, "revenue", 10.0),   # one flat figure: fine
             ("BSE2", 2019, "pbt", 1.0), ("BSE2", 2020, "pbt", 2.0)]
    ex = pd.DataFrame(rows, columns=["firm_id", "fy", "field", "value_cr"])
    assert xbrl_stale_copies(ex) == {("BSE1", 2020)}


def _resolved(values, conflicts):
    rows = []
    for field, v in values.items():
        row = {"firm_id": "BSE1", "fy": 2020, "field": field, "value_cr": v, "source": SOURCE_EXCHANGE_XBRL,
               "pdf_value_cr": np.nan, "xbrl_pdf_diff": np.nan, "pdf_source": "", "pdf_source_doc_id": "",
               "pdf_page": np.nan}
        if field in conflicts:
            pdf = conflicts[field]
            row.update({"pdf_value_cr": pdf, "pdf_source": "report_current_year",
                        "pdf_source_doc_id": "BSE1_FY2020", "pdf_page": 59,
                        "xbrl_pdf_diff": abs(pdf - v) / max(abs(pdf), abs(v))})
        rows.append(row)
    return pd.DataFrame(rows)


def test_the_value_that_balances_the_balance_sheet_wins():
    base = {"total_assets": 43.5742, "total_equity_and_liabilities": 43.5742, "total_equity": 40.9417,
            "non_current_liabilities": 0.0, "current_liabilities": 0.0945}
    out = arbitrate_by_identities(_resolved(base, {"current_liabilities": 2.6325}))
    cl = out[out.field == "current_liabilities"].iloc[0]
    assert cl.value_cr == pytest.approx(2.6325) and cl.source == "report_current_year"
    assert cl.xbrl_value_cr == pytest.approx(0.0945)
    # the other way round: the XBRL balances, the report picked a component -> XBRL stays
    good = dict(base, current_liabilities=2.6325)
    out = arbitrate_by_identities(_resolved(good, {"current_liabilities": 0.0945}))
    cl = out[out.field == "current_liabilities"].iloc[0]
    assert cl.value_cr == pytest.approx(2.6325) and cl.source == SOURCE_EXCHANGE_XBRL
    # nothing to test against (no identity can be evaluated) -> XBRL stays
    out = arbitrate_by_identities(_resolved({"current_liabilities": 0.0945}, {"current_liabilities": 2.6325}))
    assert out.iloc[0].source == SOURCE_EXCHANGE_XBRL


def test_an_asset_line_larger_than_a_balancing_balance_sheet_is_dropped():
    wide = pd.DataFrame([{"firm_id": "BSE1", "fy": 2019, "total_assets": 50.0, "balance_check": 0.0,
                          "cash_and_equivalents": 1.98e7, "inventories": 5.0, "validation_flags": "",
                          "n_validation_flags": 0},
                         {"firm_id": "BSE1", "fy": 2018, "total_assets": 50.0, "balance_check": 0.3,
                          "cash_and_equivalents": 80.0, "inventories": 5.0, "validation_flags": "",
                          "n_validation_flags": 0}])
    out = _drop_impossible_components(wide)
    assert np.isnan(out.loc[0, "cash_and_equivalents"]) and out.loc[0, "inventories"] == 5.0
    assert "cash_and_equivalents_exceeds_total_assets_dropped" in out.loc[0, "validation_flags"]
    # a balance sheet that does not balance proves nothing: kept, only flagged elsewhere
    assert out.loc[1, "cash_and_equivalents"] == 80.0


def test_a_confidently_misread_report_unit_does_not_override_consistent_xbrl():
    from bpp.features.financials import SOURCE_COMPARATIVE, SOURCE_PRIMARY, xbrl_unit_checks
    figs = pd.DataFrame([
        # FY2024 report read as crore although printed in lakh: both columns 100x
        {"firm_id": "BSE9", "fy": 2024, "field": "total_assets", "value_cr": 16868.01, "source": SOURCE_PRIMARY,
         "source_doc_id": "BSE9_FY2024", "unit_confidence": 0.95},
        {"firm_id": "BSE9", "fy": 2023, "field": "total_assets", "value_cr": 15500.00, "source": SOURCE_COMPARATIVE,
         "source_doc_id": "BSE9_FY2024", "unit_confidence": 0.95},
        # a filing 10^5 off, against a report whose next-year comparative agrees with it
        {"firm_id": "BSE8", "fy": 2019, "field": "total_assets", "value_cr": 7.5475, "source": SOURCE_PRIMARY,
         "source_doc_id": "BSE8_FY2019", "unit_confidence": 0.5},
        {"firm_id": "BSE8", "fy": 2019, "field": "total_assets", "value_cr": 7.5475, "source": SOURCE_COMPARATIVE,
         "source_doc_id": "BSE8_FY2020", "unit_confidence": 0.5},
    ])
    ex = pd.DataFrame([("BSE9", 2024, "total_assets", 168.6801), ("BSE9", 2023, "total_assets", 155.0),
                       ("BSE8", 2019, "total_assets", 753310.8), ("BSE8", 2020, "total_assets", 7.9)],
                      columns=["firm_id", "fy", "field", "value_cr"])
    checks = xbrl_unit_checks(figs, ex)
    assert checks[("BSE9", 2024)] == "pdf_unit_suspect"     # the report slipped, whatever its caption said
    assert checks[("BSE8", 2019)] == "xbrl_unit_suspect"    # the filing slipped, though the report's unit was a guess
