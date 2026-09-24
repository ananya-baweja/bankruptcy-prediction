"""Regression tests for failures found on the first real annual reports (pilot, 2026-09-23).

Each test reproduces, in miniature, a page layout seen in a real BSE report
where the Phase 3 reader returned a wrong number.
"""

from __future__ import annotations

import pytest

from bpp.features.financials import extract_document
from bpp.features.numbers import detect_unit, infer_unit_from_magnitude, parse_number, rupee_column_header
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
    # the stray figure is never taken; the rows' own sum is, because it balances the sheet
    assert got[("current_liabilities", 2022)].value == pytest.approx(79.7747)
    assert got[("current_liabilities", 2022)].match_how == "section_sum"


OLD_FORMAT_BS = [                  # pre-Ind AS layout: no section totals, one "Total" per side
    "Balance Sheet as at 31st March, 2017",
    "Particulars Note No. As at 31.03.2017 As at 31.03.2016",
    "A. EQUITY AND LIABILITIES",
    "(1) Shareholders' Funds",
    "(a) Share Capital 3 149,580,000 149,580,000",
    "(b) Reserve & Surplus 4 (13,286,278) (13,195,179)",
    "(2) Non-Current Liabilities",
    "(a) Long-term borrowings 5 - -",
    "(b) Deferred tax liabilities (Net) 5 10,972 20,862",
    "(3) Current Liabilities",
    "(a) Short-term borrowings 6 50,000 50,000",
    "(b) Trade payables 7 38,000 12,000",
    "(c) Other current liabilities 8 38,322 67,700",
    "Total 136,431,016 136,535,383",
    "B. ASSETS",
    "(1) Non-current assets",
    "(a) Fixed assets",
    "(i) Tangible assets 10 1,000,000 1,100,000",
    "(ii) Intangible assets 11 129,202 271,510",
    " 1,129,202 1,371,510",
    "(b) Non-current investments 12 95,480,472 95,480,472",
    "(2) Current assets",
    "(a) Inventories 13 38,888,794 38,475,337",
    "(b) Cash and cash equivalents 14 1,000,000 1,000,000",
    "(c) Other current assets 17 (67,452) 208,064",
    "Total 136,431,016 136,535,383",
]


def test_sections_without_totals_are_summed_when_the_sheet_balances(cfg):
    got = _read(OLD_FORMAT_BS, cfg, fy=2017)
    assert got[("total_equity", 2017)].value == pytest.approx(13.6293722)
    assert got[("non_current_liabilities", 2017)].value == pytest.approx(0.0010972)
    assert got[("current_liabilities", 2017)].value == pytest.approx(0.0126322)
    assert got[("current_assets", 2017)].value == pytest.approx(3.9821342)
    # "(a) Fixed assets" has its own unlabelled sum, but it is not the section's total
    assert got[("non_current_assets", 2017)].value == pytest.approx(9.6609674)
    assert got[("total_assets", 2017)].value == pytest.approx(13.6431016)
    assert got[("current_liabilities", 2016)].value == pytest.approx(0.0129700)


def test_summed_sections_that_do_not_balance_are_dropped(cfg):
    rows = [r for r in OLD_FORMAT_BS if not r.startswith("(b) Trade payables")]   # a row lost to OCR
    got = _read(rows, cfg, fy=2017)
    for f in ("total_equity", "non_current_liabilities", "current_liabilities"):
        assert (f, 2017) not in got
    assert got[("current_assets", 2017)].value == pytest.approx(3.9821342)    # the other side still balances


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
    # a scanned report's own text layer: spaces after a comma and around the decimal point
    assert split_label_and_figures("(iii) Cash and Cash Equivalents 7 285.12 1, 255. 53 1, 172. 16", 3) == (
        "(iii) Cash and Cash Equivalents", ["285.12", "1,255.53", "1,172.16"])
    assert split_label_and_figures("(b) Intangible Assets 2 10.06 70 .68 131 . 65", 3)[1] == [
        "10.06", "70.68", "131.65"]
    assert split_label_and_figures("Total Non-Current Assets 60,605.79 64 ,832. 53 68 ,720. 31", 3)[1] == [
        "60,605.79", "64,832.53", "68,720.31"]
    # ... but a date's year and a list of notes are never joined to a figure
    assert split_label_and_figures("Balance as at March 31, 2019 1,234.56 1,000.00")[1] == ["1,234.56", "1,000.00"]
    assert split_label_and_figures("Trade receivables (Refer Notes 5, 6) 1,234.56 1,100.00")[1] == [
        "1,234.56", "1,100.00"]
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


def test_headings_with_bare_numbers_and_spaced_hyphens(cfg):
    rows = [                        # BSE531867 FY2023 layout, lakh
        "Balance Sheet As At 31st March 2023 (Rs. In Lakhs)", "Note No. As at 31st March 2023 As at 31st March 2022",
        "ASSETS",
        "1 Non - Current assets",
        "a Property, Plant and Equipment 1 504.09 622.55",
        "b Capital work-in-progress - -",
        "h Financial Assets",
        "ii Loans & Advances 2 21.09 20.04",
        "c Other Non - Current assets 3 19.95 -",
        "Sub-Total 545.13 642.60",
        "2 Current assets",
        "a Inventories 4 704.69 1,590.51",
        "ii Trade receivables 5 2,517.40 2,354.01",
        "e Other Current Assets 9 163.36 37.78",
        "Sub-Total 3,385.45 3,982.30",
        " 3,930.58 4,624.90",
    ]
    got = _read(rows, cfg, fy=2023)
    assert got[("non_current_assets", 2023)].value == pytest.approx(5.4513)
    assert got[("current_assets", 2023)].value == pytest.approx(33.8545)
    assert got[("inventories", 2023)].value == pytest.approx(7.0469)
    assert ("total_assets", 2023) not in got     # "Sub-Total" is never the balance sheet total


def test_a_subtotal_equal_to_the_grand_total_is_not_a_section_total(cfg):
    rows = [                        # the "2 Current assets" heading lost: its rows run on under non-current
        "Balance Sheet as at 31st March, 2019", "(Rs. in Lakhs)", "Note 31.03.2019 31.03.2018",
        "Non-current assets",
        "Property, plant and equipment 2 2,868.71 2,972.80",
        "Investments 3 9,138.50 8,422.64",
        "CURRENTASSETS",
        "Inventories 7 8.03 10.16",
        "Trade receivables 8 564.96 363.61",
        "TOTAL 12,580.20 11,769.21",
    ]
    got = _read(rows, cfg, fy=2019)
    assert ("non_current_assets", 2019) not in got
    assert got[("total_assets", 2019)].value == pytest.approx(125.802)


def test_signature_lines_under_the_sheet_are_not_rows(cfg):
    rows = [                        # BSE533157 FY2016, rupees, section totals unlabelled
        "BALANCE SHEET AS AT 31 MARCH 2016 Amount in Rs.", "PARTICULARS Note No. 31.03.2016 31.03.2015",
        "I. EQUITY AND LIABILITIES", "1 Shareholders' Fund:",
        "Share Capital 2 400,000,000 400,000,000", "Reserves and Surplus 3 1,201,555,660 1,236,731,499",
        "1,601,555,660 1,636,731,499",
        "2 Non-Current Liabilities", "Long Term Borrowings 4 106,999,900 128,639,768",
        "Long Term Provisions 5 3,558,231 2,262,768", " 110,558,131 130,902,536",
        "3 Current Liabilities", "Short Term Borrowings 6 192,612,152 147,150,348",
        "Trade Payables 7 68,997,721 62,203,438", "Other Current Liabilities 8 20,175,747 29,248,163",
        "Short Term Provisions 9 12,977,854 6,454,249", " 294,763,474 245,056,198",
        "Total 2,006,877,265 2,012,690,233",
        "II. ASSETS", "1 Non Current Assets", "Fixed Assets 10",
        "Tangible Assets 203,539,265 230,242,418", "Intangible Assets 3,373,459 4,326,416",
        "206,912,724 234,568,834",
        "Non Current Investments 11 908,071,171 908,071,171", "Deferred Tax Assets (Net) 12 70,121,191 62,125,241",
        "Long Term Loans and Advances 13 135,269,853 148,927,861", "Other Non-Current Assets 14 12,293,191 17,082,692",
        " 1,125,755,406 1,136,206,965",
        "2 Current Assets", "Inventories 15 133,505,444 133,149,402", "Trade Receivables 16 305,569,022 265,769,982",
        "Cash and Bank Balances 17 13,708,663 8,035,357", "Short Term Loans and Advances 18 194,045,334 216,082,297",
        "Other Current Assets 19 27,380,672 18,877,396", " 674,209,135 641,914,434",
        "Total 2,006,877,265 2,012,690,233",
        "Firm Regn. No. 104863W", "DIN : 02675798", "Membership No. 137686", "Date: May 30, 2016",
    ]
    got = _read(rows, cfg, fy=2016)
    assert got[("current_assets", 2016)].value == pytest.approx(67.4209135)
    # the report prints two partial sums under non-current assets (fixed assets: 20.69 cr;
    # the rest: 112.58 cr); neither is the section total. The rows' sum is, because
    # non-current + current = the printed total of 200.69 cr
    assert got[("non_current_assets", 2016)].value == pytest.approx(133.266813)
    assert got[("non_current_assets", 2016)].match_how == "section_sum"
    assert got[("current_liabilities", 2016)].value == pytest.approx(29.4763474)
    assert got[("total_equity", 2016)].value == pytest.approx(160.155566)
    assert got[("total_assets", 2016)].value == pytest.approx(200.6877265)


def test_missing_totals_are_filled_from_their_definition_only():
    from bpp.features.financials import derive_missing_totals
    wide = pd.DataFrame([{"firm_id": "BSE1", "fy": 2017, "total_assets": 100.0, "total_equity_and_liabilities": np.nan,
                          "non_current_assets": 60.0, "current_assets": np.nan, "equity_share_capital": 10.0,
                          "other_equity": 30.0, "total_equity": np.nan, "non_current_liabilities": 20.0,
                          "current_liabilities": np.nan},
                         {"firm_id": "BSE1", "fy": 2016, "total_assets": 90.0, "total_equity_and_liabilities": 90.0,
                          "non_current_assets": 50.0, "current_assets": 45.0, "equity_share_capital": 10.0,
                          "other_equity": np.nan, "total_equity": 35.0, "non_current_liabilities": 20.0,
                          "current_liabilities": np.nan}])
    out = derive_missing_totals(wide)
    r = out.iloc[0]
    assert r.current_assets == 40.0 and r.total_equity == 40.0 and r.total_equity_and_liabilities == 100.0
    assert r.current_liabilities == 40.0          # 100 - 40 - 20, using the derived equity
    assert set(r.derived_fields.split(";")) == {"current_assets", "total_equity", "total_equity_and_liabilities",
                                                "current_liabilities"}
    r = out.iloc[1]
    assert r.current_assets == 45.0               # a figure that was read is never replaced
    assert r.current_liabilities == 35.0 and r.derived_fields == "current_liabilities"


# --------------------------------------------------------------------------- units on the full cohort (2026-09-25)
# The first full-cohort run read 66 company-years a power of ten off the filings. Each
# test below is one caption layout (or its absence) from those reports.
LAKH_ROWS = [
    "a) Property, Plant and Equipment 5 1,711.74 2,074.71",
    "Total Non-Current Assets 4,209.47 4,722.25",
    "a) Inventories 8 88.66 182.09",
    "iii) Cash and cash equivalents 1.99 3.36",
    "Total Current Assets 3,635.07 4,199.03",
    "Total Assets 7,844.54 8,921.28",
    "a) Equity Share Capital 11 2,165.12 2,165.12",
    "b) Other Equity 12 (5,607.03) (4,175.76)",
    "Total Equity (3,441.91) (2,010.64)",
    "Total Equity and Liabilities 7,844.54 8,921.28",
]
SIGNATURES = ["As per our report of even date", "For M.M. Parikh & Co", "Chartered Accountants",
              "Place : Hinganghat", "Date : 30 May 2019"]


def _bs_unit(pages, cfg, fy=2020):
    hits, report = extract_document({"doc_id": "BSE1_FY2020", "fy": fy, "pages": pages}, cfg)
    got = {(h.field, h.fy): h.value for h in best_hits(hits).values()}
    return got, report["statements"]["balance_sheet"]


@pytest.mark.parametrize("caption_line", [
    "st Balance sheet as at 31 March 2020 ( ` in Lakhs)",          # the heading line, with a date in it
    "Balance Sheet as at March 31, 2020 (Amount in C lakh)",       # ₹ mis-encoded as "C"
    "(All amount in lacs of Indian Rupees, except share data and as stated otherwise) "
    "(All amount in lacs of Indian Rupees, except share data and as stated otherwise",   # a two-page spread
])
def test_a_caption_anywhere_on_the_page_sets_the_unit(cfg, caption_line):
    page = _bs_page(69, None, LAKH_ROWS + SIGNATURES + [caption_line])
    got, bs = _bs_unit([page], cfg)
    assert bs["unit"] == "lakh"
    assert got[("total_assets", 2020)] == pytest.approx(78.4454)


def test_hundreds_are_a_unit_but_one_hundred_is_not():
    assert detect_unit("(` in Hundreds)") == ("hundred", 0.95)
    assert detect_unit("(Rs.in Hundred)")[0] == "hundred"
    assert detect_unit("paid one hundred shares to the trust")[1] < 0.5
    assert parse_number("3,76,867.22", "hundred").value == pytest.approx(3.7686722)


def test_amount_in_the_rupee_font_glyph_is_rupees():
    assert detect_unit("(Amount in `)\nAs at As at") == ("rupee", 0.95)


@pytest.mark.parametrize("garbled", ["(`LQ/DNKV", "$PRXQWLQ,15/DNKV"])
def test_a_caption_in_a_shifted_font_is_decoded(cfg, garbled):
    page = _bs_page(87, None, LAKH_ROWS + SIGNATURES + [garbled])
    got, bs = _bs_unit([page], cfg)
    assert bs["unit"] == "lakh" and "unit_caption_decoded_from_shifted_font" in bs["flags"]
    assert got[("total_assets", 2020)] == pytest.approx(78.4454)


def test_an_uncaptioned_balance_sheet_takes_the_unit_of_the_profit_and_loss(cfg):
    bs = _bs_page(59, None, LAKH_ROWS + SIGNATURES)
    pl = {"page": 60, "method": "text", "text": "\n".join(
        ["Statement of Profit and Loss for the year ended 31st March 2020", "(Amount in Lakhs)",
         "Particulars Note 2019-20 2018-19"] + [f"Line {i} {200 + i}.00 {190 + i}.00" for i in range(8)])}
    got, info = _bs_unit([bs, pl], cfg)
    assert info["unit"] == "lakh" and "unit_from_profit_and_loss" in info["flags"]
    assert got[("total_assets", 2020)] == pytest.approx(78.4454)


@pytest.mark.parametrize("policy,unit,total", [
    ("These standalone financial statements are presented in lakhs of Indian rupees which is also the "
     "Company's functional currency", "lakh", 78.4454),
    ("The financial statements are presented in Indian Rupee and all values are rounded to the nearest "
     "lakhs, except when otherwise stated.", "lakh", 78.4454),
])
def test_the_accounting_policy_note_states_the_unit(cfg, policy, unit, total):
    bs = _bs_page(45, None, LAKH_ROWS + SIGNATURES)
    note = {"page": 47, "method": "text", "text": "Notes to the financial statements\n1. Basis of preparation\n" + policy}
    got, info = _bs_unit([bs, note], cfg)
    assert info["unit"] == unit and "unit_from_accounting_policy_note" in info["flags"]
    assert got[("total_assets", 2020)] == pytest.approx(total)


def test_rupees_and_paise_without_a_caption_are_rupees(cfg):
    rows = ["a) Property, Plant and Equipment 3 7889733.73 8666430.73", "b) Capital Work in Progress 0.00 0.00",
            "Total Non-Current Assets 17889733.73 18666430.73", "a) Inventories 4 2345678.12 3456789.45",
            "Total Current Assets 12345678.91 13456789.02", "Total Assets 30235412.64 32123219.75",
            "a) Equity Share Capital 10A 184124400.00 184124400.00",
            "b) Other Equity 10B -249647239.11 -245720221.41"]
    got, info = _bs_unit([_bs_page(70, None, rows)], cfg)
    assert info["unit"] == "rupee" and got[("total_assets", 2020)] == pytest.approx(3.023541264)


def test_a_lakh_caption_over_whole_rupee_figures_is_overruled(cfg):
    rows = ["(a) Property, plant and equipment 3 13,16,32,774.00 14,09,00,864.00",
            "(b) Capital work-in-progress 5,59,19,487.00 5,59,19,487.00",
            "Total non current assets 24,14,44,320.00 23,99,05,581.00",
            "(a) Inventories 7 3,88,39,172.00 3,72,61,910.00",
            "Total current assets 9,95,17,312.00 9,83,93,145.00",
            "Total Assets 34,09,61,632.00 33,82,98,726.00",
            "(a) Equity Share capital 11 12,00,00,000.00 12,00,00,000.00"]
    got, info = _bs_unit([_bs_page(59, "(All amounts in lacs unless otherwise stated)", rows)], cfg)
    assert info["unit"] == "rupee" and "unit_caption_contradicted_by_magnitude" in info["flags"]
    assert got[("total_assets", 2020)] == pytest.approx(34.0961632)


def test_a_rupees_header_over_lakh_sized_figures_defers_to_the_notes(cfg):
    bs = _bs_page(82, "Rs. Rs.", LAKH_ROWS + ["M. No. : 159938"])
    note = {"page": 93, "method": "text",
            "text": "Note 11 Share capital\nEquity Share of Rs. 1 each issued, subscribed and fully paid No. Rs. In Lakhs"}
    got, info = _bs_unit([bs, note], cfg)
    assert info["unit"] == "lakh" and "unit_rupee_header_implausible" in info["flags"]
    assert got[("total_assets", 2020)] == pytest.approx(78.4454)


def test_two_slipped_filings_in_a_row_do_not_outvote_the_report():
    from bpp.features.financials import SOURCE_COMPARATIVE, xbrl_unit_checks
    figs = pd.DataFrame([
        {"firm_id": "BSE7", "fy": 2023, "field": "total_assets", "value_cr": 1177.99, "source": SOURCE_PRIMARY,
         "source_doc_id": "BSE7_FY2023", "unit_confidence": 0.95},
        {"firm_id": "BSE7", "fy": 2022, "field": "total_assets", "value_cr": 1193.89, "source": SOURCE_COMPARATIVE,
         "source_doc_id": "BSE7_FY2023", "unit_confidence": 0.95},
    ])
    ex = pd.DataFrame([("BSE7", 2021, "total_assets", 946.03), ("BSE7", 2022, "total_assets", 11.9389),
                       ("BSE7", 2023, "total_assets", 11.7799), ("BSE7", 2024, "total_assets", 1352.15)],
                      columns=["firm_id", "fy", "field", "value_cr"])
    checks = xbrl_unit_checks(figs, ex)
    assert checks[("BSE7", 2023)] == "xbrl_unit_suspect"
    assert checks[("BSE7", 2022)] == "xbrl_unit_suspect"     # a year with only next year's comparative


def test_a_report_read_at_the_wrong_scale_supplies_neither_column(cfg):
    from bpp.features.financials import SOURCE_COMPARATIVE
    cfg = {**cfg, "financials": {**cfg["financials"], "xbrl_priority": "first"}}
    row = dict(statement="balance_sheet", statement_scope="standalone", label="x", match_how="pattern",
               match_score=1.0, unit="crore", unit_confidence=0.2, parse_confidence=1.0, parse_flags="",
               ocr_pages=0, page=50)
    figs = pd.DataFrame([
        {**row, "firm_id": "BSE6", "fy": 2021, "field": "total_assets", "value_cr": 4664.72,
         "source": SOURCE_PRIMARY, "source_doc_id": "BSE6_FY2021", "printed": 4664.72},
        {**row, "firm_id": "BSE6", "fy": 2021, "field": "inventories", "value_cr": 800.0,
         "source": SOURCE_PRIMARY, "source_doc_id": "BSE6_FY2021", "printed": 800.0},
        # FY2020 has no report of its own: only the FY2021 report's prior-year column
        {**row, "firm_id": "BSE6", "fy": 2020, "field": "inventories", "value_cr": 700.0,
         "source": SOURCE_COMPARATIVE, "source_doc_id": "BSE6_FY2021", "printed": 700.0},
    ])
    ex = pd.DataFrame([("BSE6", 2021, "total_assets", 46.6472), ("BSE6", 2022, "total_assets", 50.1)],
                      columns=["firm_id", "fy", "field", "value_cr"])
    resolved, _ = resolve_figures(figs, pd.DataFrame(), cfg, ex)
    keys = set(zip(resolved["fy"], resolved["field"]))
    assert (2021, "inventories") not in keys and (2020, "inventories") not in keys
    assert resolved.set_index(["fy", "field"]).loc[(2021, "total_assets"), "value_cr"] == pytest.approx(46.6472)


def test_a_currency_only_caption_below_the_sheet_is_rupees(cfg):
    rows = ["(a) Share Capital 3 50,000,000 50,000,000", "(b) Reserves and Surplus 4 21,456,789 19,876,543",
            "Total Equity 71,456,789 69,876,543", "(a) Long-Term Borrowings 5 8,765,432 9,876,543",
            "Total Assets 89,014,180 87,654,321", "(a) Inventories 6 12,345,678 11,234,567"]
    page = _bs_page(41, None, rows + ["Place: Ahmedabad", "(In Rs.)"])
    cash_flow = {"page": 45, "method": "text", "text": "\n".join(
        ["Cash Flow Statement for the year ended 31st March 2020", "(In Lacs)", "Particulars 2019-20 2018-19"]
        + [f"Line {i} {20 + i}.00 {19 + i}.00" for i in range(8)])}
    got, info = _bs_unit([page, cash_flow], cfg)
    assert info["unit"] == "rupee" and got[("total_assets", 2020)] == pytest.approx(8.901418)


def test_a_unit_borrowed_from_another_statement_yields_to_rupee_magnitudes(cfg):
    # the balance sheet has no caption, the cash flow is in lakhs, the figures are rupees
    cash_flow = {"page": 45, "method": "text", "text": "\n".join(
        ["Cash Flow Statement for the year ended 31st March 2020", "(In Lacs)", "Particulars 2019-20 2018-19"]
        + [f"Line {i} {20 + i}.00 {19 + i}.00" for i in range(8)])}
    got, info = _bs_unit([_bs_page(63, None, RUPEE_ROWS), cash_flow], cfg)
    assert info["unit"] == "rupee" and "unit_inferred_rupee_from_magnitude" in info["flags"]
    assert got[("total_assets", 2020)] == pytest.approx(78.4454285)


def test_unit_checks_see_filings_outside_the_sample_years(cfg):
    cfg = {**cfg, "financials": {**cfg["financials"], "xbrl_priority": "first"}}
    figs = _pdf(1177.99, 0.95)          # FY2019 report in crore: right; FY2018-19 filings 100x too small
    ex = pd.DataFrame([("BSE1", 2018, "total_assets", 11.94), ("BSE1", 2019, "total_assets", 11.7799)],
                      columns=["firm_id", "fy", "field", "value_cr"])
    evidence = pd.concat([ex, pd.DataFrame([("BSE1", 2020, "total_assets", 1352.15)], columns=ex.columns)])
    resolved, _ = resolve_figures(figs, pd.DataFrame(), cfg, ex)
    got = resolved.set_index(["fy", "field"])
    assert got.loc[(2019, "total_assets"), "unit_check"] == "pdf_unit_suspect"   # the two slipped filings agree
    resolved, _ = resolve_figures(figs, pd.DataFrame(), cfg, ex, evidence=evidence)
    got = resolved.set_index(["fy", "field"])
    assert got.loc[(2019, "total_assets"), "unit_check"] == "xbrl_unit_suspect"
    assert got.loc[(2019, "total_assets"), "value_cr"] == pytest.approx(1177.99)
