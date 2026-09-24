"""Tests for the real-data sources: IBBI export, BSE API records, exchange XBRL, CIN matching."""

from __future__ import annotations

import json
import random

import pandas as pd
import pytest

from bpp.cohort.cin_match import (cins_in_text, confirm_with_report_cin, lcs_len, match_debtors, ratio,
                                  token_sort_ratio)
from bpp.features.financials import SOURCE_EXCHANGE_XBRL, SOURCE_PRIMARY, resolve_figures
from bpp.sources.bse import (annual_standalone_xbrl, parse_annual_report_list, parse_company_headers,
                             parse_result_archive, parse_traded_members)
from bpp.sources.ibbi_export import cirp_debtors, clean_cin, possibly_listed, read_ibbi_export
from bpp.sources.xbrl_results import parse_results_xbrl, result_to_rows

# --------------------------------------------------------------------------- IBBI export
IBBI_TSV = (
    "Announcement Type\t Date of Announcement\t Last date of Submission\t Name of Corporate Debtor\t CIN No.\t"
    " Name of Applicant\t Name of Insolvency Professional\t Address of Insolvency Professional\t Remarks\n"
    "Public Announcement of Corporate Insolvency Resolution Process\t 21-09-2026\t 02-10-2026\t"
    " AURI GROW INDIA LIMITED\t L68100MP2016PLC041592\t Naksh Steel Limited\t Rajesh\t Addr\t \n"
    "Public Announcement of Corporate Insolvency Resolution Process\t 10-01-2019\t 24-01-2019\t"
    " Auri Grow India Ltd.\t L68100MP2016PLC041592\t Bank\t IP\t Addr\t \n"
    "Public Announcement of Corporate Insolvency Resolution Process\t 05-05-2020\t \t"
    " PS IT INFRASTRUCTURE &amp; SERVICES LIMITED\t \t Self\t IP\t Addr\t \n"
    "Public Announcement of Corporate Insolvency Resolution Process\t 05-06-2021\t \t"
    " Small Traders Private Limited\t U51100DL2010PTC123456\t Creditor\t IP\t Addr\t \n"
)


@pytest.fixture
def ibbi_file(tmp_path):
    p = tmp_path / "ibbi.tsv"
    p.write_text(IBBI_TSV, encoding="utf-8")
    return p


def test_the_ibbi_export_is_read_with_cin_fields(ibbi_file):
    df = read_ibbi_export(ibbi_file)
    assert len(df) == 4
    assert df.loc[0, "cin"] == "L68100MP2016PLC041592"
    assert bool(df.loc[0, "cin_listed"]) and df.loc[0, "cin_nic"] == "68100"
    assert df.loc[0, "cin_company_type"] == "public"
    assert df.loc[0, "announcement_date"] == pd.Timestamp("2026-09-21")
    # HTML entities in names are decoded before normalising
    assert "&amp;" not in df.loc[2, "corporate_debtor"]
    assert pd.isna(df.loc[2, "cin"])


def test_a_debtor_is_one_row_keyed_by_cin_with_its_earliest_date(ibbi_file):
    deb = cirp_debtors(read_ibbi_export(ibbi_file))
    auri = deb[deb["cin"] == "L68100MP2016PLC041592"].iloc[0]
    assert auri["n_announcements"] == 2
    assert auri["cirp_announcement_date"] == pd.Timestamp("2019-01-10")   # earliest, not latest


def test_possibly_listed_keeps_l_cins_and_unknowns_but_not_u_cins(ibbi_file):
    pl = possibly_listed(cirp_debtors(read_ibbi_export(ibbi_file)))
    names = set(pl["corporate_debtor"])
    assert any("AURI" in n.upper() for n in names)
    assert any("PS IT" in n for n in names)                 # no CIN, not private: cannot rule out
    assert not any("Small Traders" in n for n in names)     # U CIN = never listed
    assert set(pl["listing_evidence"]) == {"cin_L", "no_cin"}


def test_clean_cin_rejects_malformed_values():
    assert clean_cin(" l68100mp2016plc041592 ") == "L68100MP2016PLC041592"
    assert clean_cin("L68100MP2016PLC04159") is None
    assert clean_cin("NA") is None


# --------------------------------------------------------------------------- name matching
def _lcs_dp(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, ca in enumerate(a):
        for j, cb in enumerate(b):
            dp[i + 1][j + 1] = dp[i][j] + 1 if ca == cb else max(dp[i][j + 1], dp[i + 1][j])
    return dp[len(a)][len(b)]


def test_bit_parallel_lcs_matches_dynamic_programming():
    rng = random.Random(3)
    for _ in range(500):
        a = "".join(rng.choice("AB C") for _ in range(rng.randint(0, 30)))
        b = "".join(rng.choice("AB C") for _ in range(rng.randint(0, 30)))
        assert lcs_len(a, b) == _lcs_dp(a, b)


def test_scores_match_rapidfuzz_definitions():
    assert ratio("this is a test", "this is a test!") == pytest.approx(96.5517, abs=1e-3)
    assert token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear") == 100
    assert token_sort_ratio("TARANG INFRA", "TARANGI INFRA") == pytest.approx(96.0)


def test_match_debtors_finds_the_listing_and_grades_it():
    universe = pd.DataFrame({
        "firm_id": ["BSE1", "BSE2", "BSE3"],
        "company_name": ["Lakshmi Precision Screws Ltd.", "Rain Industries Limited", "Alpha Steel Ltd"],
        "name_norm": ["LAKSHMI PRECISION SCREWS", "RAIN INDUSTRIES", "ALPHA STEEL"],
        "bse_code": ["1", "2", "3"], "nse_symbol": [None, None, None], "isin": [None] * 3,
        "status": ["Active"] * 3, "industry": [None] * 3,
    })
    debtors = pd.DataFrame({
        "corporate_debtor": ["Alpha Steel Limited", "Rathi Industries Limited"],
        "name_norm": ["ALPHA STEEL", "RATHI INDUSTRIES"], "cin": ["L1", "L2"],
        "cirp_announcement_date": ["2020-01-01", "2020-01-01"], "listing_evidence": ["cin_L", "cin_L"],
        "former_names": ["", ""], "all_names": ["", ""],
    })
    m = match_debtors(debtors, universe).set_index("corporate_debtor")
    assert m.loc["Alpha Steel Limited", "match_status"] == "auto_accepted"
    assert m.loc["Alpha Steel Limited", "firm_id"] == "BSE3"
    # a near-miss name is never auto-accepted: it waits for the report-CIN check
    assert m.loc["Rathi Industries Limited", "match_status"] in ("needs_review", "no_match")


def test_report_cin_check():
    text = "CIN: L14106UP1995PLC019017 ... Subsidiary: U45200DL2007PTC123456"
    assert cins_in_text(text) == ["L14106UP1995PLC019017", "U45200DL2007PTC123456"]
    assert confirm_with_report_cin("L14106UP1995PLC019017", text)[0] == "confirmed"
    # listed -> unlisted changes the first letter, not the registration number
    assert confirm_with_report_cin("U14106UP1995PLC019017", text)[0] == "confirmed_reg_no"
    assert confirm_with_report_cin("L99999MH2000PLC111111", text) == ("mismatch", "L14106UP1995PLC019017")
    assert confirm_with_report_cin("L99999MH2000PLC111111", "nothing")[0] == "cin_not_found"


# --------------------------------------------------------------------------- BSE API records
def _jsonl(tmp_path, name, records):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        for key, body in records:
            f.write(json.dumps({"key": key, "http": 200, "body": json.dumps(body)}) + "\n")
    return p


def test_company_header_gives_the_industry_path(tmp_path):
    p = _jsonl(tmp_path, "h.jsonl", [
        ("532532", {"SecurityId": "JPASSOCIAT", "ISIN": "INE455F01025", "Industry": "Civil Construction",
                    "Sector": "Industrials", "IndustryNew": "Construction", "IGroup": "Construction",
                    "ISubGroup": "Civil Construction"}),
        ("531734", {"SecurityId": "NICCOCORQ", "Industry": "-", "Sector": "", "ISubGroup": ""}),
    ])
    lst = pd.DataFrame({"isubgroup": ["Civil Construction"], "isubgroup_code": ["IN070203001"]})
    h = parse_company_headers(p, lst).set_index("bse_code")
    assert h.loc["532532", "isubgroup_code"] == "IN070203001"
    assert pd.isna(h.loc["531734", "isubgroup"])          # delisted scrips often have none


def test_result_archive_picks_the_full_year_standalone_filing(tmp_path):
    rows = [
        {"Year": "2023-2024", "Quarter": "Standalone-Mar-24;MC2023-2024;121.50;D",
         "stand_xbrl_link": "/XBRLFILES/FourOneUploadDocument/Main_Ind_As_532532_1.xml",
         "Filing_Date_Time": "2024-05-13T10:51:51", "qtr": 121.5},
        {"Year": "2023-2024", "Quarter": "Standalone-Mar-24;MQ2023-2024;121.00;D",
         "stand_xbrl_link": "/XBRLFILES/FourOneUploadDocument/Main_Ind_As_532532_1.xml",
         "Filing_Date_Time": "2024-05-13T10:51:10", "qtr": 121},
        {"Year": "2023-2024", "Quarter": "Consolidated-Mar-24;MC2023-2024;121.50;c",
         "conso_xbrl_link": "/XBRLFILES/x.xml", "Filing_Date_Time": "2024-05-13T10:53:43", "qtr": 121.5},
        {"Year": "2016-2017", "Quarter": "Standalone-Mar-17;MC2016-2017;93.50;D",
         "stand_xbrl_link": "/XBRLFILES/FourOneUploadDocument/", "Filing_Date_Time": "2017-06-05T10:00:00"},
    ]
    a = annual_standalone_xbrl(parse_result_archive(_jsonl(tmp_path, "r.jsonl", [("532532", {"Table": rows})])))
    got = a.set_index("fy")
    assert set(got.index) == {2024, 2017}
    assert got.loc[2024, "xbrl_url"].endswith("Main_Ind_As_532532_1.xml")
    assert got.loc[2024, "xbrl_url"].startswith("https://www.bseindia.com/")
    assert pd.isna(got.loc[2017, "xbrl_url"])              # an empty folder link is not a file


def test_annual_report_links_are_normalised(tmp_path):
    rows = [{"Year": "2023", "PDFDownload": "https://www.bseindia.com/xml-data/corpfiling/AttachHis/\\\\ab-cd.pdf",
             "Fld_AuthoriseDate": "2023-08-06T14:39:03"}]
    ar = parse_annual_report_list(_jsonl(tmp_path, "a.jsonl", [("500325", {"Table": rows})]))
    assert ar.loc[0, "url"] == "https://www.bseindia.com/xml-data/corpfiling/AttachHis/ab-cd.pdf"
    assert ar.loc[0, "fy"] == 2023


def test_traded_members_are_parsed(tmp_path):
    p = tmp_path / "t.jsonl"
    body = ('"bse$#$ATHERENERG,-2.81,ATHERENERG,1546.65,1546.70,1484.00,1499.80,-43.30,544397,1499.24,u,ATHERENERG,-2.81'
            '|TVSMOTOR,-0.01,TVSMOTOR,4178.95,4178.95,4100.00,4129.45,-0.55,532343,4134.49,u,TVSMOTOR,-0.01"')
    p.write_text(json.dumps({"key": "IN020101002", "http": 200, "body": body}) + "\n", encoding="utf-8")
    t = parse_traded_members(p)
    assert set(t["bse_code"]) == {"544397", "532343"}


# --------------------------------------------------------------------------- XBRL
XBRL = """<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:in-bse-fin="http://www.bseindia.com/xbrl/fin"
 xmlns:xbrldi="http://xbrl.org/2006/xbrldi">
<xbrli:context id="OneD"><xbrli:entity><xbrli:identifier scheme="s">1</xbrli:identifier></xbrli:entity>
 <xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate><xbrli:endDate>2024-03-31</xbrli:endDate></xbrli:period></xbrli:context>
<xbrli:context id="FourD"><xbrli:entity><xbrli:identifier scheme="s">1</xbrli:identifier></xbrli:entity>
 <xbrli:period><xbrli:startDate>2023-04-01</xbrli:startDate><xbrli:endDate>2024-03-31</xbrli:endDate></xbrli:period></xbrli:context>
<xbrli:context id="OneI"><xbrli:entity><xbrli:identifier scheme="s">1</xbrli:identifier></xbrli:entity>
 <xbrli:period><xbrli:instant>2024-03-31</xbrli:instant></xbrli:period></xbrli:context>
<xbrli:context id="Seg"><xbrli:entity><xbrli:identifier scheme="s">1</xbrli:identifier><xbrli:segment>
 <xbrldi:explicitMember dimension="a:Axis">a:M</xbrldi:explicitMember></xbrli:segment></xbrli:entity>
 <xbrli:period><xbrli:startDate>2023-04-01</xbrli:startDate><xbrli:endDate>2024-03-31</xbrli:endDate></xbrli:period></xbrli:context>
<in-bse-fin:DateOfStartOfFinancialYear contextRef="OneD">2023-04-01</in-bse-fin:DateOfStartOfFinancialYear>
<in-bse-fin:DateOfEndOfFinancialYear contextRef="OneD">2024-03-31</in-bse-fin:DateOfEndOfFinancialYear>
<in-bse-fin:NatureOfReportStandaloneConsolidated contextRef="OneD">Standalone</in-bse-fin:NatureOfReportStandaloneConsolidated>
<in-bse-fin:DeclarationOfUnmodifiedOpinionOrStatementOnImpactOfAuditQualification contextRef="OneD">Declaration of unmodified opinion</in-bse-fin:DeclarationOfUnmodifiedOpinionOrStatementOnImpactOfAuditQualification>
<in-bse-fin:RevenueFromOperations contextRef="OneD" unitRef="INR" decimals="-5">9354200000</in-bse-fin:RevenueFromOperations>
<in-bse-fin:RevenueFromOperations contextRef="FourD" unitRef="INR" decimals="-5">35479800000</in-bse-fin:RevenueFromOperations>
<in-bse-fin:RevenueFromOperations contextRef="Seg" unitRef="INR" decimals="-5">1</in-bse-fin:RevenueFromOperations>
<in-bse-fin:ProfitBeforeTax contextRef="FourD" unitRef="INR" decimals="-5">-13196000000</in-bse-fin:ProfitBeforeTax>
<in-bse-fin:Assets contextRef="OneI" unitRef="INR" decimals="-5">397598600000</in-bse-fin:Assets>
</xbrli:xbrl>"""


def test_xbrl_reads_the_full_year_not_the_march_quarter():
    r = parse_results_xbrl(XBRL)
    assert r.fy == 2024
    assert r.values_cr["revenue"] == pytest.approx(3547.98)       # not the quarter's 935.42
    assert r.values_cr["pbt"] == pytest.approx(-1319.6)
    assert r.values_cr["total_assets"] == pytest.approx(39759.86)
    assert r.meta["standalone"] is True
    assert r.meta["audit_modified"] is False


def test_xbrl_rows_have_the_supplement_layout():
    rows = result_to_rows(parse_results_xbrl(XBRL), "BSE532532", "u")
    assert {r["field"] for r in rows} == {"revenue", "pbt", "total_assets"}
    assert all(r["fy"] == 2024 and r["firm_id"] == "BSE532532" for r in rows)


# --------------------------------------------------------------------------- XBRL priority
def _pdf_figure(value):
    return pd.DataFrame([{
        "firm_id": "BSE1", "fy": 2024, "field": "total_assets", "value_cr": value,
        "source": SOURCE_PRIMARY, "source_doc_id": "BSE1_FY2024", "page": 40, "statement": "balance_sheet",
        "statement_scope": "standalone", "label": "Total Assets", "match_how": "pattern", "match_score": 1.0,
        "unit": "crore", "unit_confidence": 0.9, "printed": str(value), "parse_confidence": 1.0,
        "parse_flags": "", "ocr_pages": 0}])


def test_exchange_xbrl_wins_when_priority_is_first_and_the_pdf_value_is_kept(cfg):
    cfg = {**cfg, "financials": {**cfg["financials"], "xbrl_priority": "first"}}
    exch = pd.DataFrame([{"firm_id": "BSE1", "fy": 2024, "field": "total_assets", "value_cr": 100.0,
                          "source_url": "u"}])
    resolved, _ = resolve_figures(_pdf_figure(110.0), pd.DataFrame(), cfg, exch)
    row = resolved.iloc[0]
    assert row["source"] == SOURCE_EXCHANGE_XBRL and row["value_cr"] == 100.0
    assert row["pdf_value_cr"] == 110.0
    assert row["xbrl_pdf_diff"] == pytest.approx(10 / 110)


def test_under_gap_fill_the_report_still_wins(cfg):
    cfg = {**cfg, "financials": {**cfg["financials"], "xbrl_priority": "gap_fill"}}
    exch = pd.DataFrame([{"firm_id": "BSE1", "fy": 2024, "field": "total_assets", "value_cr": 100.0},
                         {"firm_id": "BSE1", "fy": 2024, "field": "revenue", "value_cr": 50.0}])
    resolved, _ = resolve_figures(_pdf_figure(110.0), pd.DataFrame(), cfg, exch)
    got = resolved.set_index("field")
    assert got.loc["total_assets", "source"] == SOURCE_PRIMARY and got.loc["total_assets", "value_cr"] == 110.0
    assert got.loc["revenue", "source"] == SOURCE_EXCHANGE_XBRL         # a gap is still filled


def test_collect_parts_are_read_together_latest_wins(tmp_path):
    import json
    from bpp.sources.bse import iter_jsonl
    from bpp.sources.jobs import write_jobs, bse_result_archive
    base = tmp_path / "arch.jsonl"
    base.write_text(json.dumps({"key": "1", "body": "old"}) + "\n" + json.dumps({"key": "2", "body": "b"}) + "\n")
    (tmp_path / "arch_012_x_a.jsonl").write_text(json.dumps({"key": "1", "body": "new"}) + "\n")
    got = {r["key"]: r["body"] for r in iter_jsonl(base)}
    assert got == {"1": "new", "2": "b"}
    paths = write_jobs(tmp_path / "jobs", "012_x", [bse_result_archive(str(c)) for c in range(5)], "n", per_job=2)
    assert [p.stem for p in paths] == ["012_x_a", "012_x_b", "012_x_c"]
    job = json.loads(paths[1].read_text())
    assert job["items"][0]["collect"].endswith("bse_result_archive_012_x_b.jsonl")
