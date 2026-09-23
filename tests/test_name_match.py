import pandas as pd

from bpp.cohort.name_match import match_debtors_to_universe
from bpp.scrape.listed import build_universe


def test_match_statuses(cfg):
    bse = pd.DataFrame([
        {"bse_code": "1", "company_name": "Zenthra Infraprojects Limited", "isin": "INE1"},
        {"bse_code": "2", "company_name": "Tarangi Infra Limited", "isin": "INE2"},
    ])
    nse = pd.DataFrame([{"nse_symbol": "ZENTHRA", "company_name": "Zenthra Infraprojects Limited", "isin": "INE1"}])
    uni = build_universe(bse, nse)
    assert set(uni["firm_id"]) == {"BSE1", "BSE2"}
    assert uni.set_index("firm_id").loc["BSE1", "nse_symbol"] == "ZENTHRA"      # merged on ISIN

    debtors = pd.DataFrame({
        "corporate_debtor": ["ZENTHRA INFRAPROJECTS LTD.", "Tarang Infra Limited", "Quietbrook Traders Limited"],
        "cirp_announcement_date": pd.to_datetime(["2019-08-14", "2020-07-22", "2021-01-01"]),
    })
    out = match_debtors_to_universe(debtors, uni, cfg).set_index("corporate_debtor")
    assert out.loc["ZENTHRA INFRAPROJECTS LTD.", "match_status"] == "auto_accepted"
    assert out.loc["ZENTHRA INFRAPROJECTS LTD.", "accept"] == "Y"
    assert out.loc["Tarang Infra Limited", "match_status"] == "needs_review"      # one letter off: a human decides
    assert out.loc["Quietbrook Traders Limited", "match_status"] == "no_match"
