import pandas as pd

from bpp.cohort.labels import assign_labels
from bpp.cohort.peers import build_sample_frame, match_peers, reference_fy


def _fin(rows):
    out = []
    for firm, ind, assets in rows:
        for fy in range(2012, 2022):
            out.append({"firm_id": firm, "company_name": firm, "fy": fy, "industry_code": ind, "total_assets": assets})
    return pd.DataFrame(out)


def test_reference_fy(cfg):
    assert reference_fy(pd.Timestamp("2019-08-14"), cfg["fiscal_year"]) == 2019
    assert reference_fy(pd.Timestamp("2019-03-31"), cfg["fiscal_year"]) == 2018   # FY must END before admission
    assert reference_fy(pd.Timestamp("2020-02-10"), cfg["fiscal_year"]) == 2019


def test_match_peers_rules(cfg):
    fin = _fin([("D1", "42101", 1000), ("H_close", "42101", 1100), ("H_far", "42101", 2000),
                ("H_other_ind", "13111", 1000), ("CIRP_FIRM", "42101", 1000)])
    distressed = pd.DataFrame({"firm_id": ["D1"], "matched_name": ["D1"],
                               "admission_date": [pd.Timestamp("2019-08-14")], "petition_date": [pd.NaT]})
    cohort, unmatched = match_peers(distressed, fin, cfg, exclude_firm_ids={"CIRP_FIRM"})
    assert unmatched.empty
    healthy = cohort[cohort["role"] == "healthy"]
    assert list(healthy["firm_id"]) == ["H_close"]           # within 30%, same industry, not a CIRP firm
    assert healthy.iloc[0]["match_quality"] == "exact_industry"


def test_no_peer_within_tolerance(cfg):
    fin = _fin([("D1", "42101", 1000), ("H_far", "42101", 5000)])
    distressed = pd.DataFrame({"firm_id": ["D1"], "admission_date": [pd.Timestamp("2019-08-14")],
                               "petition_date": [pd.NaT]})
    cohort, unmatched = match_peers(distressed, fin, cfg)
    assert cohort.empty and len(unmatched) == 1


def test_labels_exclusions_and_horizons(cfg):
    cohort = pd.DataFrame([
        {"pair_id": "P1", "firm_id": "D1", "company_name": "D1", "role": "distressed", "label": 1,
         "reference_date": "2019-08-14", "petition_date": None},
        {"pair_id": "P1", "firm_id": "H1", "company_name": "H1", "role": "healthy", "label": 0,
         "reference_date": "2019-08-14", "petition_date": None},
    ])
    frame = build_sample_frame(cohort, cfg)
    assert sorted(frame[frame.firm_id == "D1"]["fy"]) == [2016, 2017, 2018, 2019]
    docs = []
    for _, r in frame.iterrows():
        if r["firm_id"] == "D1" and r["fy"] == 2016:
            continue                                            # missing report
        pub = pd.Timestamp(f"{r['fy']}-08-30")
        docs.append({"doc_id": f"{r['firm_id']}_FY{r['fy']}", "status": "registered", "source": "manual",
                     "local_path": "x.pdf", "pub_date": pub.date().isoformat()})
    df, missing = assign_labels(frame, pd.DataFrame(docs), cohort, cfg)
    df = df.set_index("doc_id")
    # FY2019 report published 2019-08-30 is after admission -> excluded for both firms
    assert df.loc["D1_FY2019", "exclude_reason"].startswith("published_within")
    assert not df.loc["H1_FY2019", "included"]
    # D1 FY2016 missing -> H1 FY2016 dropped to keep the pair aligned
    assert df.loc["D1_FY2016", "exclude_reason"] == "missing_document"
    assert df.loc["H1_FY2016", "exclude_reason"] == "pair_partner_excluded"
    assert df.loc["D1_FY2018", "horizon"] == "t-1" and df.loc["H1_FY2018", "horizon"] == "t-1"
    assert df.loc["D1_FY2017", "horizon"] == "t-2"
    assert bool(df.loc["D1_FY2018", "within_12m"]) and not bool(df.loc["D1_FY2017", "within_12m"])
    assert len(missing) == 1


def test_petition_date_excludes_later_reports(cfg):
    cohort = pd.DataFrame([
        {"pair_id": "P1", "firm_id": "D1", "company_name": "D1", "role": "distressed", "label": 1,
         "reference_date": "2020-08-14", "petition_date": "2018-01-15"},
        {"pair_id": "P1", "firm_id": "H1", "company_name": "H1", "role": "healthy", "label": 0,
         "reference_date": "2020-08-14", "petition_date": None},
    ])
    frame = build_sample_frame(cohort, cfg)
    docs = [{"doc_id": f"{r.firm_id}_FY{r.fy}", "status": "registered", "source": "manual",
             "local_path": "x.pdf", "pub_date": f"{r.fy}-08-30"} for r in frame.itertuples()]
    df, _ = assign_labels(frame, pd.DataFrame(docs), cohort, cfg)
    df = df.set_index("doc_id")
    assert df.loc["D1_FY2018", "exclude_reason"] == "published_after_petition"
    assert df.loc["D1_FY2017", "included"]


def test_a_report_excluded_on_leakage_review_takes_its_pair_partner_with_it(cfg):
    cohort = pd.DataFrame([
        {"pair_id": "P1", "firm_id": "D1", "company_name": "D1", "role": "distressed", "label": 1,
         "reference_date": "2020-08-14", "petition_date": None},
        {"pair_id": "P1", "firm_id": "H1", "company_name": "H1", "role": "healthy", "label": 0,
         "reference_date": "2020-08-14", "petition_date": None},
    ])
    frame = build_sample_frame(cohort, cfg)
    docs = [{"doc_id": f"{r.firm_id}_FY{r.fy}", "status": "registered", "source": "manual",
             "local_path": "x.pdf", "pub_date": f"{r.fy}-08-30"} for r in frame.itertuples()]
    info = {"D1_FY2018": {"cirp_specific_mentions": 3}, "D1_FY2017": {"cirp_specific_mentions": 1}}
    review = {"D1_FY2018": "exclude", "D1_FY2017": "keep"}
    df, _ = assign_labels(frame, pd.DataFrame(docs), cohort, cfg, info, review)
    df = df.set_index("doc_id")
    assert df.loc["D1_FY2018", "exclude_reason"] == "leakage_review_excluded"
    assert df.loc["H1_FY2018", "exclude_reason"] == "pair_partner_excluded"
    assert df.loc["D1_FY2017", "included"] and not df.loc["D1_FY2017", "needs_leakage_review"]
    assert df.loc["D1_FY2017", "leakage_review"] == "keep"
