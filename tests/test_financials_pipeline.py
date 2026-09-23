"""Phase 3 end to end on synthetic reports: as-first-published, gap-filling,
restatements, missing-data flags, pair exclusion and the spot-check sheet."""

import json

import pandas as pd
import pytest

from bpp.features.financials import (SOURCE_COMPARATIVE, SOURCE_NONE, SOURCE_PRIMARY,
                                     build_financials, make_spot_check_sheet,
                                     report_coverage, score_spot_check)
from bpp.synthetic import statement_pages, year_figures

PAIR = [("BSE900101", "distressed", 1, True), ("BSE900201", "healthy", 0, False)]
YEARS = [2017, 2018, 2019]


def _write_report(paths, firm_id, fy, distressed, seed=7, restate_prior=0.0):
    pages = [{"page": 10 + i, "method": "text", "text": text}
             for i, text in enumerate(statement_pages(firm_id, fy, distressed, seed,
                                                      restate_prior=restate_prior))]
    (paths.pages / f"{firm_id}_FY{fy}.json").write_text(
        json.dumps({"doc_id": f"{firm_id}_FY{fy}", "firm_id": firm_id, "fy": fy, "pages": pages}),
        encoding="utf-8")


def _write_frame(paths):
    rows = [{"pair_id": "P1", "firm_id": firm, "fy": fy, "role": role, "label": label,
             "company_name": firm, "horizon": f"t-{2020 - fy}"}
            for firm, role, label, _ in PAIR for fy in YEARS]
    pd.DataFrame(rows).to_csv(paths.documents_labeled, index=False)


@pytest.fixture
def cohort(cfg, paths):
    """Every report present, so the happy path can be asserted before the gaps."""
    for firm, _, _, distressed in PAIR:
        for fy in YEARS:
            _write_report(paths, firm, fy, distressed)
    _write_frame(paths)
    return build_financials(paths, cfg)


def test_one_row_per_company_year(cohort):
    financials = cohort["financials"]
    assert len(financials) == len(PAIR) * len(YEARS)
    assert set(financials["fy"]) == set(YEARS)


def test_figures_are_stored_as_first_published(cohort):
    """FY2018 must come from the FY2018 report, never from FY2019's comparative."""
    figures = cohort["figures"]
    row = figures[(figures.firm_id == "BSE900101") & (figures.fy == 2018)
                  & (figures.field == "total_assets")].iloc[0]
    assert row["source"] == SOURCE_PRIMARY
    assert row["source_doc_id"] == "BSE900101_FY2018"
    assert row["value_cr"] == pytest.approx(year_figures(2018, True, 7)["total_assets"])


def test_a_consistent_comparative_raises_no_restatement(cohort):
    figures = cohort["figures"]
    row = figures[(figures.firm_id == "BSE900201") & (figures.fy == 2018)
                  & (figures.field == "total_assets")].iloc[0]
    assert bool(row["has_comparative"]) is True
    assert bool(row["restated"]) is False


def test_every_company_year_has_a_page_number(cohort):
    figures = cohort["figures"]
    assert figures["page"].notna().all()


def test_ratios_are_computed_for_every_company_year(cohort):
    ratios = cohort["ratios"]
    assert len(ratios) == len(PAIR) * len(YEARS)
    assert ratios["altman_z_em"].notna().all()


def test_the_distressed_firm_scores_worse_than_its_peer(cohort):
    """Not a claim about the model -- a check that the signs point the right way."""
    ratios = cohort["ratios"].set_index(["firm_id", "fy"])
    assert ratios.loc[("BSE900101", 2019), "altman_z_em"] < ratios.loc[("BSE900201", 2019), "altman_z_em"]
    assert ratios.loc[("BSE900101", 2019), "current_ratio"] < ratios.loc[("BSE900201", 2019), "current_ratio"]


def test_negative_equity_voids_debt_to_equity_but_not_the_row(cohort):
    ratios = cohort["ratios"].set_index(["firm_id", "fy"])
    distressed = ratios.loc[("BSE900101", 2019)]
    assert pd.isna(distressed["debt_to_equity"])
    assert bool(distressed["negative_equity"]) is True
    assert pd.notna(distressed["altman_z_em"])


# --------------------------------------------------------------------------- restatement
def test_a_restated_comparative_is_flagged_and_does_not_overwrite(cfg, paths):
    for firm, _, _, distressed in PAIR:
        for fy in YEARS:
            # the healthy firm restates its FY2018 comparative by +4% in the FY2019 report
            restate = 0.04 if (not distressed and fy == 2019) else 0.0
            _write_report(paths, firm, fy, distressed, restate_prior=restate)
    _write_frame(paths)
    figures = build_financials(paths, cfg)["figures"]

    row = figures[(figures.firm_id == "BSE900201") & (figures.fy == 2018)
                  & (figures.field == "total_assets")].iloc[0]
    assert bool(row["restated"]) is True
    assert row["restatement_diff"] == pytest.approx(0.04, abs=0.01)
    assert row["value_cr"] == pytest.approx(year_figures(2018, False, 7)["total_assets"])
    assert row["confidence"] < 1.0


# --------------------------------------------------------------------------- gaps
def test_a_missing_report_is_filled_from_the_next_years_comparative(cfg, paths):
    for firm, _, _, distressed in PAIR:
        for fy in YEARS:
            if distressed and fy == 2017:          # its own report was never found
                continue
            _write_report(paths, firm, fy, distressed)
    _write_frame(paths)
    tables = build_financials(paths, cfg)

    row = tables["figures"].query("firm_id == 'BSE900101' and fy == 2017 and field == 'total_assets'").iloc[0]
    assert row["source"] == SOURCE_COMPARATIVE
    assert row["source_doc_id"] == "BSE900101_FY2018"

    year = tables["financials"].query("firm_id == 'BSE900101' and fy == 2017").iloc[0]
    assert bool(year["financials_missing"]) is False
    assert bool(year["included_financials"]) is True


def test_an_unrecoverable_year_is_flagged_and_kept(cfg, paths):
    """Missing reports are a feature, not rows to drop (project plan, issue 7)."""
    for firm, _, _, distressed in PAIR:
        for fy in YEARS:
            if distressed and fy in (2017, 2018):   # no report, and no comparative either
                continue
            _write_report(paths, firm, fy, distressed)
    _write_frame(paths)
    financials = build_financials(paths, cfg)["financials"]

    year = financials.query("firm_id == 'BSE900101' and fy == 2017").iloc[0]
    assert bool(year["financials_missing"]) is True
    assert year["financials_source"] == SOURCE_NONE
    assert year["financials_exclude_reason"] == "missing_financials"
    assert len(financials) == len(PAIR) * len(YEARS)      # the row is kept


def test_the_peers_year_is_dropped_when_its_partner_is_missing(cfg, paths):
    """Phase 2's keep_pairs_aligned rule, applied to financials."""
    for firm, _, _, distressed in PAIR:
        for fy in YEARS:
            if distressed and fy in (2017, 2018):
                continue
            _write_report(paths, firm, fy, distressed)
    _write_frame(paths)
    financials = build_financials(paths, cfg)["financials"]

    peer = financials.query("firm_id == 'BSE900201' and fy == 2017").iloc[0]
    assert peer["financials_exclude_reason"] == "pair_partner_missing_financials"
    assert bool(peer["included_financials"]) is False
    assert pd.notna(peer["total_assets"])                 # the figures are still there


def test_unrecoverable_company_years_are_counted_by_class(cfg, paths):
    for firm, _, _, distressed in PAIR:
        for fy in YEARS:
            if distressed and fy in (2017, 2018):
                continue
            _write_report(paths, firm, fy, distressed)
    _write_frame(paths)
    tables = build_financials(paths, cfg)
    summary = report_coverage(tables["financials"], tables["ratios"])
    assert summary["unrecoverable_distressed"] == 1
    assert summary["unrecoverable_healthy"] == 0


def test_the_xbrl_supplement_fills_a_gap(cfg, paths):
    for firm, _, _, distressed in PAIR:
        for fy in YEARS:
            if distressed and fy in (2017, 2018):
                continue
            _write_report(paths, firm, fy, distressed)
    _write_frame(paths)
    pd.DataFrame([{"firm_id": "BSE900101", "fy": 2017, "field": "total_assets", "value_cr": 9100.0},
                  {"firm_id": "BSE900101", "fy": 2017, "field": "current_assets", "value_cr": 3200.0},
                  {"firm_id": "BSE900101", "fy": 2017, "field": "current_liabilities", "value_cr": 4000.0},
                  {"firm_id": "BSE900101", "fy": 2017, "field": "total_equity", "value_cr": -400.0}]
                 ).to_csv(paths.xbrl_financials, index=False)
    financials = build_financials(paths, cfg)["financials"]

    year = financials.query("firm_id == 'BSE900101' and fy == 2017").iloc[0]
    assert year["financials_source"] == "manual_xbrl"
    assert bool(year["financials_missing"]) is False
    assert year["total_assets"] == pytest.approx(9100.0)


def test_the_xbrl_supplement_never_overrides_a_published_figure(cfg, paths):
    for firm, _, _, distressed in PAIR:
        for fy in YEARS:
            _write_report(paths, firm, fy, distressed)
    _write_frame(paths)
    pd.DataFrame([{"firm_id": "BSE900101", "fy": 2019, "field": "total_assets", "value_cr": 1.0}]
                 ).to_csv(paths.xbrl_financials, index=False)
    financials = build_financials(paths, cfg)["financials"]

    year = financials.query("firm_id == 'BSE900101' and fy == 2019").iloc[0]
    assert year["total_assets"] == pytest.approx(year_figures(2019, True, 7)["total_assets"])


# --------------------------------------------------------------------------- spot check
def test_the_spot_check_sheet_samples_and_gives_page_numbers(cfg, paths, cohort):
    cfg["financials"]["spot_check_fraction"] = 0.5
    sheet = make_spot_check_sheet(paths, cfg, cohort["figures"], cohort["financials"],
                                  overwrite=True)
    sampled = sheet[["firm_id", "fy"]].drop_duplicates()
    assert len(sampled) == 3                                  # 50% of 6 company-years
    assert sheet["page"].notna().all()
    for column in ("value_correct", "page_correct", "true_value_cr", "notes"):
        assert column in sheet.columns and (sheet[column] == "").all()


def test_the_spot_check_sheet_is_not_silently_overwritten(cfg, paths, cohort):
    make_spot_check_sheet(paths, cfg, cohort["figures"], cohort["financials"], overwrite=True)
    with pytest.raises(FileExistsError):
        make_spot_check_sheet(paths, cfg, cohort["figures"], cohort["financials"])


def test_the_filled_spot_check_sheet_scores(cfg, paths, cohort):
    sheet = make_spot_check_sheet(paths, cfg, cohort["figures"], cohort["financials"],
                                  overwrite=True)
    sheet["value_correct"] = ["Y"] * (len(sheet) - 1) + ["N"]
    sheet["page_correct"] = "Y"
    sheet.to_csv(paths.financials_qa_sheet, index=False)
    scores = score_spot_check(paths)
    assert {"field", "n_checked", "value_accuracy", "page_accuracy"} <= set(scores.columns)
    assert scores["n_checked"].sum() == len(sheet)


def test_the_manual_financials_input_is_never_written_to(cfg, paths, cohort):
    """data/manual/firm_financials.csv is the team's Phase 1 input, not a Phase 3 output."""
    assert not paths.financials.exists()
