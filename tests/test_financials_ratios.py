import pandas as pd
import pytest

from bpp.features.ratios import (ALTMAN_DISTRESS, ALTMAN_EM_COEFFICIENTS, ALTMAN_SAFE,
                                 INDICATOR_FIELDS, RATIO_FIELDS,
                                 altman_z_em, compute_ratios, derive, winsorise, winsorise_bounds)

# A firm on its way into the NCLT: negative net worth, EBITDA gone, interest unpaid.
DISTRESSED = dict(current_assets=2845.70, current_liabilities=4516.90, inventories=1204.55,
                  cash_and_equivalents=88.12, total_assets=7856.15,
                  borrowings_long_term=4120.00, borrowings_short_term=2310.50,
                  total_equity=-780.75, other_equity=-1230.75, revenue=3200.00,
                  pbt=-1450.00, finance_costs=610.00, depreciation=380.00, net_profit=-1502.00)

HEALTHY = dict(current_assets=3100.00, current_liabilities=1250.00, inventories=700.00,
               cash_and_equivalents=540.00, total_assets=9200.00,
               borrowings_long_term=900.00, borrowings_short_term=350.00,
               total_equity=5600.00, other_equity=4900.00, revenue=7400.00,
               pbt=980.00, finance_costs=110.00, depreciation=300.00, net_profit=720.00)


def test_the_twelve_stream_c_ratios_are_all_present():
    """Table 7 of the progress report fixes stream C at exactly 12 values."""
    assert len(RATIO_FIELDS) == 12
    assert set(compute_ratios(HEALTHY).values) == set(RATIO_FIELDS)


def test_ebit_and_ebitda_are_built_from_the_stated_components():
    d = derive(DISTRESSED)
    assert d["ebit"] == pytest.approx(-1450.00 + 610.00)
    assert d["ebitda"] == pytest.approx(-1450.00 + 610.00 + 380.00)
    assert d["total_borrowings"] == pytest.approx(4120.00 + 2310.50)
    assert d["working_capital"] == pytest.approx(2845.70 - 4516.90)
    assert d["total_liabilities"] == pytest.approx(7856.15 - (-780.75))


def test_capital_employed_falls_back_to_the_schedule_iii_construction():
    d = derive(HEALTHY)
    assert d["capital_employed"] == pytest.approx(9200.00 - 1250.00)


def test_a_reported_capital_employed_is_preferred():
    d = derive({**HEALTHY, "capital_employed": 8000.0})
    assert d["capital_employed"] == pytest.approx(8000.0)


@pytest.mark.parametrize("ratio,expected", [
    ("current_ratio", 3100.00 / 1250.00),
    ("quick_ratio", (3100.00 - 700.00) / 1250.00),
    ("cash_to_assets", 540.00 / 9200.00),
    ("ebitda_margin", (980.00 + 110.00 + 300.00) / 7400.00),
    ("roce", (980.00 + 110.00) / (9200.00 - 1250.00)),
    ("roa", 720.00 / 9200.00),
    ("debt_to_equity", 1250.00 / 5600.00),
    ("debt_to_ebitda", 1250.00 / (980.00 + 110.00 + 300.00)),
    ("interest_coverage", (980.00 + 110.00) / 110.00),
    ("retained_earnings_to_assets", 4900.00 / 9200.00),
])
def test_each_ratio_on_a_healthy_firm(ratio, expected):
    assert compute_ratios(HEALTHY).values[ratio] == pytest.approx(expected)


def test_roa_uses_net_profit_not_ebit():
    """Kept comparable with Gupta (2022), the Indian IBC ratio benchmark."""
    result = compute_ratios(HEALTHY)
    assert result.values["roa"] == pytest.approx(720.00 / 9200.00)
    assert result.values["roa"] != pytest.approx((980.00 + 110.00) / 9200.00)


@pytest.mark.parametrize("ratio", ["debt_to_equity", "debt_to_ebitda"])
def test_ratios_refuse_a_negative_denominator(ratio):
    """A sign-inverted ratio would rank a wiped-out firm above a merely weak one."""
    result = compute_ratios(DISTRESSED)
    assert result.values[ratio] is None
    assert result.reasons[ratio] == "denominator_negative"


def test_interest_coverage_keeps_a_negative_numerator():
    """Negative coverage is meaningful: EBIT does not cover the interest bill."""
    result = compute_ratios(DISTRESSED)
    assert result.values["interest_coverage"] == pytest.approx((-1450.00 + 610.00) / 610.00)


def test_the_conditions_behind_an_empty_ratio_are_kept_as_features():
    result = compute_ratios(DISTRESSED)
    assert result.indicators["negative_equity"] is True
    assert result.indicators["negative_ebitda"] is True
    assert result.indicators["negative_working_capital"] is True
    assert set(INDICATOR_FIELDS) == set(result.indicators)


def test_indicators_are_none_rather_than_false_when_unknown():
    result = compute_ratios({"total_assets": 100.0})
    assert result.indicators["negative_equity"] is None


def test_promoter_pledge_is_reserved_for_the_shareholding_filings():
    result = compute_ratios(HEALTHY)
    assert "promoter_pledge" in result.values
    assert result.values["promoter_pledge"] is None
    assert result.reasons["promoter_pledge"] == "pending_shareholding_filings"


# --------------------------------------------------------------------------- Altman
def test_altman_em_coefficients_match_the_published_model():
    assert ALTMAN_EM_COEFFICIENTS == {"x1": 6.56, "x2": 3.26, "x3": 6.72,
                                      "x4": 1.05, "intercept": 3.25}


def test_altman_em_matches_a_hand_computation():
    total_assets = 7856.15
    x1 = (2845.70 - 4516.90) / total_assets
    x2 = -1230.75 / total_assets
    x3 = (-1450.00 + 610.00) / total_assets
    x4 = -780.75 / (total_assets - (-780.75))
    expected = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4 + 3.25
    assert altman_z_em(DISTRESSED)[0] == pytest.approx(expected, rel=1e-9)


def test_altman_keeps_a_negative_equity_term():
    """Z'' is linear, so negative equity should push the score down, not void it."""
    score, reason = altman_z_em(DISTRESSED)
    assert score is not None and reason == ""
    assert score < altman_z_em(HEALTHY)[0]


@pytest.mark.parametrize("figures,zone", [(DISTRESSED, "distress"), (HEALTHY, "safe")])
def test_altman_zones(figures, zone):
    assert compute_ratios(figures).altman_zone == zone


def test_altman_grey_zone():
    """Between 1.1 and 2.6 on the Z'' scale, the model says it does not know."""
    grey = dict(total_assets=1000.0, current_assets=450.0, current_liabilities=350.0,
                total_equity=300.0, other_equity=80.0, pbt=20.0, finance_costs=25.0)
    result = compute_ratios(grey)
    assert ALTMAN_DISTRESS < result.derived["altman_z_dprime"] < ALTMAN_SAFE
    assert result.altman_zone == "grey"


def test_the_zone_comes_from_the_unshifted_score_not_the_ems_scale():
    """The +3.25 constant only puts a D rating at zero on the EMS scale; it is not
    part of the discriminant. Reading the 2.6/1.1 cutoffs against the shifted score
    would lift every firm by 3.25 and call a failing one safe."""
    failing = dict(total_assets=1000.0, current_assets=300.0, current_liabilities=330.0,
                   total_equity=120.0, other_equity=10.0, pbt=-30.0, finance_costs=35.0,
                   depreciation=40.0, net_profit=-40.0)
    result = compute_ratios(failing)
    ems = result.values["altman_z_em"]
    dprime = result.derived["altman_z_dprime"]

    assert ems == pytest.approx(dprime + 3.25)
    assert ems > ALTMAN_SAFE                       # would read as "safe" on the wrong scale
    assert dprime < ALTMAN_DISTRESS
    assert result.altman_zone == "distress"


def test_altman_says_why_it_could_not_be_computed():
    score, reason = altman_z_em({"total_assets": 1000.0})
    assert score is None and "x1_missing" in reason


def test_altman_refuses_nonpositive_total_assets():
    score, reason = altman_z_em({"total_assets": 0.0})
    assert score is None and reason == "total_assets_missing_or_nonpositive"


# --------------------------------------------------------------------------- winsorising
def test_winsorise_bounds_are_learnt_on_the_training_fold():
    """Bounds must come from the training fold only, or the test fold leaks in."""
    train = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    low, high = winsorise_bounds(train, 0.0, 1.0)
    assert (low, high) == pytest.approx((1.0, 5.0))

    # a held-out outlier is clipped to the training bound; it must not move it
    held_out = pd.Series([1.0, 2.0, 500.0])
    assert held_out.clip(low, high).max() == pytest.approx(5.0)


def test_winsorise_clips_the_extremes():
    clipped = winsorise(pd.Series([1.0, 2.0, 3.0, 4.0, 100.0]), 0.0, 0.75)
    assert clipped.max() == pytest.approx(4.0)


def test_missing_inputs_give_missing_ratios_not_zeros():
    result = compute_ratios({})
    assert all(value is None for value in result.values.values())
    assert result.altman_zone is None
