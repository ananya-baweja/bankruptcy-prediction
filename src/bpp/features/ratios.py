"""The 12 stream-C financial ratios.

``docs/00_project_plan.md`` fixes stream C at "~10 ratios + promoter pledge % +
Altman Z''", and Table 7 of the progress report names all twelve: current
ratio, quick ratio, cash to assets, EBITDA margin, ROCE, ROA, debt-to-equity,
debt-to-EBITDA, interest coverage, retained earnings to assets, promoter pledge
and the Altman Z-score. Promoter pledge comes from exchange shareholding
filings in a later step; its column exists here and is left empty.

All inputs are in ₹ crore. Ratios are scale-free, so the unit only matters in
that every input must share it -- which :mod:`bpp.features.financials` enforces
before calling in.

Why a ratio can come back empty
-------------------------------
A distressed Indian firm routinely has negative equity, negative EBITDA and, in
the worst year, no finance cost at all because lenders stopped accruing. Divide
by any of those and the number's sign inverts its meaning: debt-to-equity of
-5.3 on wiped-out equity sorts as *healthier* than +8.0 on thin equity, so a
model trained on it learns exactly the wrong thing. Those ratios are therefore
returned as ``None`` with a reason, and the condition is preserved separately
in the indicator flags so Phase 4 can impute and still see what happened.

Altman X4 is the exception: Z'' is a linear score, so a negative
equity-to-liabilities term pushes it down, which is the correct direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# --------------------------------------------------------------------------- Altman
# Altman's emerging-market score (Altman, Hartzell & Peck 1995; the "Z''" or EMS
# form of Altman 1968). Sales/total assets is dropped so the score does not
# reward asset-light industries, which is the point of the EM variant, and the
# +3.25 constant standardises it so a defaulted (D-rated) issuer sits at zero.
ALTMAN_EM_COEFFICIENTS = {"x1": 6.56, "x2": 3.26, "x3": 6.72, "x4": 1.05, "intercept": 3.25}

# Two scales, and popular sources conflate them.
#   Z''  = 6.56X1 + 3.26X2 + 6.72X3 + 1.05X4          the discriminant score
#   EMS  = Z'' + 3.25                                  the rating-equivalent scale
# The +3.25 exists only so that EMS = 0 corresponds to a bond rating of D; it
# carries no information, being a constant. The published safe/grey/distress
# cutoffs of 2.60 and 1.10 belong to Z''. Applying them to EMS shifts every firm
# up by 3.25 and almost nothing ever lands in distress -- a firm with negative
# working capital and near-zero EBIT would score "safe". So the zone is taken
# from Z'' and both numbers are reported. See docs/decisions_log.md.
ALTMAN_SAFE = 2.60
ALTMAN_DISTRESS = 1.10

#: the stream-C vector, in the order of Table 7
RATIO_FIELDS = [
    "current_ratio",
    "quick_ratio",
    "cash_to_assets",
    "ebitda_margin",
    "roce",
    "roa",
    "debt_to_equity",
    "debt_to_ebitda",
    "interest_coverage",
    "retained_earnings_to_assets",
    "promoter_pledge",
    "altman_z_em",
]

#: conditions that explain an empty ratio, kept as features in their own right
INDICATOR_FIELDS = [
    "borrowings_partial",
    "negative_equity",
    "negative_ebitda",
    "zero_finance_costs",
    "negative_working_capital",
    "negative_capital_employed",
]


@dataclass
class RatioResult:
    values: dict[str, float | None] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    indicators: dict[str, bool | None] = field(default_factory=dict)
    derived: dict[str, float | None] = field(default_factory=dict)
    altman_zone: str | None = None

    def set(self, name: str, value: float | None, reason: str = "") -> None:
        self.values[name] = value
        if reason:
            self.reasons[name] = reason


def _divide(numerator: float | None, denominator: float | None,
            positive_denominator: bool = False) -> tuple[float | None, str]:
    """Divide, or refuse and say why."""
    if numerator is None and denominator is None:
        return None, "both_missing"
    if numerator is None:
        return None, "numerator_missing"
    if denominator is None:
        return None, "denominator_missing"
    if denominator == 0:
        return None, "denominator_zero"
    if positive_denominator and denominator < 0:
        return None, "denominator_negative"
    return numerator / denominator, ""


def derive(fin: dict[str, float | None]) -> dict[str, float | None]:
    """Aggregates that more than one ratio needs."""
    get = fin.get
    pbt, finance_costs, depreciation = get("pbt"), get("finance_costs"), get("depreciation")

    ebit = None if (pbt is None or finance_costs is None) else pbt + finance_costs
    ebitda = None if (ebit is None or depreciation is None) else ebit + depreciation

    current_assets, current_liabilities = get("current_assets"), get("current_liabilities")
    working_capital = (None if (current_assets is None or current_liabilities is None)
                       else current_assets - current_liabilities)

    total_assets, total_equity = get("total_assets"), get("total_equity")
    # prefer the reported figure; the derived one is only as good as total equity
    total_liabilities = get("total_liabilities")
    if total_liabilities is None and total_assets is not None and total_equity is not None:
        total_liabilities = total_assets - total_equity

    # Capital employed: the reported figure when there is one, else the
    # Schedule III construction (total assets less current liabilities).
    capital_employed = get("capital_employed")
    if capital_employed is None and total_assets is not None and current_liabilities is not None:
        capital_employed = total_assets - current_liabilities

    # Borrowings: summing one side and calling it total understates leverage,
    # and short-term borrowings only map when the Schedule III sub-heading was
    # found, so a half-mapped balance sheet is routine. Sum either way, but say
    # which side was missing so the number is not mistaken for complete.
    total_borrowings = get("total_borrowings")
    long_term, short_term = get("borrowings_long_term"), get("borrowings_short_term")
    borrowings_partial = False
    if total_borrowings is None and (long_term is not None or short_term is not None):
        total_borrowings = (long_term or 0.0) + (short_term or 0.0)
        borrowings_partial = long_term is None or short_term is None

    # Schedule III reports retained earnings inside other equity; the face of
    # the balance sheet usually shows only the total, so other equity stands in.
    retained_earnings = get("retained_earnings")
    retained_earnings_proxied = False
    if retained_earnings is None:
        retained_earnings = get("other_equity")
        retained_earnings_proxied = retained_earnings is not None

    return {
        "ebit": ebit,
        "ebitda": ebitda,
        "working_capital": working_capital,
        "total_liabilities": total_liabilities,
        "capital_employed": capital_employed,
        "total_borrowings": total_borrowings,
        "borrowings_partial": borrowings_partial,
        "retained_earnings_used": retained_earnings,
        "retained_earnings_proxied": retained_earnings_proxied,
    }


def compute_ratios(fin: dict[str, float | None]) -> RatioResult:
    """Compute the 12 stream-C ratios for one company-year (figures in ₹ crore)."""
    result = RatioResult()
    d = derive(fin)
    result.derived = d
    get = fin.get

    current_assets, current_liabilities = get("current_assets"), get("current_liabilities")
    inventories, total_assets = get("inventories"), get("total_assets")
    total_equity = get("total_equity")
    ebit, ebitda = d["ebit"], d["ebitda"]
    debt = d["total_borrowings"]

    # --- liquidity
    value, why = _divide(current_assets, current_liabilities, positive_denominator=True)
    result.set("current_ratio", value, why)

    quick_assets = (None if (current_assets is None or inventories is None)
                    else current_assets - inventories)
    value, why = _divide(quick_assets, current_liabilities, positive_denominator=True)
    result.set("quick_ratio", value, why)

    value, why = _divide(get("cash_and_equivalents"), total_assets, positive_denominator=True)
    result.set("cash_to_assets", value, why)

    # --- profitability
    value, why = _divide(ebitda, get("revenue"), positive_denominator=True)
    result.set("ebitda_margin", value, why)

    value, why = _divide(ebit, d["capital_employed"], positive_denominator=True)
    result.set("roce", value, why)

    # ROA on net profit rather than EBIT, to stay comparable with Gupta (2022),
    # the Indian IBC ratio benchmark this project measures itself against.
    value, why = _divide(get("net_profit"), total_assets, positive_denominator=True)
    result.set("roa", value, why)

    # --- leverage and coverage
    value, why = _divide(debt, total_equity, positive_denominator=True)
    result.set("debt_to_equity", value, why)

    value, why = _divide(debt, ebitda, positive_denominator=True)
    result.set("debt_to_ebitda", value, why)

    # A negative numerator is meaningful here (EBIT does not cover interest), so
    # only the denominator is constrained.
    value, why = _divide(ebit, get("finance_costs"), positive_denominator=True)
    result.set("interest_coverage", value, why)

    value, why = _divide(d["retained_earnings_used"], total_assets, positive_denominator=True)
    result.set("retained_earnings_to_assets", value, why)

    # --- promoter pledge: exchange shareholding filings, a later step
    result.set("promoter_pledge", None, "pending_shareholding_filings")

    # --- Altman Z''
    z, why = altman_z_em(fin, d)
    result.set("altman_z_em", z, why)
    result.derived["altman_z_dprime"] = (None if z is None
                                         else z - ALTMAN_EM_COEFFICIENTS["intercept"])
    if z is not None:
        dprime = result.derived["altman_z_dprime"]
        result.altman_zone = ("safe" if dprime > ALTMAN_SAFE
                              else "distress" if dprime < ALTMAN_DISTRESS else "grey")

    finance_costs = get("finance_costs")
    result.indicators = {
        "borrowings_partial": bool(d["borrowings_partial"]),
        "negative_equity": None if total_equity is None else bool(total_equity < 0),
        "negative_ebitda": None if ebitda is None else bool(ebitda < 0),
        "zero_finance_costs": None if finance_costs is None else bool(finance_costs == 0),
        "negative_working_capital": (None if d["working_capital"] is None
                                     else bool(d["working_capital"] < 0)),
        "negative_capital_employed": (None if d["capital_employed"] is None
                                      else bool(d["capital_employed"] < 0)),
    }
    return result


def altman_z_em(fin: dict[str, float | None],
                d: dict[str, float | None] | None = None) -> tuple[float | None, str]:
    """Altman emerging-market Z''. Returns ``(score, reason_if_missing)``."""
    d = d if d is not None else derive(fin)
    total_assets = fin.get("total_assets")
    if total_assets is None or total_assets <= 0:
        return None, "total_assets_missing_or_nonpositive"

    terms: dict[str, float] = {}
    missing: list[str] = []
    for name, numerator, denominator in (
        ("x1", d["working_capital"], total_assets),
        ("x2", d["retained_earnings_used"], total_assets),
        ("x3", d["ebit"], total_assets),
        ("x4", fin.get("total_equity"), d["total_liabilities"]),
    ):
        if numerator is None or denominator is None:
            missing.append(f"{name}_missing")
        elif denominator == 0:
            missing.append(f"{name}_denominator_zero")
        else:
            terms[name] = numerator / denominator
    if missing:
        return None, ",".join(missing)

    c = ALTMAN_EM_COEFFICIENTS
    score = (c["x1"] * terms["x1"] + c["x2"] * terms["x2"] + c["x3"] * terms["x3"]
             + c["x4"] * terms["x4"] + c["intercept"])
    return score, ""


def winsorise(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip a ratio at the given percentiles.

    Ratios built from near-zero denominators produce extreme values that would
    dominate any scaler. The project plan calls for winsorising at the 1st and
    99th percentiles -- but the bounds must be learnt on the training fold only
    and then applied to the held-out fold, exactly like a scaler. Computing them
    over the whole table leaks the test fold's distribution into training, so
    Phase 3 stores raw ratios and this helper is called from the modelling code
    with fold-specific bounds. See ``docs/decisions_log.md``.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return numeric
    low, high = numeric.quantile(lower), numeric.quantile(upper)
    return numeric.clip(low, high)


def winsorise_bounds(series: pd.Series, lower: float = 0.01,
                     upper: float = 0.99) -> tuple[float, float]:
    """Learn clipping bounds on a training fold, to apply to a held-out fold."""
    numeric = pd.to_numeric(series, errors="coerce")
    return float(numeric.quantile(lower)), float(numeric.quantile(upper))

