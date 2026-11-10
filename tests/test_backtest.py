"""Unit tests for src/backtest.py: monthly-rebalanced backtest statistics
versus the 60/40 reference portfolio (paper section 5, Table 4, p.5-6)."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    backtest_statistics,
    compounded_gap,
    information_ratio,
    model_band,
    portfolio_returns,
    running_gap,
    share_inside_band,
    worst_fall,
)


@pytest.fixture
def toy_returns():
    rng = np.random.default_rng(3)
    n = 216
    dates = pd.date_range("2008-04-30", periods=n, freq="ME")
    data = {t: rng.normal(0.005, 0.03, n) for t in ["ACWI", "IEF", "VNQ", "HYG", "IWN", "DBC", "TLT"]}
    return pd.DataFrame(data, index=dates)


def test_portfolio_returns_weighted_sum(toy_returns):
    weights = {"ACWI": 0.6, "IEF": 0.4}
    pr = portfolio_returns(toy_returns, weights)
    expected = 0.6 * toy_returns["ACWI"] + 0.4 * toy_returns["IEF"]
    pd.testing.assert_series_equal(pr, expected, check_names=False)


def test_backtest_statistics_keys_and_properties(toy_returns):
    weights = {"ACWI": 0.6, "IEF": 0.4}
    pr = portfolio_returns(toy_returns, weights)
    stats = backtest_statistics(pr)
    for key in ("growth", "average_return", "volatility", "worst_fall"):
        assert key in stats
    assert stats["volatility"] > 0
    assert stats["worst_fall"] <= 0
    # AM-GM at the monthly level: the geometric mean of (1+r) never exceeds
    # the arithmetic mean of (1+r); growth and average_return use different
    # annualisation conventions (compounding vs. x12), so compare monthly.
    monthly_geometric = float((1.0 + pr).prod()) ** (1.0 / len(pr)) - 1.0
    assert monthly_geometric <= float(pr.mean()) + 1e-9


def test_worst_fall_is_nonpositive_and_bounded(toy_returns):
    wf = worst_fall(toy_returns["ACWI"])
    assert -1.0 <= wf <= 0.0


def test_running_gap_last_value_equals_sum_of_differences(toy_returns):
    reference = portfolio_returns(toy_returns, {"ACWI": 0.6, "IEF": 0.4})
    best_mix = portfolio_returns(toy_returns, {"ACWI": 0.5, "IEF": 0.3, "VNQ": 0.2})
    gap = running_gap(best_mix, reference)
    diff = (best_mix - reference).dropna()
    assert gap.iloc[-1] == pytest.approx(diff.sum())
    assert len(gap) == len(diff)


def test_compounded_gap_differs_from_arithmetic_sum(toy_returns):
    reference = portfolio_returns(toy_returns, {"ACWI": 0.6, "IEF": 0.4})
    best_mix = portfolio_returns(toy_returns, {"ACWI": 0.5, "IEF": 0.3, "VNQ": 0.2})
    comp_gap = compounded_gap(best_mix, reference)
    arithmetic_gap = running_gap(best_mix, reference).iloc[-1]
    assert np.isfinite(comp_gap)
    assert comp_gap != pytest.approx(arithmetic_gap, abs=1e-12)


def test_model_band_widens_with_sqrt_time():
    band = model_band(n_months=216, annual_gap=0.0018, annual_band=0.02)
    row1 = band.loc[band["t_years"] == 1.0].iloc[0]
    row4 = band.loc[band["t_years"] == 4.0].iloc[0]
    width_1y = row1["upper_1s"] - row1["lower_1s"]
    width_4y = row4["upper_1s"] - row4["lower_1s"]
    assert width_4y / width_1y == pytest.approx(2.0, rel=0.02)


def test_share_inside_band_between_zero_and_one():
    band = model_band(n_months=216, annual_gap=0.0018, annual_band=0.02)
    rng = np.random.default_rng(5)
    gap = pd.Series(band["expectation"].to_numpy() + rng.normal(0, 0.01, 216))
    share = share_inside_band(gap, band, n_sigma=1)
    assert 0.0 <= share <= 1.0


def test_information_ratio_matches_paper_value():
    assert information_ratio(0.0018, 0.02) == pytest.approx(0.09, abs=0.005)


def test_backtest_statistics_single_month_no_crash():
    pr = pd.Series([0.01])
    stats = backtest_statistics(pr)
    assert np.isfinite(stats["growth"])
