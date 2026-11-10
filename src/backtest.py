"""Monthly-rebalanced backtest statistics for a fixed-weight portfolio against
the 60/40 reference (paper section 5, "A look back, not a backtest", p.5-6,
Table 4).

This is a look at how the portfolio's risk behaved historically, not a test
of forecast skill: the weights are derived from 2026 capital-market
assumptions and applied to 2008-2026 returns, which is look-ahead bias by
construction (paper p.5).
"""

MONTHS_PER_YEAR = 12

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox


def portfolio_returns(returns: pd.DataFrame, weights: dict) -> pd.Series:
    """Monthly-rebalanced portfolio return: weights are held fixed and
    re-applied every month (paper p.5: 'monthly rebalanced')."""
    cols = list(weights.keys())
    return (returns[cols] * pd.Series(weights)).sum(axis=1)


def annualised_growth(returns: pd.Series) -> float:
    """Geometric ('growth a year') return (paper Table 4)."""
    wealth_end = float((1.0 + returns).prod())
    n_years = len(returns) / MONTHS_PER_YEAR
    return wealth_end ** (1.0 / n_years) - 1.0


def average_return(returns: pd.Series) -> float:
    """Arithmetic average monthly return, annualised by x12 (paper Table 4)."""
    return float(returns.mean()) * MONTHS_PER_YEAR


def annualised_volatility(returns: pd.Series) -> float:
    """Monthly standard deviation, annualised by sqrt(12) (paper Table 4)."""
    return float(returns.std(ddof=1)) * np.sqrt(MONTHS_PER_YEAR)


def worst_fall(returns: pd.Series) -> float:
    """Maximum peak-to-trough compounded drawdown over the sample (paper
    Table 4, 'Worst fall')."""
    wealth = (1.0 + returns).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    return float(drawdown.min())


def tracking_error(portfolio: pd.Series, reference: pd.Series) -> float:
    """std(monthly differences) * sqrt(12) (paper Table 4)."""
    diff = (portfolio - reference).dropna()
    return float(diff.std(ddof=1)) * np.sqrt(MONTHS_PER_YEAR)


def backtest_statistics(returns: pd.Series) -> dict:
    """Growth, average return, volatility and worst fall for one return
    series (paper Table 4)."""
    return {
        "growth": annualised_growth(returns),
        "average_return": average_return(returns),
        "volatility": annualised_volatility(returns),
        "worst_fall": worst_fall(returns),
    }


def running_gap(portfolio: pd.Series, reference: pd.Series) -> pd.Series:
    """Running arithmetic sum of monthly return differences (paper p.6,
    Figure 4: 'the running sum of monthly return differences')."""
    diff = (portfolio - reference).dropna()
    return diff.cumsum()


def compounded_gap(portfolio: pd.Series, reference: pd.Series) -> float:
    """End-of-sample compounded gap: wealth ratio of the two compounded paths
    minus one (paper p.6: 'compounded, the end gap is -3.6% rather than
    -3.3%')."""
    diff = (portfolio - reference).dropna()
    port_wealth = float((1.0 + portfolio.reindex(diff.index)).prod())
    ref_wealth = float((1.0 + reference.reindex(diff.index)).prod())
    return port_wealth / ref_wealth - 1.0


def model_band(n_months: int, annual_gap: float, annual_band: float) -> pd.DataFrame:
    """Modelled expected cumulative gap path and its +/- sqrt(t) bands (paper
    p.6, Figure 4 caption: bands of 2% * sqrt(t))."""
    t_years = np.arange(1, n_months + 1) / MONTHS_PER_YEAR
    expectation = annual_gap * t_years
    one_sigma = annual_band * np.sqrt(t_years)
    return pd.DataFrame(
        {
            "t_years": t_years,
            "expectation": expectation,
            "upper_1s": expectation + one_sigma,
            "lower_1s": expectation - one_sigma,
            "upper_2s": expectation + 2 * one_sigma,
            "lower_2s": expectation - 2 * one_sigma,
        }
    )


def share_inside_band(gap: pd.Series, band: pd.DataFrame, n_sigma: int) -> float:
    """Fraction of months with the running gap inside the +/- n_sigma band
    (paper p.6: 'inside the two-in-three range in 83% of months')."""
    upper = band[f"upper_{n_sigma}s"].to_numpy()
    lower = band[f"lower_{n_sigma}s"].to_numpy()
    gap_vals = gap.to_numpy()
    inside = (gap_vals >= lower) & (gap_vals <= upper)
    return float(inside.mean())


def ljung_box_autocorrelation(diff: pd.Series, lags: int = 12) -> pd.DataFrame:
    """Ljung-Box test for autocorrelation of the monthly return differences
    (paper p.6: 'no meaningful autocorrelation', supporting the sqrt(t)
    scaling of the bands)."""
    return acorr_ljungbox(diff.dropna(), lags=[lags], return_df=True)


def information_ratio(annual_gap: float, annual_band: float) -> float:
    """Implied information ratio = modelled gap / active-risk budget (paper
    p.6: 0.18% / 2% = 0.09)."""
    return annual_gap / annual_band
