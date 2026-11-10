"""Residual covariance matrix Omega under three correlation modes: ACTUAL,
ZERO and FACTOR (paper p.4, p.10, Table 6; p.16 Appendix A4, FA_TPA_Cov).

The FACTOR mode is a two-secondary-factor residual model: residuals of the
small-cap value (IWN) and high-yield (HYG) ETFs are treated as the secondary
factors, and every other asset's residual is regressed on them. The paper
states only that "US small-cap value and US high yield" are the secondary
factors (p.4, p.10); the exact numerical construction (Staley's Table 6) is
not reproduced in the note, so this low-rank-plus-diagonal construction is a
defensible implementation of the description, not a quoted formula.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

MONTHS_PER_YEAR = 12


def residual_frame(
    returns: pd.DataFrame, fits: dict, equity_ticker: str, bond_ticker: str, rf_annual: float
) -> pd.DataFrame:
    """Monthly regression residuals eps_i,t for every asset in ``fits``."""
    rf_monthly = rf_annual / MONTHS_PER_YEAR
    eq = returns[equity_ticker] - rf_monthly
    bd = returns[bond_ticker] - rf_monthly
    out = {}
    for ticker, fit in fits.items():
        y = returns[ticker] - rf_monthly
        fitted = fit.alpha_annual / MONTHS_PER_YEAR + fit.beta_equity * eq + fit.beta_bond * bd
        out[ticker] = (y - fitted).dropna()
    return pd.concat(out, axis=1).dropna(how="all")


def covariance_actual(residuals: pd.DataFrame, tickers: list) -> np.ndarray:
    """Sample residual covariance, annualised (paper: 'actual' correlation mode)."""
    cov_monthly = residuals[tickers].cov().values
    return cov_monthly * MONTHS_PER_YEAR


def covariance_zero(residuals: pd.DataFrame, tickers: list) -> np.ndarray:
    """Diagonal residual covariance: zero correlation between residuals
    (paper: 'zero' correlation mode)."""
    var_monthly = residuals[tickers].var().values
    return np.diag(var_monthly * MONTHS_PER_YEAR)


def covariance_factor(residuals: pd.DataFrame, tickers: list, secondary_factors: list) -> np.ndarray:
    """Low-rank-plus-diagonal residual covariance: every asset's residual is
    regressed on the secondary-factor residuals (IWN, HYG), splitting its
    variance into a systematic (factor) part and an idiosyncratic part
    (custom implementation of the FACTOR description, paper p.4, p.10)."""
    factor_resid = residuals[secondary_factors].dropna()
    n_assets, n_factors = len(tickers), len(secondary_factors)
    loadings = np.zeros((n_assets, n_factors))
    idio_var_monthly = np.zeros(n_assets)
    factor_index = {f: j for j, f in enumerate(secondary_factors)}
    for i, ticker in enumerate(tickers):
        if ticker in factor_index:
            loadings[i, factor_index[ticker]] = 1.0
            idio_var_monthly[i] = 0.0
            continue
        y = residuals[ticker].reindex(factor_resid.index).dropna()
        X = sm.add_constant(factor_resid.loc[y.index])
        model = sm.OLS(y, X).fit()
        for factor in secondary_factors:
            loadings[i, factor_index[factor]] = float(model.params[factor])
        idio_var_monthly[i] = float(model.resid.var(ddof=X.shape[1]))
    factor_cov_monthly = factor_resid.cov().values
    systematic_monthly = loadings @ factor_cov_monthly @ loadings.T
    cov_monthly = systematic_monthly + np.diag(idio_var_monthly)
    return cov_monthly * MONTHS_PER_YEAR


def build_omega(
    returns: pd.DataFrame,
    fits: dict,
    tickers: list,
    equity_ticker: str,
    bond_ticker: str,
    rf_annual: float,
    mode: str,
    secondary_factors: list | None = None,
) -> np.ndarray:
    """Dispatch to the requested residual-covariance mode: 'actual', 'zero' or
    'factor' (paper p.4, Dashboard settings table)."""
    residuals = residual_frame(returns, fits, equity_ticker, bond_ticker, rf_annual)
    if mode == "actual":
        return covariance_actual(residuals, tickers)
    if mode == "zero":
        return covariance_zero(residuals, tickers)
    if mode == "factor":
        assert secondary_factors, "FACTOR mode needs secondary_factors"
        return covariance_factor(residuals, tickers, secondary_factors)
    raise ValueError(f"unknown residual-covariance mode: {mode}")
