"""Two-factor regression, cost of capital, spread, funding gap and appraisal
ratio (paper Technical appendix A1, p.15; section 2, p.2-3).

Regression: r_i - r_f = alpha_i + beta_iE (r_E - r_f) + beta_iD (r_D - r_f) + eps_i
Cost of capital: CoC_i = r_f + beta_iE (mu_E - r_f) + beta_iD (mu_D - r_f)
Spread: s_i = mu_i - CoC_i
Funding gap: g_i = 1 - beta_iE - beta_iD
Appraisal ratio: AR_i = s_i / sigma_eps_i  (own, i.e. residual, risk)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

MONTHS_PER_YEAR = 12


@dataclass
class AssetFit:
    """Two-factor fit for one asset (paper Appendix A1)."""

    ticker: str
    beta_equity: float
    beta_bond: float
    alpha_annual: float  # annualised intercept (historical alpha), monthly alpha * 12
    residual_risk_annual: float  # sigma_eps, annualised: monthly std(eps) * sqrt(12)
    funding_gap: float  # g_i = 1 - beta_E - beta_D
    r_squared: float
    n_obs: int


def fit_two_factor(
    asset_returns: pd.Series,
    equity_returns: pd.Series,
    bond_returns: pd.Series,
    rf_annual: float,
) -> AssetFit:
    """OLS fit of one asset's monthly excess returns on excess equity and bond
    returns (paper Appendix A1 regression equation)."""
    rf_monthly = rf_annual / MONTHS_PER_YEAR
    df = pd.concat(
        {"asset": asset_returns, "equity": equity_returns, "bond": bond_returns}, axis=1
    ).dropna()
    assert len(df) > 24, f"{asset_returns.name}: too few overlapping months ({len(df)})"
    y = df["asset"] - rf_monthly
    x_equity = df["equity"] - rf_monthly
    x_bond = df["bond"] - rf_monthly
    X = sm.add_constant(pd.concat({"equity": x_equity, "bond": x_bond}, axis=1))
    model = sm.OLS(y, X).fit()
    beta_equity = float(model.params["equity"])
    beta_bond = float(model.params["bond"])
    residual_risk_annual = float(model.resid.std(ddof=X.shape[1]) * np.sqrt(MONTHS_PER_YEAR))
    alpha_annual = float(model.params["const"] * MONTHS_PER_YEAR)
    return AssetFit(
        ticker=str(asset_returns.name),
        beta_equity=beta_equity,
        beta_bond=beta_bond,
        alpha_annual=alpha_annual,
        residual_risk_annual=residual_risk_annual,
        funding_gap=1.0 - beta_equity - beta_bond,
        r_squared=float(model.rsquared),
        n_obs=int(len(df)),
    )


def cost_of_capital(beta_equity: float, beta_bond: float, mu_equity: float, mu_bond: float, rf: float) -> float:
    """CoC_i = r_f + beta_iE (mu_E - r_f) + beta_iD (mu_D - r_f) (paper Appendix A1)."""
    return rf + beta_equity * (mu_equity - rf) + beta_bond * (mu_bond - rf)


def spread(expected_return: float, cost_of_capital_value: float) -> float:
    """s_i = mu_i - CoC_i (paper Appendix A1)."""
    return expected_return - cost_of_capital_value


def funding_gap(beta_equity: float, beta_bond: float) -> float:
    """g_i = 1 - beta_iE - beta_iD (paper Appendix A1)."""
    return 1.0 - beta_equity - beta_bond


def appraisal_ratio(spread_value: float, own_risk: float) -> float:
    """AQR-style appraisal ratio: spread divided by own (residual) risk (paper p.3)."""
    assert own_risk > 0, "own risk must be positive"
    return spread_value / own_risk


def funding_decomposition(amount: float, beta_equity: float, beta_bond: float, gap: float) -> dict:
    """Sell beta_E * amount of global shares, beta_D * amount of 10-year bonds,
    use gap * amount of cash (paper p.3, HYG $100 example)."""
    return {
        "sell_global_equity": beta_equity * amount,
        "sell_bonds": beta_bond * amount,
        "use_cash": gap * amount,
    }


def fit_all_assets(
    returns: pd.DataFrame, equity_ticker: str, bond_ticker: str, etf_tickers: list, rf_annual: float
) -> dict:
    """Fit every ETF in ``etf_tickers`` against ``equity_ticker`` and
    ``bond_ticker`` columns of ``returns``. Returns a dict ticker -> AssetFit."""
    fits = {}
    for ticker in etf_tickers:
        fits[ticker] = fit_two_factor(
            returns[ticker], returns[equity_ticker], returns[bond_ticker], rf_annual
        )
    return fits


def build_hurdle_table(
    fits: dict, expected_returns: dict, mu_equity: float, mu_bond: float, rf: float
) -> pd.DataFrame:
    """Cost-of-capital / spread / funding-gap / appraisal-ratio table for every
    asset in ``fits`` (reproduces Staley's Table 2, paper Figure 2)."""
    rows = []
    for ticker, fit in fits.items():
        coc = cost_of_capital(fit.beta_equity, fit.beta_bond, mu_equity, mu_bond, rf)
        s = spread(expected_returns[ticker], coc)
        ar = appraisal_ratio(s, fit.residual_risk_annual)
        rows.append(
            {
                "ticker": ticker,
                "beta_equity": fit.beta_equity,
                "beta_bond": fit.beta_bond,
                "cost_of_capital": coc,
                "expected_return": expected_returns[ticker],
                "spread": s,
                "funding_gap": fit.funding_gap,
                "own_risk": fit.residual_risk_annual,
                "appraisal_ratio": ar,
                "r_squared": fit.r_squared,
                "n_obs": fit.n_obs,
            }
        )
    return pd.DataFrame(rows).set_index("ticker")
