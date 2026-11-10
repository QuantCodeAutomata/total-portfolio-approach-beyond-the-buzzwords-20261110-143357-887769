"""Unit tests for src/returns.py: two-factor regression, cost of capital,
spread, funding gap and appraisal ratio (paper Technical appendix A1)."""

import numpy as np
import pytest

import src.config as cfg
from src.returns import (
    appraisal_ratio,
    build_hurdle_table,
    cost_of_capital,
    fit_all_assets,
    fit_two_factor,
    funding_decomposition,
    funding_gap,
    spread,
)


def test_fit_two_factor_recovers_known_betas(synthetic_returns):
    """Recovers the betas used to generate the synthetic series within a
    reasonable tolerance (no look-ahead: uses only the full historical
    window, as the paper's regression does)."""
    fit = fit_two_factor(
        synthetic_returns["HYG"], synthetic_returns["ACWI"], synthetic_returns["IEF"], cfg.RF
    )
    assert fit.beta_equity == pytest.approx(0.50, abs=0.15)
    assert fit.beta_bond == pytest.approx(0.14, abs=0.15)
    assert fit.residual_risk_annual > 0
    assert fit.n_obs == 216
    assert fit.funding_gap == pytest.approx(1.0 - fit.beta_equity - fit.beta_bond)


def test_hyg_worked_example_matches_paper():
    """Paper worked example (p.2-3): CoC 5.95%, spread +0.51%, funding gap
    0.356, appraisal ratio ~0.08 for HYG using Staley's rounded betas."""
    coc = cost_of_capital(cfg.HYG_BETA_EQUITY, cfg.HYG_BETA_BOND, cfg.MU_EQUITY, cfg.MU_BOND, cfg.RF)
    assert coc == pytest.approx(cfg.HYG_COC_EXPECTED, abs=1e-4)
    s = spread(cfg.ETF_EXPECTED_RETURN["HYG"], coc)
    assert s == pytest.approx(cfg.HYG_SPREAD_EXPECTED, abs=1e-4)
    gap = funding_gap(cfg.HYG_BETA_EQUITY, cfg.HYG_BETA_BOND)
    assert gap == pytest.approx(cfg.HYG_FUNDING_GAP_EXPECTED, abs=1e-3)
    ar = appraisal_ratio(s, cfg.HYG_OWN_RISK_WORKED)
    assert ar == pytest.approx(cfg.HYG_APPRAISAL_RATIO_EXPECTED, abs=0.01)


def test_funding_decomposition_hyg_100_dollars():
    """Paper p.3: a $100 HYG purchase sells $50.2 of global shares, $14.2 of
    10-year bonds, and uses $35.6 of cash (within the paper's rounding)."""
    decomp = funding_decomposition(100.0, cfg.HYG_BETA_EQUITY, cfg.HYG_BETA_BOND, cfg.HYG_FUNDING_GAP_EXPECTED)
    assert decomp["sell_global_equity"] == pytest.approx(50.2, abs=0.1)
    assert decomp["sell_bonds"] == pytest.approx(14.2, abs=0.1)
    assert decomp["use_cash"] == pytest.approx(35.6, abs=0.1)


def test_spread_sign_flips_with_expected_return():
    """s_i = mu_i - CoC_i must change sign when mu_i crosses CoC_i."""
    coc = 0.05
    assert spread(0.06, coc) > 0
    assert spread(0.04, coc) < 0


def test_build_hurdle_table_columns_and_order(synthetic_returns):
    fits = fit_all_assets(synthetic_returns, cfg.EQUITY_TICKER, cfg.BOND_TICKER, cfg.ETF_TICKERS, cfg.RF)
    table = build_hurdle_table(fits, cfg.ETF_EXPECTED_RETURN, cfg.MU_EQUITY, cfg.MU_BOND, cfg.RF)
    assert set(cfg.ETF_TICKERS) == set(table.index)
    assert "spread" in table.columns and "cost_of_capital" in table.columns
    assert (table["own_risk"] > 0).all()


def test_appraisal_ratio_requires_positive_risk():
    with pytest.raises(AssertionError):
        appraisal_ratio(0.01, 0.0)


def test_four_of_seven_clear_hurdle_in_figure2():
    """Paper expected outcome: four of seven ETFs clear the hurdle (positive
    spread), three do not (Figure 2, DBC/IWN/VNQ/HYG positive, EMB/LQD/TLT
    negative)."""
    positive = [t for t, s in cfg.FIGURE2_SPREAD.items() if s > 0]
    negative = [t for t, s in cfg.FIGURE2_SPREAD.items() if s < 0]
    assert len(positive) == 4
    assert len(negative) == 3
    assert set(positive) == {"DBC", "IWN", "VNQ", "HYG"}
