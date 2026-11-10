"""Unit tests for src/tpa_optimizer.py (paper section 3, Technical appendix
A1, p.3-4, p.15): Treynor-Black sizing, active-risk budget, completion,
shadow price of cash, and the M2 haircut."""

import numpy as np
import pytest

from src.tpa_optimizer import m2_haircut, solve_tpa, treynor_black_unconstrained


def _two_asset_problem():
    # Asset B has a negative funding gap (its beta sum exceeds one, as a long
    # Treasury-like holding does in the paper, p.4: "each dollar in TLT ...
    # frees cash"), so a no-borrowing solution with both weights positive is
    # feasible; with two positive gaps the no-borrowing constraint would force
    # the trivial all-zero solution regardless of the solver.
    tickers = ["A", "B"]
    spreads = {"A": 0.01, "B": 0.005}
    gaps = {"A": 0.2, "B": -0.3}
    beta_e = {"A": 0.6, "B": 0.3}
    beta_d = {"A": 0.2, "B": 1.0}
    omega = np.array([[0.02**2, 0.0], [0.0, 0.015**2]])
    return tickers, spreads, gaps, beta_e, beta_d, omega


def test_treynor_black_unconstrained_hits_budget_exactly():
    s = np.array([0.01, 0.005])
    omega = np.diag([0.02**2, 0.015**2])
    budget = 0.02
    w = treynor_black_unconstrained(s, np.linalg.inv(omega), budget)
    realised = np.sqrt(w @ omega @ w)
    assert realised == pytest.approx(budget, rel=1e-6)
    # Treynor-Black: weight proportional to spread / own variance.
    ratio_a = w[0] / (s[0] / omega[0, 0])
    ratio_b = w[1] / (s[1] / omega[1, 1])
    assert ratio_a == pytest.approx(ratio_b, rel=1e-6)


def test_solve_tpa_long_only_hits_active_risk_budget():
    tickers, spreads, gaps, beta_e, beta_d, omega = _two_asset_problem()
    sol = solve_tpa(tickers, spreads, gaps, beta_e, beta_d, omega, budget=0.02, borrowing_limit=0.0)
    assert sol.success
    assert all(w >= -1e-8 for w in sol.weights.values())
    assert sol.active_risk == pytest.approx(0.02, abs=1e-3)
    assert sol.value_add > 0  # both spreads positive, so value-add must be positive


def test_solve_tpa_borrowing_constraint_binds_when_tight():
    tickers, spreads, gaps, beta_e, beta_d, omega = _two_asset_problem()
    unconstrained = solve_tpa(tickers, spreads, gaps, beta_e, beta_d, omega, budget=0.02, borrowing_limit=None)
    cash_used_unconstrained = sum(gaps[t] * unconstrained.weights[t] for t in tickers)
    tight_limit = cash_used_unconstrained / 2
    constrained = solve_tpa(tickers, spreads, gaps, beta_e, beta_d, omega, budget=0.02, borrowing_limit=tight_limit)
    cash_used_constrained = sum(gaps[t] * constrained.weights[t] for t in tickers)
    assert cash_used_constrained <= tight_limit + 1e-4
    # A binding constraint must have a nonzero shadow price.
    assert constrained.shadow_price_cash > 0


def test_completion_restores_reference_exposure():
    tickers, spreads, gaps, beta_e, beta_d, omega = _two_asset_problem()
    sol = solve_tpa(tickers, spreads, gaps, beta_e, beta_d, omega, budget=0.02, borrowing_limit=0.0)
    total_equity = sol.completion_equity + sum(beta_e[t] * sol.weights[t] for t in tickers)
    total_bond = sol.completion_bond + sum(beta_d[t] * sol.weights[t] for t in tickers)
    assert total_equity == pytest.approx(0.60, abs=1e-6)
    assert total_bond == pytest.approx(0.40, abs=1e-6)


def test_m2_haircut_zero_when_vol_unchanged():
    assert m2_haircut(0.10, 0.10, 0.04) == pytest.approx(0.0)


def test_m2_haircut_positive_when_vol_rises():
    assert m2_haircut(0.105, 0.10, 0.04) > 0


def test_hyg_three_etf_table1_scenario_matches_paper_within_tolerance():
    """Reproduces the sign and rough magnitude of Table 1's three-ETF,
    no-borrowing case (IWN, HYG, TLT) using Staley's parameters and a
    diagonal residual covariance (own risks from Table 6, p.10)."""
    import src.config as cfg
    from src.returns import cost_of_capital, funding_gap, spread

    tickers = ["IWN", "HYG", "TLT"]
    betas = {"IWN": (0.75, -0.05), "HYG": (0.502, 0.142), "TLT": (-0.20, 1.80)}
    spreads, gaps, beta_e, beta_d = {}, {}, {}, {}
    for t in tickers:
        bE, bD = betas[t]
        coc = cost_of_capital(bE, bD, cfg.MU_EQUITY, cfg.MU_BOND, cfg.RF)
        spreads[t] = spread(cfg.ETF_EXPECTED_RETURN[t], coc)
        gaps[t] = funding_gap(bE, bD)
        beta_e[t], beta_d[t] = bE, bD
    own_risk = {t: cfg.TABLE6_OWN_RISK[t] / 100.0 for t in tickers}
    omega = np.diag([own_risk[t] ** 2 for t in tickers])
    sol = solve_tpa(tickers, spreads, gaps, beta_e, beta_d, omega, budget=cfg.ACTIVE_RISK_BUDGET, borrowing_limit=0.0)
    assert sol.success
    assert sol.active_risk == pytest.approx(cfg.ACTIVE_RISK_BUDGET, abs=1e-3)
    # Value-add must be positive and of the right order of magnitude (paper: 13.1 bps).
    assert 0.0 < sol.value_add < 0.01
