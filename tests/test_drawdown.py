"""Unit tests for src/drawdown.py: conditional drawdown and its per-holding
decomposition (paper section 9, p.12-14)."""

import numpy as np
import pandas as pd
import pytest

from src.drawdown import (
    block_bootstrap_breach_share,
    conditional_drawdown,
    per_holding_decomposition,
    underwater_depth,
    wealth_path,
)


@pytest.fixture
def toy_returns():
    rng = np.random.default_rng(1)
    n = 120
    dates = pd.date_range("2010-01-31", periods=n, freq="ME")
    a = pd.Series(rng.normal(0.004, 0.03, n), index=dates, name="A")
    b = pd.Series(rng.normal(0.003, 0.02, n), index=dates, name="B")
    return pd.concat([a, b], axis=1)


def test_underwater_depth_nonnegative_and_zero_at_peak():
    returns = pd.Series([0.05, -0.1, 0.02, -0.05, 0.1])
    wealth = wealth_path(returns, compounded=True)
    depth = underwater_depth(wealth)
    assert (depth >= -1e-9).all()
    assert depth.iloc[0] == pytest.approx(0.0)


def test_conditional_drawdown_is_worse_than_average_depth():
    """The conditional drawdown (worst 5% average) must be at least as large
    as the mean underwater depth, since it averages only the tail."""
    returns = pd.Series(np.random.default_rng(0).normal(0.0, 0.03, 216))
    cdar = conditional_drawdown(returns, tail_fraction=0.05, compounded=False)
    wealth = wealth_path(returns, compounded=False)
    mean_depth = float(underwater_depth(wealth).mean())
    assert cdar >= mean_depth


def test_conditional_drawdown_single_point_is_zero():
    returns = pd.Series([0.01])
    assert conditional_drawdown(returns, tail_fraction=0.05) == pytest.approx(0.0)


def test_per_holding_decomposition_sums_to_total(toy_returns):
    weights = {"A": 0.6, "B": 0.4}
    decomp = per_holding_decomposition(toy_returns, weights, tail_fraction=0.05)
    total_from_shares = decomp["drawdown_contribution"].sum()
    portfolio_returns = (toy_returns * pd.Series(weights)).sum(axis=1)
    total_direct = conditional_drawdown(portfolio_returns, tail_fraction=0.05, compounded=False)
    assert total_from_shares == pytest.approx(total_direct, abs=1e-9)


def test_block_bootstrap_breach_share_between_zero_and_one(toy_returns):
    rng = np.random.default_rng(7)
    share = block_bootstrap_breach_share(
        toy_returns, {"A": 0.6, "B": 0.4}, block_months=12, n_paths=20,
        threshold=0.1, share_months=0.05, rng=rng,
    )
    assert 0.0 <= share <= 1.0
