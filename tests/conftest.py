"""Shared fixtures for the Total Portfolio Approach test suite.

Synthetic monthly returns (not from the paper) are generated here purely to
exercise the shared ``src/`` components on small, deterministic inputs. Market
data tests use a reduced slice loaded once through ``data/data_loader.py`` and
cached between runs via pytest's cache, and are skipped when
``DATALOADER_API_KEY`` is absent.
"""

import os

import numpy as np
import pandas as pd
import pytest

import src.config as cfg


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "market_data: tests that load a reduced slice of real market data"
    )


@pytest.fixture
def synthetic_returns():
    """216 months of synthetic returns for ACWI, a bond proxy and the seven
    ETFs, built from a known two-factor model so beta/spread recovery can be
    checked against a ground truth."""
    rng = np.random.default_rng(42)
    n = 216
    dates = pd.date_range("2008-04-30", periods=n, freq="ME")
    equity = rng.normal(0.006, 0.045, n)
    bond = rng.normal(0.003, 0.015, n)
    true_betas = {
        "DBC": (0.30, 0.05), "IWN": (1.10, -0.05), "VNQ": (0.95, 0.10),
        "HYG": (0.50, 0.14), "EMB": (0.55, 0.30), "LQD": (0.15, 0.70), "TLT": (-0.10, 1.00),
    }
    data = {"ACWI": equity, "IEF": bond}
    for ticker, (bE, bD) in true_betas.items():
        idio = rng.normal(0.0005, 0.03, n)
        data[ticker] = bE * equity + bD * bond + idio
    return pd.DataFrame(data, index=dates)


@pytest.fixture(scope="session")
def market_data_slice(request):
    """A short, cached slice of real monthly returns for a few tickers, used
    only by tests marked ``market_data``."""
    if not os.environ.get("DATALOADER_API_KEY"):
        pytest.skip("DATALOADER_API_KEY not set; skipping market-data test")
    cache = request.config.cache
    cached = cache.get("tpa/market_slice", None)
    if cached is not None:
        df = pd.DataFrame(cached["data"])
        df.index = pd.to_datetime(df.pop("index"))
        return df
    from data.data_loader import load_monthly_returns

    tickers = ["ACWI", "IEF", "HYG", "IWN"]
    returns, _ = load_monthly_returns(tickers, "2022-01-31", "2023-12-31")
    stored = returns.reset_index()
    cache.set("tpa/market_slice", {"data": stored.to_dict(orient="list")})
    return returns
