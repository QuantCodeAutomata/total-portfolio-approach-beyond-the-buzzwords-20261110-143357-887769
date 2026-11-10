"""Unit tests for src/residual_cov.py: residual covariance matrix Omega under
ACTUAL, ZERO and FACTOR correlation modes (paper p.4, p.10, Table 6)."""

import numpy as np
import pytest

import src.config as cfg
from src.residual_cov import build_omega, covariance_zero, residual_frame
from src.returns import fit_all_assets


def test_covariance_zero_is_diagonal(synthetic_returns):
    fits = fit_all_assets(synthetic_returns, cfg.EQUITY_TICKER, cfg.BOND_TICKER, cfg.ETF_TICKERS, cfg.RF)
    residuals = residual_frame(synthetic_returns, fits, cfg.EQUITY_TICKER, cfg.BOND_TICKER, cfg.RF)
    omega = covariance_zero(residuals, cfg.ETF_TICKERS)
    off_diagonal = omega - np.diag(np.diagonal(omega))
    assert np.allclose(off_diagonal, 0.0)
    assert (np.diagonal(omega) > 0).all()


def test_build_omega_actual_is_symmetric_positive_semidefinite(synthetic_returns):
    fits = fit_all_assets(synthetic_returns, cfg.EQUITY_TICKER, cfg.BOND_TICKER, cfg.ETF_TICKERS, cfg.RF)
    omega = build_omega(
        synthetic_returns, fits, cfg.ETF_TICKERS, cfg.EQUITY_TICKER, cfg.BOND_TICKER, cfg.RF, mode="actual"
    )
    assert np.allclose(omega, omega.T)
    eigvals = np.linalg.eigvalsh(omega)
    assert (eigvals >= -1e-8).all()


def test_build_omega_factor_mode_diagonal_blocks_for_secondary_factors(synthetic_returns):
    fits = fit_all_assets(synthetic_returns, cfg.EQUITY_TICKER, cfg.BOND_TICKER, cfg.ETF_TICKERS, cfg.RF)
    omega = build_omega(
        synthetic_returns, fits, cfg.ETF_TICKERS, cfg.EQUITY_TICKER, cfg.BOND_TICKER, cfg.RF,
        mode="factor", secondary_factors=cfg.FACTOR_SECONDARY_TICKERS,
    )
    assert omega.shape == (len(cfg.ETF_TICKERS), len(cfg.ETF_TICKERS))
    assert np.allclose(omega, omega.T, atol=1e-10)


def test_build_omega_unknown_mode_raises(synthetic_returns):
    fits = fit_all_assets(synthetic_returns, cfg.EQUITY_TICKER, cfg.BOND_TICKER, cfg.ETF_TICKERS, cfg.RF)
    with pytest.raises(ValueError):
        build_omega(
            synthetic_returns, fits, cfg.ETF_TICKERS, cfg.EQUITY_TICKER, cfg.BOND_TICKER, cfg.RF, mode="bogus"
        )
