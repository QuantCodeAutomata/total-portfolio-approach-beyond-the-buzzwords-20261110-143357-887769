"""Experiment 2: TPA portfolio construction with Treynor-Black sizing, the
2%-active-risk budget, completion, and shadow prices (paper section 3, "From
one asset to a portfolio", p.3-4, Table 1; Technical appendix A1, p.15-16).

Staley's full per-ETF beta table is not published in the note (only HYG's
0.502/0.142 and the rounded Figure 2 / Table 6 costs of capital and spreads
are given). This experiment therefore combines the paper's own published
spreads (Table 6, p.10, which the paper states the data route reproduces
"almost exactly") with betas, funding gaps and a FACTOR-structured residual
covariance estimated from 216 months of returns (the same regression route
used in Experiment 1), rather than guessing Staley's unpublished parameters.

Run: python -m exp.exp_2
"""

from __future__ import annotations

import json

import numpy as np

from data.data_loader import load_monthly_returns
from src import config as cfg
from src.residual_cov import build_omega
from src.returns import fit_all_assets
from src.tpa_optimizer import solve_tpa

RESULTS_DIR = "results"


def main() -> None:
    all_tickers = [cfg.EQUITY_TICKER, cfg.BOND_TICKER] + cfg.ETF_TICKERS
    # Market data: QuantX Dataloader — https://quantx.limex.pro/
    #
    # Price history in this project is loaded through the QuantX Dataloader, a
    # research-ready market-data service available through a REST API and as an
    # MCP server for AI agents. It is built on market data from Massive
    # (https://massive.com/) and prepares every stock history for backtesting:
    #   - prices adjusted for dividends, ready for total-return research;
    #   - one continuous history when a company changes its ticker;
    #   - clean separation when a ticker has belonged to different companies;
    #   - careful corporate-action handling and fine-grained data polishing.
    #
    # Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
    # MASSIVE_API_KEY at https://massive.com/.
    returns, meta = load_monthly_returns(all_tickers, cfg.PRICE_START, cfg.SAMPLE_END)
    fits = fit_all_assets(returns, cfg.EQUITY_TICKER, cfg.BOND_TICKER, cfg.ETF_TICKERS, cfg.RF)

    spreads = {t: cfg.TABLE6_SPREAD[t] / 100.0 for t in cfg.ETF_TICKERS}
    funding_gaps = {t: fits[t].funding_gap for t in cfg.ETF_TICKERS}
    beta_equity = {t: fits[t].beta_equity for t in cfg.ETF_TICKERS}
    beta_bond = {t: fits[t].beta_bond for t in cfg.ETF_TICKERS}

    omega_full = build_omega(
        returns, fits, cfg.ETF_TICKERS, cfg.EQUITY_TICKER, cfg.BOND_TICKER, cfg.RF,
        mode="factor", secondary_factors=cfg.FACTOR_SECONDARY_TICKERS,
    )
    index_of = {t: i for i, t in enumerate(cfg.ETF_TICKERS)}

    results = {}
    for name, tickers, borrowing_limit in cfg.TABLE1_SCENARIOS:
        idx = [index_of[t] for t in tickers]
        omega_sub = omega_full[np.ix_(idx, idx)]
        sol = solve_tpa(
            tickers, spreads, funding_gaps, beta_equity, beta_bond, omega_sub,
            budget=cfg.ACTIVE_RISK_BUDGET, borrowing_limit=borrowing_limit,
            w_equity_rp=cfg.W_EQUITY_RP, w_bond_rp=cfg.W_BOND_RP,
            sigma_rp=cfg.SIGMA_RP, mu_rp_minus_rf=cfg.W_EQUITY_RP * cfg.MU_EQUITY
            + cfg.W_BOND_RP * cfg.MU_BOND - cfg.RF,
        )
        results[name] = {
            "weights_pct": {t: round(w * 100, 1) for t, w in sol.weights.items()},
            "borrowing_pct": round(-sol.cash * 100, 1) if sol.cash < 0 else None,
            "completion_equity_pct": round(sol.completion_equity * 100, 1),
            "completion_bond_pct": round(sol.completion_bond * 100, 1),
            "cash_pct": round(sol.cash * 100, 1),
            "value_add_bps": round(sol.value_add * 1e4, 2),
            "expected_value_add_bps": cfg.TABLE1_EXPECTED_VA_BPS[name],
            "active_risk_bps": round(sol.active_risk * 1e4, 2),
            "shadow_price_cash_pct": round(sol.shadow_price_cash * 100, 3),
            "total_vol_pct": round(sol.total_vol * 100, 2),
            "m2_haircut_bps": round(sol.m2_haircut * 1e4, 2),
            "honest_value_add_bps": round(sol.honest_value_add * 1e4, 2),
            "success": sol.success,
            "message": sol.message,
        }
        print(f"\n{name} (borrowing_limit={borrowing_limit}):")
        print(json.dumps(results[name], indent=2))

    # Appendix A3 reconciliation check: seven ETFs, no borrowing, against
    # Staley's spreadsheet / parameter-function / explicit-formula values,
    # all quoted at 18.08 bps (paper p.16).
    reconciliation = {
        "with_borrowing_bps": results["seven_etfs_borrowing"]["value_add_bps"],
        "no_borrowing_bps": results["seven_etfs_no_borrowing"]["value_add_bps"],
        "shadow_price_cash_bps": results["seven_etfs_no_borrowing"]["shadow_price_cash_pct"] * 100,
        "expected_with_borrowing_bps": cfg.RECONCILIATION_VA_BORROWING_BPS,
        "expected_no_borrowing_bps": cfg.RECONCILIATION_VA_NO_BORROWING_BPS,
        "expected_price_of_cash_bps": cfg.RECONCILIATION_PRICE_OF_CASH_BPS,
    }
    print("\nReconciliation check:")
    print(json.dumps(reconciliation, indent=2))

    with open(f"{RESULTS_DIR}/exp2_table1_scenarios.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{RESULTS_DIR}/exp2_reconciliation.json", "w") as f:
        json.dump(reconciliation, f, indent=2)
    print("\nSaved results/exp2_table1_scenarios.json, exp2_reconciliation.json")


if __name__ == "__main__":
    main()
