"""Experiment 1: cost of capital, spread, and funding gap for the seven
candidate ETFs (paper section 2, "One idea: every asset must beat its
lookalike", p.2-3; Figure 2; Technical appendix A1, p.15).

Two routes are run and compared:

* the parameter route, using Staley's rounded HYG betas (0.502 / 0.142)
  directly in the cost-of-capital formula, reproducing the worked example;
* the regression route, estimating every asset's two betas from 216 months
  of ACWI / 10-year Treasury / ETF returns (30-Apr-2008 to 31-Mar-2026) and
  building the full seven-asset hurdle table (Figure 2).

Run: python -m exp.exp_1
"""

from __future__ import annotations

import json

import pandas as pd

from data.data_loader import load_monthly_returns
from src import config as cfg
from src.returns import (
    appraisal_ratio,
    build_hurdle_table,
    cost_of_capital,
    fit_all_assets,
    funding_decomposition,
    funding_gap,
    spread,
)

RESULTS_DIR = "results"


def run_worked_hyg_example() -> dict:
    """Staley's HYG worked example using his rounded parameter betas (paper
    p.2-3): CoC 5.95%, spread +0.51%, funding gap 0.356, appraisal ratio 0.08."""
    coc = cost_of_capital(cfg.HYG_BETA_EQUITY, cfg.HYG_BETA_BOND, cfg.MU_EQUITY, cfg.MU_BOND, cfg.RF)
    s = spread(cfg.ETF_EXPECTED_RETURN["HYG"], coc)
    g = funding_gap(cfg.HYG_BETA_EQUITY, cfg.HYG_BETA_BOND)
    ar = appraisal_ratio(s, cfg.HYG_OWN_RISK_WORKED)
    decomposition = funding_decomposition(100.0, cfg.HYG_BETA_EQUITY, cfg.HYG_BETA_BOND, g)
    return {
        "beta_equity": cfg.HYG_BETA_EQUITY,
        "beta_bond": cfg.HYG_BETA_BOND,
        "cost_of_capital": coc,
        "expected_return": cfg.ETF_EXPECTED_RETURN["HYG"],
        "spread": s,
        "funding_gap": g,
        "own_risk": cfg.HYG_OWN_RISK_WORKED,
        "appraisal_ratio": ar,
        "funding_decomposition_per_100": decomposition,
        "expected": {
            "cost_of_capital": cfg.HYG_COC_EXPECTED,
            "spread": cfg.HYG_SPREAD_EXPECTED,
            "funding_gap": cfg.HYG_FUNDING_GAP_EXPECTED,
            "appraisal_ratio": cfg.HYG_APPRAISAL_RATIO_EXPECTED,
        },
    }


def main() -> None:
    worked = run_worked_hyg_example()
    print("HYG worked example (Staley's parameter betas):")
    print(json.dumps(worked, indent=2))

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
    all_tickers = [cfg.EQUITY_TICKER, cfg.BOND_TICKER] + cfg.ETF_TICKERS
    returns, meta = load_monthly_returns(all_tickers, cfg.PRICE_START, cfg.SAMPLE_END)
    print(f"Loaded {len(returns)} monthly rows, {returns.index.min()} to {returns.index.max()}")
    print(f"Universe: {list(returns.columns)}")
    if meta["failed"]:
        raise RuntimeError(f"Dataloader tickers failed: {meta['failed']}")
    if meta["non_massive_source"]:
        print(f"Non-Massive sources used: {meta['non_massive_source']}")

    fits = fit_all_assets(returns, cfg.EQUITY_TICKER, cfg.BOND_TICKER, cfg.ETF_TICKERS, cfg.RF)
    table = build_hurdle_table(fits, cfg.ETF_EXPECTED_RETURN, cfg.MU_EQUITY, cfg.MU_BOND, cfg.RF)
    table["coc_pct"] = (table["cost_of_capital"] * 100).round(1)
    table["spread_pct"] = (table["spread"] * 100).round(2)
    table["figure2_coc_pct"] = pd.Series(cfg.FIGURE2_COC)
    table["figure2_spread_pct"] = pd.Series(cfg.FIGURE2_SPREAD)
    table["clears_hurdle"] = table["spread"] > 0

    print("\nSeven-ETF hurdle table (regression route) vs Figure 2:")
    print(table[["beta_equity", "beta_bond", "coc_pct", "figure2_coc_pct", "spread_pct",
                 "figure2_spread_pct", "funding_gap", "appraisal_ratio", "clears_hurdle"]])

    hyg_decomposition = funding_decomposition(
        100.0, fits["HYG"].beta_equity, fits["HYG"].beta_bond, fits["HYG"].funding_gap
    )

    table.to_csv(f"{RESULTS_DIR}/exp1_hurdle_table.csv")
    with open(f"{RESULTS_DIR}/exp1_worked_hyg_example.json", "w") as f:
        json.dump(worked, f, indent=2)
    with open(f"{RESULTS_DIR}/exp1_hyg_regression_decomposition.json", "w") as f:
        json.dump(
            {
                "beta_equity": fits["HYG"].beta_equity,
                "beta_bond": fits["HYG"].beta_bond,
                "funding_gap": fits["HYG"].funding_gap,
                "decomposition_per_100": hyg_decomposition,
                "n_obs": fits["HYG"].n_obs,
                "r_squared": fits["HYG"].r_squared,
            },
            f,
            indent=2,
        )
    with open(f"{RESULTS_DIR}/exp1_data_meta.json", "w") as f:
        json.dump(
            {
                "ticker_details": [
                    {k: v for k, v in d.items() if k != "segments"} for d in meta["ticker_details"]
                ],
                "failed": meta["failed"],
                "non_massive_source": meta["non_massive_source"],
                "n_months": len(returns),
                "date_range": [str(returns.index.min()), str(returns.index.max())],
            },
            f,
            indent=2,
            default=str,
        )
    print("\nSaved results/exp1_hurdle_table.csv, exp1_worked_hyg_example.json, "
          "exp1_hyg_regression_decomposition.json, exp1_data_meta.json")


if __name__ == "__main__":
    main()
