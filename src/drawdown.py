"""Conditional drawdown (CDaR-style tail-averaged underwater depth) and its
per-holding decomposition (paper section 9, "What the risk budget does not
control", p.12-14).

Conditional drawdown: for each month t, the underwater depth
D_t = 1 - W_t / max(W_1..W_t) on the portfolio's wealth path; the conditional
drawdown is the average of the worst 5% of these depths (about eleven of 216
months), following the tail-averaging convention of Rockafellar and Uryasev's
(2002) CVaR and Chekhlov, Uryasev and Zabarankin's (2005) conditional
drawdown-at-risk, as referenced by the paper (p.12).
"""

import numpy as np
import pandas as pd


def wealth_path(returns: pd.Series, compounded: bool) -> pd.Series:
    """Wealth path from monthly returns: compounded (cumulative product) or
    uncompounded (running sum, paper p.12: 'the uncompounded measure scales
    with holdings')."""
    if compounded:
        return (1.0 + returns).cumprod()
    return 1.0 + returns.cumsum()


def underwater_depth(wealth: pd.Series) -> pd.Series:
    """D_t = 1 - W_t / running-max(W) (uncompounded: running max of the level,
    not a ratio, since the uncompounded path is not a ratio scale; use the
    additive peak-to-current gap instead, scaled by the running peak)."""
    running_peak = wealth.cummax()
    return 1.0 - wealth / running_peak


def conditional_drawdown(returns: pd.Series, tail_fraction: float, compounded: bool = False) -> float:
    """Average of the worst ``tail_fraction`` share of monthly underwater
    depths (paper p.12)."""
    wealth = wealth_path(returns, compounded)
    depth = underwater_depth(wealth)
    n_tail = max(1, int(round(len(depth) * tail_fraction)))
    worst = depth.nlargest(n_tail)
    return float(worst.mean())


def tail_months(returns: pd.Series, tail_fraction: float, compounded: bool = False) -> pd.Index:
    """The months in the worst ``tail_fraction`` tail of the underwater path,
    used to align the per-holding decomposition (paper p.13)."""
    wealth = wealth_path(returns, compounded)
    depth = underwater_depth(wealth)
    n_tail = max(1, int(round(len(depth) * tail_fraction)))
    return depth.nlargest(n_tail).index


def per_holding_decomposition(
    returns: pd.DataFrame, weights: dict, tail_fraction: float
) -> pd.DataFrame:
    """Additive decomposition of the uncompounded conditional drawdown into
    per-holding contributions (paper p.13: "the uncompounded measure scales
    with holdings, so shares are exact").

    For the uncompounded wealth path, W_t = 1 + sum_i w_i * R_i,t (cumulative),
    so the peak-to-current gap at the portfolio's own tail months splits
    additively across holdings: contribution_i = w_i * mean_t(peak - R_i,cum,t)
    over the portfolio's own worst-tail months.
    """
    portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
    wealth = wealth_path(portfolio_returns, compounded=False)
    running_peak = wealth.cummax()
    tail_idx = tail_months(portfolio_returns, tail_fraction, compounded=False)

    rows = []
    total_drawdown = float((1.0 - wealth / running_peak).loc[tail_idx].mean())
    for ticker, w in weights.items():
        holding_cum = w * returns[ticker].cumsum()
        # Each holding's own cumulative contribution relative to the same
        # portfolio-level peak dates and tail months (Euler/additive split of
        # the linear uncompounded path, paper p.13).
        holding_drawdown_per_dollar = []
        for t in tail_idx:
            peak_idx = wealth.loc[:t].idxmax()
            peak_level = wealth.loc[peak_idx]
            # Normalise by the portfolio's own running peak so that summing
            # every holding's contribution reproduces the normalised depth
            # 1 - W_t/peak exactly (paper p.13: "shares are exact").
            gap = (holding_cum.loc[peak_idx] - holding_cum.loc[t]) / peak_level
            holding_drawdown_per_dollar.append(gap)
        mean_gap = float(np.mean(holding_drawdown_per_dollar))
        rows.append(
            {
                "ticker": ticker,
                "weight": w,
                "drawdown_contribution": mean_gap,
                "share_of_total": mean_gap / total_drawdown if total_drawdown != 0 else np.nan,
                "drawdown_per_dollar": mean_gap / w if w != 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


def block_bootstrap_breach_share(
    returns: pd.DataFrame,
    weights: dict,
    block_months: int,
    n_paths: int,
    threshold: float,
    share_months: float,
    rng: np.random.Generator,
) -> float:
    """Resample the monthly return history in contiguous ``block_months``
    blocks ``n_paths`` times (with replacement across blocks) and compute the
    share of paths on which the portfolio's uncompounded underwater depth
    exceeds ``threshold`` in more than ``share_months`` of months (paper
    p.14, block bootstrap)."""
    portfolio_returns = (returns * pd.Series(weights)).sum(axis=1).to_numpy()
    n = len(portfolio_returns)
    n_blocks = int(np.ceil(n / block_months))
    starts = np.arange(0, n - block_months + 1)
    breaches = 0
    for _ in range(n_paths):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        path = np.concatenate([portfolio_returns[s : s + block_months] for s in chosen])[:n]
        wealth = 1.0 + np.cumsum(path)
        peak = np.maximum.accumulate(wealth)
        depth = 1.0 - wealth / peak
        if (depth > threshold).mean() > share_months:
            breaches += 1
    return breaches / n_paths
