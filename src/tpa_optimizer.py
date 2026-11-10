"""The TPA allocation problem (paper section 3 and Technical appendix A1,
p.3-4, p.15):

    maximise   sum_i w_i * s_i
    subject to omega = sqrt(w' Omega w) = budget
               lower <= w <= upper
               sum_i g_i * w_i <= borrowing_limit
               (optional) completion floors c_E, c_D >= 0

With no binding bounds the closed-form solution is the Treynor-Black rule
w = k * Omega^-1 * (s - p*g), k chosen so omega = budget. Because the
objective is linear and (absent upper bounds) the feasible region is a convex
cone scaled by the active-risk ball, the inequality version
sqrt(w'Omega w) <= budget always binds at the optimum, so the problem is the
convex quadratically-constrained program

    maximise   s' w
    subject to w' Omega w <= budget^2
               lower <= w <= upper
               g' w <= borrowing_limit

solved with cvxpy (CLARABEL), which also returns the constraints' dual variables;
these are the shadow prices Steiner (2026, cited paper p.4) shows are the dual
prices of Staley's charge-and-check loop. Using a disciplined convex solver
instead of a general nonlinear one avoids the small bound violations that a
local NLP solver can leave on this equality-constrained, near-degenerate
problem.
"""

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np


@dataclass
class TPASolution:
    weights: dict
    completion_equity: float
    completion_bond: float
    cash: float
    value_add: float  # sum w_i s_i, fraction a year
    active_risk: float  # sqrt(w' Omega w)
    shadow_price_cash: float  # multiplier on the borrowing constraint, fraction a year
    total_vol: float
    m2_haircut: float
    honest_value_add: float
    success: bool
    message: str = ""
    extra: dict = field(default_factory=dict)


def treynor_black_unconstrained(s: np.ndarray, omega_inv: np.ndarray, budget: float) -> np.ndarray:
    """Closed-form Treynor-Black weights scaled to exactly ``budget`` of active
    risk, ignoring the borrowing limit and bounds (paper Appendix A1)."""
    raw = omega_inv @ s
    scale = budget / np.sqrt(raw @ s)
    return scale * raw


def m2_haircut(total_vol: float, sigma_rp: float, mu_rp_minus_rf: float) -> float:
    """(mu_RP - rf) * (sigma_total / sigma_RP - 1) (paper Appendix A1)."""
    return mu_rp_minus_rf * (total_vol / sigma_rp - 1.0)


def solve_tpa(
    tickers: list,
    spreads: dict,
    funding_gaps: dict,
    beta_equity: dict,
    beta_bond: dict,
    omega: np.ndarray,
    budget: float,
    borrowing_limit: float | None,
    *,
    w_equity_rp: float = 0.60,
    w_bond_rp: float = 0.40,
    sigma_rp: float = 0.1030,
    mu_rp_minus_rf: float = 0.0382,
    lower: dict | None = None,
    upper: dict | None = None,
    require_nonneg_completion: bool = False,
) -> TPASolution:
    """Solve the TPA problem (paper Appendix A1) for the given asset set.

    A non-ETF asset (the stock picker, exp_5) is added like any other asset:
    include its ticker in ``tickers`` with its own spread/gap/betas entries,
    and give it its own ``lower``/``upper`` bound to allow shorting.
    """
    n = len(tickers)
    s = np.array([spreads[t] for t in tickers])
    g = np.array([funding_gaps[t] for t in tickers])
    bE = np.array([beta_equity[t] for t in tickers])
    bD = np.array([beta_bond[t] for t in tickers])
    lower_arr = np.array([0.0 if lower is None else lower.get(t, 0.0) for t in tickers])
    upper_arr = np.array([np.inf if upper is None else upper.get(t, np.inf) for t in tickers])

    # Symmetrise and lightly regularise Omega: sample residual-covariance
    # estimates can be asymmetric by floating-point noise or near-singular,
    # which breaks cvxpy's PSD check for quad_form.
    omega_psd = 0.5 * (omega + omega.T) + 1e-12 * np.eye(n)

    w = cp.Variable(n)
    risk_constraint = cp.quad_form(w, omega_psd) <= budget**2
    constraints = [risk_constraint, w >= lower_arr]
    if np.any(np.isfinite(upper_arr)):
        constraints.append(w <= np.where(np.isfinite(upper_arr), upper_arr, 1e6))
    borrowing_constraint = None
    if borrowing_limit is not None:
        borrowing_constraint = g @ w <= borrowing_limit
        constraints.append(borrowing_constraint)
    completion_equity_constraint = completion_bond_constraint = None
    if require_nonneg_completion:
        completion_equity_constraint = bE @ w <= w_equity_rp
        completion_bond_constraint = bD @ w <= w_bond_rp
        constraints += [completion_equity_constraint, completion_bond_constraint]

    problem = cp.Problem(cp.Maximize(s @ w), constraints)
    problem.solve(solver=cp.CLARABEL)
    success = problem.status in ("optimal", "optimal_inaccurate")
    if not success:
        # Fall back to SCS, which handles poorly scaled or near-degenerate
        # quadratic constraints more robustly than CLARABEL.
        problem.solve(solver=cp.SCS)
        success = problem.status in ("optimal", "optimal_inaccurate")

    w_val = np.clip(w.value, lower_arr, np.where(np.isfinite(upper_arr), upper_arr, w.value))
    shadow_price_cash = (
        float(borrowing_constraint.dual_value) if borrowing_constraint is not None else 0.0
    )

    completion_equity = w_equity_rp - float(bE @ w_val)
    completion_bond = w_bond_rp - float(bD @ w_val)
    cash = -float(g @ w_val)
    value_add = float(w_val @ s)
    omega_val = float(np.sqrt(max(w_val @ omega_psd @ w_val, 0.0)))
    total_vol = float(np.sqrt(sigma_rp**2 + omega_val**2))
    haircut = m2_haircut(total_vol, sigma_rp, mu_rp_minus_rf)
    weights = {t: float(w_val[i]) for i, t in enumerate(tickers)}
    return TPASolution(
        weights=weights,
        completion_equity=completion_equity,
        completion_bond=completion_bond,
        cash=cash,
        value_add=value_add,
        active_risk=omega_val,
        shadow_price_cash=shadow_price_cash,
        total_vol=total_vol,
        m2_haircut=haircut,
        honest_value_add=value_add - haircut,
        success=success,
        message=str(problem.status),
        extra={
            "completion_equity_dual": completion_equity_constraint.dual_value
            if completion_equity_constraint is not None
            else None,
            "completion_bond_dual": completion_bond_constraint.dual_value
            if completion_bond_constraint is not None
            else None,
        },
    )
