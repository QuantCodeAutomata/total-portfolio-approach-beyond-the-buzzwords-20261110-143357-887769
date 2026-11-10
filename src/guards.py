"""Forecast-error guards: t-statistics, empirical-Bayes shrinkage and the
signal-to-noise active-risk rule (paper "Guards against forecast error",
p.10-11, Technical appendix A2, p.16).

t_i = s_i * sqrt(Y) / sigma_eps_i
shrinkage: s_i -> s_i * t_i^2 / (1 + t_i^2)   (Efron-Morris, 1973, empirical
    Bayes shrinkage toward zero when the prior variance equals the squared
    estimate; cited by the paper as the modelling source, not a formal test)
MinIR = t_star / sqrt(Y)
omega_used = budget * min(1, IR / MinIR); hold the reference portfolio if IR <= 0
"""

import numpy as np


def t_statistic(spread_value: float, own_risk: float, years: float) -> float:
    """t_i = s_i * sqrt(Y) / sigma_eps_i (paper Appendix A2)."""
    assert own_risk > 0, "own risk must be positive"
    return spread_value * np.sqrt(years) / own_risk


def shrink_spread(spread_value: float, t_stat: float) -> float:
    """s_i -> s_i * t^2 / (1 + t^2) (paper Appendix A2, Efron-Morris shrinkage)."""
    shrinkage_weight = t_stat**2 / (1.0 + t_stat**2)
    return spread_value * shrinkage_weight


def shrinkage_weight(t_stat: float) -> float:
    """Share of the spread kept after shrinkage, t^2 / (1 + t^2)."""
    return t_stat**2 / (1.0 + t_stat**2)


def min_information_ratio(t_star: float, years: float) -> float:
    """MinIR = t* / sqrt(Y) (paper Appendix A2)."""
    return t_star / np.sqrt(years)


def risk_rule_active_risk(budget: float, information_ratio: float, min_ir: float) -> float:
    """omega_used = budget * min(1, IR / MinIR); 0 if IR <= 0 (paper Appendix A2,
    following Grinold and Kahn: optimal active risk rises with the information
    ratio)."""
    if information_ratio <= 0:
        return 0.0
    return budget * min(1.0, information_ratio / min_ir)
