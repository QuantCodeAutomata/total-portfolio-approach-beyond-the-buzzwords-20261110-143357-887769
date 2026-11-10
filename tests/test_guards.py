"""Unit tests for src/guards.py: t-statistics, empirical-Bayes shrinkage and
the signal-to-noise active-risk rule (paper Appendix A2, p.10-11, p.16)."""

import pytest

import src.config as cfg
from src.guards import (
    min_information_ratio,
    risk_rule_active_risk,
    shrink_spread,
    shrinkage_weight,
    t_statistic,
)


def test_table6_t_statistics_match_paper():
    """Paper Table 6, p.10: t_i = s_i*sqrt(18)/sigma_eps_i for the seven ETFs."""
    for ticker, expected_t in cfg.TABLE6_T_STAT.items():
        t = t_statistic(
            cfg.TABLE6_SPREAD[ticker] / 100.0, cfg.TABLE6_OWN_RISK[ticker] / 100.0, cfg.GUARD_YEARS_ETF
        )
        assert t == pytest.approx(expected_t, abs=0.02)


def test_picker_t_statistic_matches_paper():
    """Paper p.10: picker t = 3.00%/7.38% over 10 years = 1.29."""
    t = t_statistic(cfg.PICKER_ALPHA, cfg.PICKER_TRACKING_ERROR, cfg.PICKER_RECORD_YEARS)
    assert t == pytest.approx(1.29, abs=0.02)


def test_shrinkage_weight_matches_table6_shares():
    for ticker, expected_share in cfg.TABLE6_SHARE_KEPT.items():
        t = t_statistic(
            cfg.TABLE6_SPREAD[ticker] / 100.0, cfg.TABLE6_OWN_RISK[ticker] / 100.0, cfg.GUARD_YEARS_ETF
        )
        share = shrinkage_weight(t)
        assert share == pytest.approx(expected_share, abs=0.01)


def test_shrinkage_weight_bounds():
    """t^2/(1+t^2) must lie strictly between 0 and 1 and increase with |t|."""
    w_small = shrinkage_weight(0.1)
    w_large = shrinkage_weight(5.0)
    assert 0.0 < w_small < w_large < 1.0


def test_shrink_spread_preserves_sign():
    assert shrink_spread(0.01, 0.5) > 0
    assert shrink_spread(-0.01, 0.5) < 0


def test_min_information_ratio_matches_paper_threshold():
    min_ir = min_information_ratio(cfg.MIN_T_STAR, cfg.GUARD_YEARS_ETF)
    assert min_ir == pytest.approx(cfg.MIN_IR_THRESHOLD, abs=0.01)


def test_risk_rule_holds_reference_when_ir_non_positive():
    assert risk_rule_active_risk(cfg.ACTIVE_RISK_BUDGET, -0.1, cfg.MIN_IR_THRESHOLD) == 0.0
    assert risk_rule_active_risk(cfg.ACTIVE_RISK_BUDGET, 0.0, cfg.MIN_IR_THRESHOLD) == 0.0


def test_risk_rule_caps_at_full_budget():
    very_high_ir = 10.0
    assert risk_rule_active_risk(cfg.ACTIVE_RISK_BUDGET, very_high_ir, cfg.MIN_IR_THRESHOLD) == pytest.approx(
        cfg.ACTIVE_RISK_BUDGET
    )


def test_risk_rule_scales_linearly_below_threshold():
    half_threshold_ir = cfg.MIN_IR_THRESHOLD / 2
    used = risk_rule_active_risk(cfg.ACTIVE_RISK_BUDGET, half_threshold_ir, cfg.MIN_IR_THRESHOLD)
    assert used == pytest.approx(cfg.ACTIVE_RISK_BUDGET / 2, rel=1e-6)
