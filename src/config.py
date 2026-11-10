"""Parameters used across the experiments, taken from the paper
"Total Portfolio Approach: Beyond the Buzzwords" (KnowRisk, November 2026),
which itself reports Staley, M. (2026) "Cost of Capital in the Total Portfolio
Approach" (King's Hill Research working paper) and the paper's own Dashboard
(data-route) calculations.

No classes or dictionaries: every parameter is a module-level UPPER_CASE
variable with a one-line comment naming its meaning, unit and source.
"""

# ---------------------------------------------------------------------------
# Data window and universe (paper, "Data" section, page 18; Dashboard
# settings table, page 4)
# ---------------------------------------------------------------------------
PRICE_START = "2008-03-31"  # one month before the first return date, needed to compute it
SAMPLE_START = "2008-04-30"  # first return month of the 216-month window (paper p.4, p.18)
SAMPLE_END = "2026-03-31"  # last month of the 216-month window (paper p.4, p.18)
N_MONTHS_PAPER = 216  # number of monthly observations, 30-Apr-2008 to 31-Mar-2026 (paper p.4)

EQUITY_TICKER = "ACWI"  # global-equity reference factor, MSCI ACWI ETF (paper "Data", p.18)
BOND_TICKER = "IEF"  # 10-year US Treasury proxy; paper names no exact ticker, so the
# standard iShares 7-10 Year Treasury total-return ETF is used (not specified in the paper)
ETF_TICKERS = ["DBC", "IWN", "VNQ", "HYG", "EMB", "LQD", "TLT"]  # seven candidate ETFs (paper p.2-3)

ASSET_NAMES = {  # display names, paper Figure 2 / Table 6
    "DBC": "Commodities",
    "IWN": "US small-cap value",
    "VNQ": "US REITs",
    "HYG": "US high yield",
    "EMB": "EM sovereign bonds",
    "LQD": "US investment grade",
    "TLT": "US long Treasuries",
}

# ---------------------------------------------------------------------------
# J.P. Morgan 2026 Long-Term Capital Market Assumptions, as used by Staley
# (paper section 2, page 2-3)
# ---------------------------------------------------------------------------
MU_EQUITY = 0.0828  # expected return of global equities (ACWI), % a year / 100 (paper p.2)
MU_BOND = 0.0488  # expected return of 10-year US Treasuries, % a year / 100 (paper p.2)
RF = 0.0310  # cash (risk-free) rate, % a year / 100 (paper p.2)

# Expected returns of the seven ETFs, Figure 2 (paper p.3). HYG is quoted both
# as 6.46% (worked example, p.3) and 6.5% (rounded in Figure 2); the unrounded
# figure is used here and the rounded one only for display/verification.
ETF_EXPECTED_RETURN = {
    "DBC": 0.062,
    "IWN": 0.089,
    "VNQ": 0.102,
    "HYG": 0.0646,
    "EMB": 0.066,
    "LQD": 0.057,
    "TLT": 0.058,
}

# Reference 60/40 portfolio (paper section 2-3)
W_EQUITY_RP = 0.60  # global-equity weight of the reference portfolio (paper p.2)
W_BOND_RP = 0.40  # 10-year Treasury weight of the reference portfolio (paper p.2)
SIGMA_RP = 0.1030  # annualised volatility of the 60/40 reference portfolio, Table 4 (paper p.6)

# ---------------------------------------------------------------------------
# Figure 2 worked example and verification targets (paper p.2-3)
# ---------------------------------------------------------------------------
HYG_BETA_EQUITY = 0.502  # Staley's rounded equity beta for HYG (paper p.2)
HYG_BETA_BOND = 0.142  # Staley's rounded bond beta for HYG (paper p.2)
HYG_COC_EXPECTED = 0.0595  # worked cost-of-capital check for HYG (paper p.2)
HYG_SPREAD_EXPECTED = 0.0051  # worked spread check for HYG (paper p.3)
HYG_FUNDING_GAP_EXPECTED = 0.356  # worked funding-gap check for HYG (paper p.3)
HYG_OWN_RISK_WORKED = 0.0613  # HYG own (residual) risk quoted in the worked example (paper p.3)
HYG_APPRAISAL_RATIO_EXPECTED = 0.08  # spread / own risk = 0.51 / 6.13 (paper p.3)

# Figure 2 table: rounded cost of capital and spread per ETF, % a year (paper
# p.3). Used only to check the regression-route output against the paper, to
# within the rounding the paper itself applies (one decimal place).
FIGURE2_COC = {
    "DBC": 5.0, "IWN": 8.2, "VNQ": 9.4, "HYG": 6.0, "EMB": 6.8, "LQD": 5.8, "TLT": 6.3,
}
FIGURE2_SPREAD = {
    "DBC": 1.19, "IWN": 0.74, "VNQ": 0.69, "HYG": 0.51, "EMB": -0.23, "LQD": -0.41, "TLT": -0.53,
}

# Table 6 (data-route) own (residual) risks, % a year (paper p.10)
TABLE6_OWN_RISK = {
    "IWN": 11.39, "HYG": 6.15, "TLT": 4.92, "DBC": 14.87, "VNQ": 13.77, "LQD": 4.59, "EMB": 6.64,
}
# Table 6 (data-route) spreads, % a year, used for the t-statistic guards (paper p.10)
TABLE6_SPREAD = {
    "IWN": 0.74, "HYG": 0.49, "TLT": -0.54, "DBC": 1.21, "VNQ": 0.65, "LQD": -0.42, "EMB": -0.23,
}
TABLE6_T_STAT = {  # paper p.10, for cross-checking src/guards.py
    "IWN": 0.27, "HYG": 0.33, "TLT": -0.46, "DBC": 0.34, "VNQ": 0.20, "LQD": -0.38, "EMB": -0.15,
}
TABLE6_SHARE_KEPT = {  # paper p.10
    "IWN": 0.07, "HYG": 0.10, "TLT": 0.18, "DBC": 0.11, "VNQ": 0.04, "LQD": 0.13, "EMB": 0.02,
}

# ---------------------------------------------------------------------------
# TPA optimisation (paper section 3 and Technical appendix A1, p.3-4, p.15)
# ---------------------------------------------------------------------------
ACTIVE_RISK_BUDGET = 0.02  # tracking-error budget against the 60/40, 200 bps a year (paper p.3)
FACTOR_SECONDARY_TICKERS = ["IWN", "HYG"]  # FACTOR residual-correlation secondary factors (paper p.10, Table 6)

# Table 1 scenarios (paper p.3-4): (tickers, borrowing_limit); borrowing_limit
# is the cap on sum(g_i * w_i); None means unlimited borrowing.
TABLE1_SCENARIOS = [
    ("two_etfs_borrowing", ["IWN", "HYG"], None),
    ("three_etfs_no_borrowing", ["IWN", "HYG", "TLT"], 0.0),
    ("seven_etfs_borrowing", ETF_TICKERS, None),
    ("seven_etfs_no_borrowing", ETF_TICKERS, 0.0),
]
TABLE1_EXPECTED_VA_BPS = {  # Table 1 value-add, bps a year (paper p.4)
    "two_etfs_borrowing": 21.0,
    "three_etfs_no_borrowing": 13.1,
    "seven_etfs_borrowing": 26.1,
    "seven_etfs_no_borrowing": 18.1,
}
RECONCILIATION_VA_BORROWING_BPS = 26.06  # solver reconciliation, all seven ETFs, borrowing (paper p.4)
RECONCILIATION_VA_NO_BORROWING_BPS = 18.08  # solver reconciliation, all seven ETFs, no borrowing (paper p.4)
RECONCILIATION_PRICE_OF_CASH_BPS = 71.1  # shadow price of cash, all seven ETFs (paper p.4)
SHADOW_PRICE_CASH_3ETF = 0.0079  # three-ETF shadow price of cash, % a year (paper p.4)
SHADOW_PRICE_CASH_7ETF = 0.0071  # seven-ETF shadow price of cash, % a year (paper p.4)
M2_HAIRCUT_7ETF_BPS = 7.1  # M2 haircut, seven-ETF no-borrowing case, bps a year (paper p.4)
TOTAL_VOL_7ETF = 0.1049  # total volatility, seven-ETF no-borrowing case (paper p.4)

# Appendix A3 reconciliation across routes, all seven ETFs, no borrowing, bps a year (paper p.16)
RECONCILIATION_TABLE = {
    "staley_spreadsheet": 18.08,
    "tpa_paper_parameters": 18.08,
    "tpa_explicit_formulas": 18.08,
    "data_route_actual_correlation": 17.60,
    "data_route_factor": 17.78,
}
DASHBOARD_WEIGHTS_EXPECTED = {  # data-route Best mix / Table 7 weights, % (paper p.4-5)
    "ACWI": 37.3, "IEF": 33.5, "VNQ": 10.2, "HYG": 6.7, "IWN": 5.9, "DBC": 4.1, "TLT": 2.4,
    "LQD": 0.0, "EMB": 0.0,
}

# ---------------------------------------------------------------------------
# Backtest look-back, Table 4 (paper p.6)
# ---------------------------------------------------------------------------
BEST_MIX_WEIGHTS = {  # Dashboard (data-route) Best mix, monthly rebalanced (paper p.4-5)
    "ACWI": 0.373, "IEF": 0.335, "VNQ": 0.102, "HYG": 0.067, "IWN": 0.059, "DBC": 0.041, "TLT": 0.024,
}
TABLE4_REFERENCE = {  # 60/40 reference backtest statistics (paper Table 4, p.6)
    "growth": 0.0596, "average_return": 0.0633, "volatility": 0.1029, "worst_fall": -0.319,
}
TABLE4_BEST_MIX = {  # Best mix backtest statistics (paper Table 4, p.6)
    "growth": 0.0574, "average_return": 0.0615, "volatility": 0.1048, "worst_fall": -0.328,
    "tracking_error": 0.0204,
}
GAP_MODEL_EXPECTATION_ANNUAL = 0.0018  # modelled annual gap, 17.8-18.1 bps a year (paper p.6)
GAP_BAND_ANNUAL = ACTIVE_RISK_BUDGET  # +/- 2% * sqrt(t) band around the modelled gap path (paper p.6, Fig.4)
GAP_SUMMED_EXPECTED = -0.033  # summed (arithmetic) end gap vs the model path (paper p.6)
GAP_COMPOUNDED_EXPECTED = -0.036  # compounded end gap (paper p.6)
GAP_INSIDE_TWO_IN_THREE_SHARE = 0.83  # share of months inside the one-sigma band (paper p.6)
INFORMATION_RATIO_EXPECTED = 0.09  # 0.18% / 2% (paper p.6)

# ---------------------------------------------------------------------------
# Stock-picker case, Table 5 and Figure 7 (paper p.6-9)
# ---------------------------------------------------------------------------
PICKER_ALPHA = 0.03  # stock picker's alpha over the small-cap index, % a year / 100 (paper p.6)
PICKER_TRACKING_ERROR = 0.0738  # stock picker's own (residual) risk / tracking error (paper p.6)
PICKER_RECORD_YEARS = 10  # length of the picker's track record, years (paper p.6, p.10)
PICKER_BETA_TICKER = "IWN"  # the picker is modelled with the small-cap index's betas (paper p.6)
SAA_SMALL_CAP_SLOT = 0.058  # traditional SAA small-cap allocation replaced by the picker (paper p.7)
TABLE5_VALUE_ADD_BPS = {  # Table 5 value-add decomposition, bps a year (paper p.7)
    "saa_total": 35.5, "saa_etf_mix": 18.1, "saa_picker": 17.4,
    "tpa_no_borrow_total": 53.4, "tpa_no_borrow_etf_mix": 18.1, "tpa_no_borrow_picker": 35.4,
    "tpa_borrow_total": 59.9, "tpa_borrow_etf_mix": 26.1, "tpa_borrow_picker": 33.8,
    "tpa_borrow_short_total": 85.4, "tpa_borrow_short_etf_mix": 26.1, "tpa_borrow_short_picker": 59.3,
}
TABLE5_HONEST_VALUE_ADD_BPS = {  # after the M2 haircut, bps a year (paper p.7)
    "saa": 28.0, "tpa_no_borrow": 46.3, "tpa_borrow": 52.7, "tpa_borrow_short": 78.2,
}
SAA_M2_HAIRCUT_BPS = 7.5  # SAA M2 haircut, bps a year (paper p.8)
ALPHA_GRID = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]  # Figure 7 alpha sweep, 0-3% in 0.5% steps (paper p.9)

# ---------------------------------------------------------------------------
# Forecast-error guards, Appendix A2 (paper p.10, p.16)
# ---------------------------------------------------------------------------
GUARD_YEARS_ETF = 18  # years of history behind the ETF t-statistics (paper p.10)
GUARD_YEARS_PICKER = PICKER_RECORD_YEARS  # years behind the picker's t-statistic (paper p.10)
MIN_IR_THRESHOLD = 0.47  # MinIR = t* / sqrt(Y) with t* = 2 over 18 years (paper p.10)
MIN_T_STAR = 2.0  # confidence threshold expressed as a t-statistic (paper p.10)

FIGURE8_RESULTS = {  # active risk taken (bps), value-add (bps), value-add after haircut (bps) (paper p.11)
    "no_guards": (200, 17.8, 10.6),
    "shrinkage_only": (200, 1.2, -5.9),
    "risk_rule_only": (38, 3.4, 3.1),
    "both_guards": (3, 0.0, 0.0),
}
PICKER_GUARDED_ALPHA_SHARE = 0.62  # share of the picker's alpha kept after shrinkage (paper p.10, Table 6)
PICKER_GUARDED_TPA_VA_BPS = 37.5  # TPA value-add with guards applied to the picker case (paper p.11)
PICKER_GUARDED_SAA_VA_BPS = 28.9  # SAA value-add with the same guards (paper p.11)

# ---------------------------------------------------------------------------
# Conditional drawdown, Figure 9 and the drawdown decomposition (paper p.12-14)
# ---------------------------------------------------------------------------
CDAR_TAIL_FRACTION = 0.05  # worst 5% of months averaged for the conditional drawdown (paper p.12)
CDAR_REFERENCE_60_40 = 0.252  # conditional drawdown of the plain 60/40 (paper p.13)
CDAR_BEST_MIX_UNCOMPOUNDED = 0.250  # Best mix conditional drawdown, uncompounded convention (paper p.12)
CDAR_BEST_MIX_COMPOUNDED = 0.235  # Best mix conditional drawdown, compounded convention (paper p.12)
CDAR_FLOOR_IN_BUDGET = 0.219  # lowest conditional drawdown reachable inside the 2% active-risk budget (paper p.13)
DRAWDOWN_DECOMPOSITION = {  # weight (%), share of drawdown (%), drawdown per $1 (paper p.13, Table)
    "ACWI": (37.3, 66.4, 0.44), "VNQ": (10.2, 20.8, 0.51), "IWN": (5.9, 7.2, 0.31),
    "DBC": (4.1, 5.9, 0.36), "HYG": (6.7, 5.0, 0.19), "TLT": (2.4, -0.5, -0.05), "IEF": (33.5, -4.8, -0.04),
}
DRAWDOWN_CEILING_CURVE = {  # ceiling (fraction), value-add (bps a year) (paper p.14, Figure 9)
    0.250: 17.8, 0.24: 17.2, 0.22: 9.0,
}
DRAWDOWN_BOOTSTRAP_BLOCK_MONTHS = 12  # block length for the block bootstrap (paper p.14)
DRAWDOWN_BOOTSTRAP_PATHS = 400  # number of resampled paths (paper p.14)
DRAWDOWN_BOOTSTRAP_THRESHOLD = 0.15  # drawdown threshold tested in the bootstrap (paper p.14)
DRAWDOWN_BOOTSTRAP_SHARE_MONTHS = 0.05  # "more than 5% of months" breach condition (paper p.14)
DRAWDOWN_BOOTSTRAP_EXPECTED_SHARE = 0.64  # share of paths breaching the condition (paper p.14)

# ---------------------------------------------------------------------------
# Runtime configuration (download and compute budget)
# ---------------------------------------------------------------------------
BOOTSTRAP_PATHS_RUN = 400  # executed block-bootstrap resamples, matching the paper's count (paper p.14)
