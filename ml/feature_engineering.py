"""
feature_engineering.py
======================
Feature engineering pipeline for the UK Electricity Price Forecasting Engine.

Reads the cleaned master dataset and constructs a rich feature matrix ready
for XGBoost model training. Every feature group has a documented rationale
grounded in electricity market domain knowledge and empirical analysis
(ACF/PACF, rolling sensitivity, Mutual Information) on the 2024-2025 dataset.

Feature groups
--------------
1.  Outlier winsorisation  — cap extreme prices before lag/rolling creation
                             to prevent the Jan-2025 ~£700 spike contaminating
                             downstream features. Target is NOT winsorised.
2.  Price lag features     — lag_1 to lag_9 (short-range, MI-validated),
                             lag_48 (daily cycle), lag_336 (weekly cycle)
3.  Demand lag features    — demand_lag_1, demand_lag_48
4.  Price rolling stats    — mean_1h, std_24h, mean_24h (data-optimal windows)
5.  Demand rolling stats   — mean_1h, mean_24h
6.  Price momentum         — 30-min and 2-hour rate-of-change
7.  Price position         — price vs daily rolling mean; daily price range
8.  Demand context         — demand vs daily mean; demand momentum
9.  Cyclical time          — hour, day-of-week, month sin/cos pairs
10. Calendar flags         — is_weekend, is_holiday, is_morning_peak,
                             is_evening_peak, is_overnight, settlement_period
11. Generation mix ratios  — wind, gas, nuclear fraction of total
12. Supply-demand balance  — supply_demand_ratio, interconnector_net_mw,
                             renewable_surplus
13. Weather features       — hdd, wind_power_proxy, wind_surprise,
                             temp_spread, carbon_forecast_error
14. Interaction features   — wind x demand, cold x evening peak
15. Target variable        — price_next (T+30 min, shift -1 on raw price)

Design constraints
------------------
* Strictly chronological — no data leakage across the time axis.
* All lag/rolling features use shift(1) before rolling so the current row
  is never included in its own window.
* Winsorisation is applied only to the price series used for feature
  construction, never to the target variable.
* Rows with NaN values from burn-in (first 336 rows) and the final row
  (NaN target) are dropped after all features are built.

Usage
-----
    python feature_engineering.py                          # defaults
    python feature_engineering.py \\
        --input  Data/processed/master_clean.parquet \\
        --output Data/processed/features.parquet \\
        --log-level DEBUG

Dependencies
------------
    pip install pandas numpy pyarrow holidays
    ('holidays' is optional but strongly recommended for is_holiday flag)

Author: UK Electricity Price Forecasting Engine
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import holidays as holidays_lib
    _HOLIDAYS_AVAILABLE = True
except ImportError:
    _HOLIDAYS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Half-periods per time window (each period = 30 min)
PERIODS_PER_HOUR: int = 2
PERIODS_PER_DAY: int = 48
PERIODS_PER_WEEK: int = 336

# Winsorisation percentiles applied to price before lag/rolling feature
# construction. The target variable (price_next) is NEVER winsorised.
# The Jan-2025 ~£700 spike sits at the extreme tail; capping at 99.9th
# percentile preserves genuine high-price episodes while preventing one
# extreme event from distorting hundreds of downstream feature rows.
WINSOR_LOWER_QUANTILE: float = 0.001
WINSOR_UPPER_QUANTILE: float = 0.999

# UK base temperature for Heating Degree Days (industry standard)
HDD_BASE_TEMP_C: float = 15.5

# Peak period hour boundaries (UTC).
# UK is UTC in winter, UTC+1 in summer — using UTC throughout for consistency.
MORNING_PEAK_START: int = 7
MORNING_PEAK_END: int = 9
EVENING_PEAK_START: int = 16
EVENING_PEAK_END: int = 19
OVERNIGHT_START: int = 23
OVERNIGHT_END: int = 5

# Interconnector columns present in master_clean.parquet.
# Positive values = import into UK (supply supplement).
INTERCONNECTOR_COLS: list[str] = [
    "gen_intfr_mw",    # France (IFA1)
    "gen_intned_mw",   # Netherlands (BritNed)
    "gen_intirl_mw",   # Ireland (Moyle)
    "gen_intew_mw",    # East-West Ireland
    "gen_intnem_mw",   # NEMO Link (Belgium)
    "gen_intifa2_mw",  # IFA2 (France)
    "gen_intelec_mw",  # ElecLink (France)
    "gen_intvik_mw",   # Viking Link (Denmark)
]

# ---------------------------------------------------------------------------
# Lag specifications: (output_col, source_col, n_periods)
#
# Determined from ACF/PACF and Mutual Information analysis on master_clean.
# Short-range (lag_1-9): MI above median throughout. lag_3 has PACF ~0
# (linear chain explains it) but MI=0.49 — strong non-linear signal for
# XGBoost. lag_10+ drop below MI median: excluded.
# Seasonal: lag_48 (daily) and lag_336 (weekly) retained on structural
# grounds — electricity markets have hard daily/weekly rhythms.
# ---------------------------------------------------------------------------
LAG_SPECS: list[tuple[str, str, int]] = [
    # Short-range price lags (winsorised)
    ("price_lag_1",   "price_winsorised", 1),    # 30 min  | PACF=0.83, MI=0.88
    ("price_lag_2",   "price_winsorised", 2),    # 1 hour  | PACF=0.065, MI=0.61
    ("price_lag_3",   "price_winsorised", 3),    # 1.5 hrs | PACF~0, MI=0.49 non-linear
    ("price_lag_4",   "price_winsorised", 4),    # 2 hours | PACF=0.113, MI=0.40
    ("price_lag_5",   "price_winsorised", 5),    # 2.5 hrs | MI=0.34
    ("price_lag_6",   "price_winsorised", 6),    # 3 hours | MI=0.30
    ("price_lag_7",   "price_winsorised", 7),    # 3.5 hrs | MI=0.27
    ("price_lag_8",   "price_winsorised", 8),    # 4 hours | MI=0.25
    ("price_lag_9",   "price_winsorised", 9),    # 4.5 hrs | MI=0.24 at median
    # Seasonal price lags
    ("price_lag_48",  "price_winsorised", 48),   # 24 hrs  | MI=0.23, daily cycle
    ("price_lag_336", "price_winsorised", 336),  # 1 week  | weekly seasonality
    # Demand lags
    ("demand_lag_1",  "demand_mw",        1),    # 30 min  | primary price driver
    ("demand_lag_48", "demand_mw",        48),   # 24 hrs  | daily demand cycle
]

# ---------------------------------------------------------------------------
# Rolling specifications: (output_col, source_col, window_periods, statistic)
#
# Price rolling windows chosen from sensitivity analysis (peak Spearman
# correlation vs price_next) and MI ranking.
#   mean_1h (2p):  peak of both Pearson & Spearman curves; MI rank 3rd overall.
#   std_24h (48p): highest MI of all std candidates; Spearman peaked at ~21h.
#   mean_24h (48p): daily context anchor.
# Demand rolling gives the model demand regime context beyond single-point lags.
# ---------------------------------------------------------------------------
ROLLING_SPECS: list[tuple[str, str, int, str]] = [
    # Price rolling (winsorised, shifted by 1 before rolling)
    ("rolling_mean_1h",          "price_winsorised", 2,  "mean"),
    ("rolling_std_24h",          "price_winsorised", 48, "std"),
    ("rolling_mean_24h",         "price_winsorised", 48, "mean"),
    # Demand rolling (shifted by 1 before rolling)
    ("demand_rolling_mean_1h",   "demand_mw",        2,  "mean"),
    ("demand_rolling_mean_24h",  "demand_mw",        48, "mean"),
]

# Columns to drop from the final feature matrix before saving.
# These are raw original columns that must not be fed to the model:
#
# settlement_date    — duplicate of the DatetimeIndex, adds nothing
# price_ssp_gbp_mwh — System Sell Price, a direct component of price_mid_gbp_mwh
# price_sbp_gbp_mwh — System Buy Price, a direct component of price_mid_gbp_mwh
# price_mid_gbp_mwh — the raw target variable itself. Keeping it would give the
#                     model the current period's price as a feature, which is
#                     data leakage — at inference time this value is unknown.
#                     (The lagged versions price_lag_1 etc. are safe to use.)
COLS_TO_DROP: list[str] = [
    "settlement_date",
    "price_ssp_gbp_mwh",
    "price_sbp_gbp_mwh",
    "price_mid_gbp_mwh",
]

# Generation mix ratio specs: (output_col, numerator_col)
RATIO_SPECS: list[tuple[str, str]] = [
    ("wind_ratio",    "gen_wind_mw"),
    ("gas_ratio",     "gen_ccgt_mw"),
    ("nuclear_ratio", "gen_nuclear_mw"),
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _configure_logging(level: str) -> logging.Logger:
    """Configure and return the module-level logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("feature_engineering")


# ---------------------------------------------------------------------------
# Feature engineering functions
# ---------------------------------------------------------------------------

def add_winsorised_price(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Add price_winsorised — a clip-bounded copy of price_mid_gbp_mwh.

    Used exclusively for constructing lag and rolling features. Prevents
    extreme events (e.g. the Jan-2025 ~£700 spike) from propagating into
    feature rows that are otherwise in a normal market regime.

    The target variable always uses the original, unclipped price.
    """
    lower = df["price_mid_gbp_mwh"].quantile(WINSOR_LOWER_QUANTILE)
    upper = df["price_mid_gbp_mwh"].quantile(WINSOR_UPPER_QUANTILE)
    df["price_winsorised"] = df["price_mid_gbp_mwh"].clip(lower, upper)
    n_clipped = (
        (df["price_mid_gbp_mwh"] < lower) | (df["price_mid_gbp_mwh"] > upper)
    ).sum()
    logger.info(
        "Winsorisation: lower=£%.2f (p%.1f)  upper=£%.2f (p%.1f)  "
        "clipped %d row(s) (%.3f%%).",
        lower, WINSOR_LOWER_QUANTILE * 100,
        upper, WINSOR_UPPER_QUANTILE * 100,
        n_clipped, 100.0 * n_clipped / len(df),
    )
    return df


def add_lag_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Add backward-looking lag features for price (winsorised) and demand.

    shift(N) moves values down N rows so row t receives the value from t-N.
    No future information is introduced.
    """
    logger.info("Creating %d lag feature(s) ...", len(LAG_SPECS))
    for col_name, source_col, n_periods in LAG_SPECS:
        if source_col not in df.columns:
            raise KeyError(
                f"Source column '{source_col}' not found for lag '{col_name}'."
            )
        df[col_name] = df[source_col].shift(n_periods)
        logger.debug("  %-28s  <- shift(%s, %d)", col_name, source_col, n_periods)
    return df


def add_rolling_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Add rolling window statistics on price (winsorised) and demand.

    Each source column is shifted by 1 before rolling so the window at row t
    covers [t-window, t-1] exclusively — never including t itself.
    This mirrors inference conditions where the current period's value is
    the one being predicted, not an observable input.
    """
    logger.info("Creating %d rolling feature(s) ...", len(ROLLING_SPECS))
    shifted_cache: dict[str, pd.Series] = {}

    for col_name, source_col, window, stat in ROLLING_SPECS:
        if source_col not in df.columns:
            raise KeyError(
                f"Source column '{source_col}' not found for rolling '{col_name}'."
            )
        if source_col not in shifted_cache:
            shifted_cache[source_col] = df[source_col].shift(1)

        roller = shifted_cache[source_col].rolling(window=window, min_periods=window)
        if stat == "mean":
            df[col_name] = roller.mean()
        elif stat == "std":
            df[col_name] = roller.std()
        else:
            raise ValueError(f"Unsupported rolling statistic: '{stat}'.")
        logger.debug(
            "  %-28s  <- rolling_%s(%s, w=%d)",
            col_name, stat, source_col, window,
        )
    return df


def add_price_momentum(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Add price rate-of-change (momentum) features.

    Provides an explicit direction signal — whether prices are rising or
    falling. XGBoost can infer this by comparing lag values, but an explicit
    difference makes it easier and more robust to learn.

    price_momentum_1p: change over the last 30 minutes (immediate direction)
    price_momentum_4p: change over the last 2 hours (trend direction)

    Both use winsorised price to avoid spike contamination.
    """
    logger.info("Creating price momentum features ...")
    p = df["price_winsorised"]
    df["price_momentum_1p"] = p.shift(1) - p.shift(2)
    df["price_momentum_4p"] = p.shift(1) - p.shift(5)
    logger.debug("  price_momentum_1p  <- shift(1) - shift(2)")
    logger.debug("  price_momentum_4p  <- shift(1) - shift(5)")
    return df


def add_price_position_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Add features describing price position relative to its recent context.

    price_vs_daily_mean: last observed price minus the 24h rolling mean.
      A price of £80 is unremarkable if the day mean is £78, but significant
      if the day mean is £55. This relative position is a key signal.

    daily_price_range: rolling 24h max minus min. Captures the volatility
      regime for the day. A wide range = chaotic market; narrow = stable.

    Requires rolling_mean_24h to already exist in df.
    """
    logger.info("Creating price position features ...")
    if "rolling_mean_24h" not in df.columns:
        raise RuntimeError(
            "rolling_mean_24h must be created before add_price_position_features."
        )
    p_s1 = df["price_winsorised"].shift(1)
    df["price_vs_daily_mean"] = p_s1 - df["rolling_mean_24h"]
    roll_24 = p_s1.rolling(PERIODS_PER_DAY, min_periods=PERIODS_PER_DAY)
    df["daily_price_range"] = roll_24.max() - roll_24.min()
    logger.debug("  price_vs_daily_mean  <- price_lag_1 - rolling_mean_24h")
    logger.debug("  daily_price_range    <- rolling_24h_max - rolling_24h_min")
    return df


def add_demand_context_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Add demand deviation and momentum features.

    demand_vs_daily_mean: demand relative to its 24h rolling mean. Captures
      whether demand is unusually high or low for the time of day — the key
      driver of marginal generator dispatch decisions.

    demand_momentum: rate of change of demand over 30 minutes. Rising demand
      is a leading indicator of rising price as progressively more expensive
      generators are called online.

    Requires demand_rolling_mean_24h to already exist in df.
    """
    logger.info("Creating demand context features ...")
    if "demand_rolling_mean_24h" not in df.columns:
        raise RuntimeError(
            "demand_rolling_mean_24h must be created before add_demand_context_features."
        )
    d = df["demand_mw"]
    df["demand_vs_daily_mean"] = d.shift(1) - df["demand_rolling_mean_24h"]
    df["demand_momentum"]      = d.shift(1) - d.shift(2)
    logger.debug("  demand_vs_daily_mean  <- demand_lag_1 - demand_rolling_mean_24h")
    logger.debug("  demand_momentum       <- demand shift(1) - shift(2)")
    return df


def add_cyclical_time_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Encode cyclical time components using sine/cosine projections.

    Raw hour/day/month integers place midnight (0) and 23:30 far apart on
    the number line. Projecting onto a unit circle makes them geometrically
    adjacent. Both sin and cos are always required — sin alone is ambiguous
    (06:00 and 18:00 share the same sine value).

    Three cycles encoded: 24-hour day, 7-day week, 12-month year.
    """
    logger.info("Creating cyclical time encoding features ...")
    idx = df.index

    hour = idx.hour + idx.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    dow = idx.dayofweek.astype(float)
    df["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7.0)

    month = idx.month.astype(float)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)

    logger.debug("  hour, day_of_week, month sin/cos pairs created")
    return df


def add_calendar_flags(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Add binary and ordinal calendar features capturing market structure.

    settlement_period: ordinal 1-48 for the half-hour slot within the day.
      Distinct from cyclical hour features — this is a direct market label
      that the model can use to learn period-specific mean price levels.

    is_weekend: demand is structurally 10-15% lower on weekends; the same
      generation mix carries different price implications on a Sunday.

    is_holiday: UK public holidays have Sunday-like demand profiles regardless
      of the calendar day. Requires the 'holidays' package (pip install holidays).

    is_morning_peak (07:00-09:00 UTC): commuter and industrial load ramp.
    is_evening_peak (16:00-19:00 UTC): the highest-price, highest-volatility
      period in UK electricity markets. The sin/cos encoding approximates
      this but a binary flag nails it precisely.
    is_overnight (23:00-05:00 UTC): low-demand trough where negative prices
      are most likely to occur.
    """
    logger.info("Creating calendar flag features ...")
    idx = df.index

    df["settlement_period"] = (idx.hour * 2 + idx.minute // 30 + 1).astype(np.int8)
    df["is_weekend"]        = (idx.dayofweek >= 5).astype(np.int8)

    if _HOLIDAYS_AVAILABLE:
        years = sorted(idx.year.unique().tolist())
        uk_hols = holidays_lib.UK(years=years)
        # Convert holiday date objects (datetime.date) to tz-aware Timestamps
        # before isin() comparison. The holidays library returns plain date keys;
        # the DataFrame index is tz-aware UTC. Without this conversion isin()
        # silently returns all-False because Timestamp(tz=UTC) never equals a
        # naive datetime.date object — the root cause of is_holiday=0.
        hol_index = pd.DatetimeIndex(
            [pd.Timestamp(d, tz="UTC") for d in uk_hols.keys()]
        )
        df["is_holiday"] = idx.normalize().isin(hol_index).astype(np.int8)
        logger.debug(
            "  is_holiday: %d UK public holidays loaded, %d periods flagged.",
            len(uk_hols), int(df["is_holiday"].sum()),
        )
    else:
        df["is_holiday"] = np.int8(0)
        logger.warning(
            "  'holidays' package not installed — is_holiday=0 for all rows. "
            "Install with: pip install holidays"
        )

    hour = idx.hour
    df["is_morning_peak"] = (
        (hour >= MORNING_PEAK_START) & (hour <= MORNING_PEAK_END)
    ).astype(np.int8)
    df["is_evening_peak"] = (
        (hour >= EVENING_PEAK_START) & (hour <= EVENING_PEAK_END)
    ).astype(np.int8)
    df["is_overnight"] = (
        (hour >= OVERNIGHT_START) | (hour <= OVERNIGHT_END)
    ).astype(np.int8)

    logger.debug(
        "  settlement_period, is_weekend, is_holiday, "
        "is_morning_peak, is_evening_peak, is_overnight created."
    )
    return df


def add_generation_mix_ratios(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Compute generation mix ratios (fuel share of total generation).

    Raw MW values scale with total demand. Ratios strip out scale and reveal
    market structure: who is setting the marginal price. In UK electricity
    markets, gas CCGTs are typically the marginal generator — high gas_ratio
    strongly correlates with high prices. High wind_ratio signals cheap
    renewable surplus and downward price pressure.

    Each ratio is clipped to [0, 1]. Zero total generation is replaced with
    NaN to avoid division errors (those rows are caught by dropna).
    """
    logger.info("Creating %d generation mix ratio(s) ...", len(RATIO_SPECS))
    if "gen_total_mw" not in df.columns:
        raise KeyError("'gen_total_mw' is required for generation ratio calculations.")
    total = df["gen_total_mw"].replace(0, np.nan)
    for ratio_name, numerator_col in RATIO_SPECS:
        if numerator_col not in df.columns:
            raise KeyError(f"Column '{numerator_col}' not found for ratio '{ratio_name}'.")
        df[ratio_name] = (df[numerator_col] / total).clip(0.0, 1.0)
        logger.debug("  %-16s  = %s / gen_total_mw", ratio_name, numerator_col)
    return df


def add_supply_demand_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Add features capturing the physical supply-demand balance of the grid.

    supply_demand_ratio: demand_mw / gen_total_mw (lagged by 1 period).
      The most direct proxy for grid tightness. Ratio > 1 means demand
      exceeds metered domestic generation; the system must draw on
      interconnectors or storage to balance. High ratio = price spike risk.

    interconnector_net_mw: sum of all cross-border flows (lagged by 1).
      Positive = net import. Heavy imports signal domestic generation
      shortfall and upward price pressure.

    renewable_surplus: gen_renewable_total_mw minus 50% of demand (lagged).
      A proxy for excess renewable generation. When renewables exceed half
      of demand, negative prices become likely — directly observable in
      the March 2024 weekly price chart.

    All computed on lagged values to prevent look-ahead leakage.
    """
    logger.info("Creating supply-demand balance features ...")

    demand_s1    = df["demand_mw"].shift(1)
    gen_total_s1 = df["gen_total_mw"].shift(1).replace(0, np.nan)

    # Clip at 2.0: values above this are caused by Elexon BMRS API reporting
    # partial generation totals during 11:00-12:00 UTC (generators submit data
    # with variable delays). Confirmed in 17 rows across 2024-2025. Real grid
    # tightness never exceeds ~1.05 in practice — 2.0 is a conservative cap.
    df["supply_demand_ratio"] = (demand_s1 / gen_total_s1).clip(upper=2.0)
    logger.debug("  supply_demand_ratio  <- (demand_lag_1 / gen_total_lag_1).clip(2.0)")

    available_ic = [c for c in INTERCONNECTOR_COLS if c in df.columns]
    if available_ic:
        df["interconnector_net_mw"] = df[available_ic].shift(1).sum(axis=1)
        logger.debug(
            "  interconnector_net_mw  <- sum of %d columns: %s",
            len(available_ic), available_ic,
        )
    else:
        df["interconnector_net_mw"] = 0.0
        logger.warning("  No interconnector columns found — interconnector_net_mw=0.")

    if "gen_renewable_total_mw" in df.columns:
        df["renewable_surplus"] = (
            df["gen_renewable_total_mw"].shift(1) - 0.5 * demand_s1
        )
        logger.debug("  renewable_surplus  <- renewable_lag_1 - 0.5 x demand_lag_1")
    else:
        df["renewable_surplus"] = 0.0
        logger.warning("  'gen_renewable_total_mw' not found — renewable_surplus=0.")

    return df


def add_weather_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Engineer informative weather-derived features from raw weather columns.

    hdd (Heating Degree Days): max(0, 15.5 - temp_avg_c).
      UK industry standard measure of heating demand. Captures the non-linear
      threshold: below 15.5 degC every degree colder adds proportional heating
      load. Raw temperature treats 20C and 25C as meaningfully different when
      neither drives any heating — HDD correctly shows both as zero.

    wind_power_proxy: wind_speed_avg_kmh squared.
      Wind turbine power output scales roughly with the square of wind speed
      up to rated speed. Squaring linearises this relationship for the model
      and better represents the actual generation impact of wind speed changes.

    wind_surprise: wind_speed vs same time yesterday (shift 48).
      The unexpected component of wind generation. A sudden calm on a day
      forecast to be windy forces expensive gas plants online, lifting prices.

    temp_spread: temp_london_c minus temp_edinburgh_c.
      Regional temperature gradient affecting north-south power flows and
      regional balancing costs.

    carbon_forecast_error: carbon_actual minus carbon_forecast.
      When the grid runs dirtier than forecast, more gas than expected came
      online — a signal of tight supply and elevated prices.
    """
    logger.info("Creating weather-derived features ...")

    if "temp_avg_c" in df.columns:
        df["hdd"] = np.maximum(0.0, HDD_BASE_TEMP_C - df["temp_avg_c"])
        logger.debug("  hdd  <- max(0, %.1f - temp_avg_c)", HDD_BASE_TEMP_C)
    else:
        df["hdd"] = 0.0
        logger.warning("  'temp_avg_c' not found — hdd=0.")

    if "wind_speed_avg_kmh" in df.columns:
        df["wind_power_proxy"] = df["wind_speed_avg_kmh"] ** 2
        df["wind_surprise"]    = (
            df["wind_speed_avg_kmh"] - df["wind_speed_avg_kmh"].shift(PERIODS_PER_DAY)
        )
        logger.debug("  wind_power_proxy  <- wind_speed_avg_kmh^2")
        logger.debug("  wind_surprise     <- wind_speed - wind_speed_lag_48")
    else:
        df["wind_power_proxy"] = 0.0
        df["wind_surprise"]    = 0.0
        logger.warning("  'wind_speed_avg_kmh' not found — wind features=0.")

    if "temp_london_c" in df.columns and "temp_edinburgh_c" in df.columns:
        df["temp_spread"] = df["temp_london_c"] - df["temp_edinburgh_c"]
        logger.debug("  temp_spread  <- temp_london_c - temp_edinburgh_c")
    else:
        df["temp_spread"] = 0.0
        logger.warning("  City temperature columns not found — temp_spread=0.")

    if "carbon_actual" in df.columns and "carbon_forecast" in df.columns:
        df["carbon_forecast_error"] = df["carbon_actual"] - df["carbon_forecast"]
        logger.debug("  carbon_forecast_error  <- carbon_actual - carbon_forecast")
    else:
        df["carbon_forecast_error"] = 0.0
        logger.warning("  Carbon columns not found — carbon_forecast_error=0.")

    return df


def add_interaction_features(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Add cross-feature interaction terms capturing joint effects.

    wind_demand_interaction: wind_ratio x demand_lag_1.
      High wind + low demand = strong negative price pressure.
      High wind + high demand = prices may stay elevated despite renewables.
      The product captures this conditionality that additive features miss —
      it is the joint effect, not the sum of the individual effects.

    cold_evening_peak: hdd x is_evening_peak.
      Cold weather during the evening peak is when the grid is most stressed.
      Cold at 3am is largely irrelevant; cold at 6pm when everyone turns the
      heating on simultaneously can double peak prices. The interaction is
      non-additive by nature.

    Both prerequisite features must already exist in df.
    """
    logger.info("Creating interaction features ...")

    if "wind_ratio" in df.columns and "demand_lag_1" in df.columns:
        df["wind_demand_interaction"] = df["wind_ratio"] * df["demand_lag_1"]
        logger.debug("  wind_demand_interaction  <- wind_ratio x demand_lag_1")
    else:
        df["wind_demand_interaction"] = 0.0
        logger.warning("  wind_ratio or demand_lag_1 missing — wind_demand_interaction=0.")

    if "hdd" in df.columns and "is_evening_peak" in df.columns:
        df["cold_evening_peak"] = df["hdd"] * df["is_evening_peak"].astype(float)
        logger.debug("  cold_evening_peak  <- hdd x is_evening_peak")
    else:
        df["cold_evening_peak"] = 0.0
        logger.warning("  hdd or is_evening_peak missing — cold_evening_peak=0.")

    return df


def add_target_variable(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Define the supervised learning target: next settlement period's spot price.

    price_next = price_mid_gbp_mwh.shift(-1)

    Each row's target is the price observed in the following row — a T+30 min
    forecast horizon. Critically this uses the ORIGINAL unwinsorised price so
    the model learns to predict true market prices including extreme events.
    The final row will be NaN and is removed by drop_nan_rows().
    """
    logger.info("Creating target variable 'price_next' (T+30 min) ...")
    df["price_next"] = df["price_mid_gbp_mwh"].shift(-1)
    logger.debug(
        "  price_next <- shift(price_mid_gbp_mwh, -1)  [raw, NOT winsorised]"
    )
    return df


def drop_nan_rows(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Remove rows with any NaN values introduced by lag/rolling operations.

    Primary sources of NaN:
    - First 336 rows: burn-in for price_lag_336 (longest lag = 1 week).
    - Last row: price_next is NaN (no subsequent row).
    - wind_surprise: first 48 rows also NaN from shift(48) on wind speed.
      (These overlap with the lag_336 burn-in and add no additional loss.)
    """
    n_before = len(df)

    null_counts = df.isnull().sum()
    null_cols   = null_counts[null_counts > 0]
    if not null_cols.empty:
        logger.debug("Columns with NaN before dropna:")
        for col, cnt in null_cols.items():
            logger.debug("  %-35s  %d NaN(s)", col, cnt)

    df = df.dropna()
    n_dropped = n_before - len(df)
    logger.info(
        "Dropped %d NaN row(s) (%.2f%% of dataset).  Remaining: %d rows.",
        n_dropped,
        100.0 * n_dropped / n_before,
        len(df),
    )
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_features(input_path: Path, output_path: Path, logger: logging.Logger) -> None:
    """
    End-to-end feature engineering pipeline.

    Step ordering is significant — dependencies are:
      winsorise FIRST (lags consume it)
      rolling BEFORE position features (position needs rolling_mean_24h)
      rolling BEFORE demand context (context needs demand_rolling_mean_24h)
      ratios BEFORE interactions (interactions consume wind_ratio)
      target LAST among engineered columns
      drop_nan AFTER all columns exist
      drop price_winsorised AFTER drop_nan (it carries NaN rows too)

    Parameters
    ----------
    input_path  : Path to master_clean.parquet.
    output_path : Destination path for features.parquet.
    logger      : Module logger.
    """
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    logger.info("Loading input dataset from '%s' ...", input_path)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    df = pd.read_parquet(input_path)
    logger.info(
        "Loaded %d rows x %d columns.  Date range: %s -> %s.",
        len(df),
        df.shape[1],
        df.index.min() if isinstance(df.index, pd.DatetimeIndex) else "?",
        df.index.max() if isinstance(df.index, pd.DatetimeIndex) else "?",
    )

    # ------------------------------------------------------------------
    # 2. Validate and normalise index
    # ------------------------------------------------------------------
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.info(
            "Index is not DatetimeIndex — converting via 'settlement_period_start'."
        )
        if "settlement_period_start" in df.columns:
            df["settlement_period_start"] = pd.to_datetime(
                df["settlement_period_start"], utc=True
            )
            df = df.set_index("settlement_period_start")
        else:
            logger.error(
                "Cannot determine timestamp index. "
                "Expected DatetimeIndex or 'settlement_period_start' column."
            )
            sys.exit(1)

    if df.index.tz is None:
        logger.warning("Index has no timezone — localising to UTC.")
        df.index = df.index.tz_localize("UTC")

    df = df.sort_index()

    if not df.index.is_monotonic_increasing:
        logger.error("Index not monotonically increasing after sort — data integrity issue.")
        sys.exit(1)

    n_price_nulls = df["price_mid_gbp_mwh"].isna().sum()
    if n_price_nulls > 0:
        logger.warning(
            "%d NaN(s) in 'price_mid_gbp_mwh' before feature engineering.", n_price_nulls
        )

    # ------------------------------------------------------------------
    # 3. Feature engineering — execution order matters, see docstring
    # ------------------------------------------------------------------
    df = add_winsorised_price(df, logger)
    df = add_lag_features(df, logger)
    df = add_rolling_features(df, logger)
    df = add_price_momentum(df, logger)
    df = add_price_position_features(df, logger)
    df = add_demand_context_features(df, logger)
    df = add_cyclical_time_features(df, logger)
    df = add_calendar_flags(df, logger)
    df = add_generation_mix_ratios(df, logger)
    df = add_supply_demand_features(df, logger)
    df = add_weather_features(df, logger)
    df = add_interaction_features(df, logger)
    df = add_target_variable(df, logger)

    # ------------------------------------------------------------------
    # 4. Drop NaN rows (burn-in + last row)
    # ------------------------------------------------------------------
    df = drop_nan_rows(df, logger)

    # ------------------------------------------------------------------
    # 5a. Drop leakage-risk raw columns
    # ------------------------------------------------------------------
    cols_to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info("Dropped %d leakage-risk column(s): %s", len(cols_to_drop), cols_to_drop)

    # ------------------------------------------------------------------
    # 5b. Drop intermediate winsorised column (construction artefact only)
    # ------------------------------------------------------------------
    if "price_winsorised" in df.columns:
        df = df.drop(columns=["price_winsorised"])
        logger.debug("Dropped intermediate 'price_winsorised' column.")

    # ------------------------------------------------------------------
    # 6. Final validation
    # ------------------------------------------------------------------
    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls > 0:
        bad_cols = df.columns[df.isnull().any()].tolist()
        logger.error(
            "%d NaN(s) remain after dropna in: %s", remaining_nulls, bad_cols
        )
        sys.exit(1)

    if not df.index.is_monotonic_increasing:
        logger.error("Output index not monotonically increasing — aborting.")
        sys.exit(1)

    # Count engineered vs original columns
    original_cols = set(pd.read_parquet(input_path).columns)
    new_cols = [c for c in df.columns if c not in original_cols]
    logger.info(
        "Output: %d rows x %d columns  "
        "(%d engineered, %d original, target=price_next).  Zero NaNs confirmed.",
        len(df), df.shape[1], len(new_cols),
        len([c for c in df.columns if c in original_cols]),
    )
    logger.debug("Engineered columns (%d): %s", len(new_cols), new_cols)

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=True, engine="pyarrow", compression="snappy")

    elapsed = time.perf_counter() - t_start
    logger.info(
        "Saved -> '%s'  (%.2f MB, %.2f s elapsed).",
        output_path,
        output_path.stat().st_size / 1_048_576,
        elapsed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "UK Electricity Price Forecasting — Feature Engineering Pipeline.\n\n"
            "Reads master_clean.parquet, engineers a comprehensive feature matrix\n"
            "(lags, rolling stats, momentum, position, demand context, calendar\n"
            "flags, supply-demand balance, weather, interactions) and writes\n"
            "features.parquet ready for XGBoost model training."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Data/processed/master_clean.parquet"),
        metavar="PATH",
        help="Path to master_clean.parquet (default: Data/processed/master_clean.parquet).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/processed/features.parquet"),
        metavar="PATH",
        help="Destination for features.parquet (default: Data/processed/features.parquet).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = _parse_args(argv)
    logger = _configure_logging(args.log_level)

    logger.info("=" * 65)
    logger.info("UK Electricity Price Forecasting — Feature Engineering")
    logger.info("=" * 65)
    logger.info("Input  : %s", args.input)
    logger.info("Output : %s", args.output)

    build_features(
        input_path=args.input,
        output_path=args.output,
        logger=logger,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()