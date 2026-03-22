"""
features.py — Real-time feature construction for the UK Electricity Price Forecasting API.

For live inference, the 89 model features cannot all be observed directly —
many are lagged prices, rolling statistics, and engineered interactions that
require historical context. This module:

  1. Uses real recent prices fetched from Elexon (last 8 days) as the
     price lookback window — ensuring lag features are always fresh
  2. Appends the live raw data snapshot as the current row
  3. Computes all lagged and rolling features for that new row
  4. Returns the 89-feature vector ready for model.predict()

For historical inference, it simply looks up the pre-computed feature row
from features.parquet by datetime index.

The feature construction logic mirrors feature_engineering.py exactly,
ensuring train/serve consistency.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — must match feature_engineering.py exactly
# ---------------------------------------------------------------------------

TARGET: str = "price_next"

# Price lag periods (in half-hourly steps)
PRICE_LAGS: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 48, 336]

# Demand lag periods
DEMAND_LAGS: list[int] = [1, 48]

# Rolling windows (in half-hourly periods)
ROLLING_MEAN_1H: int = 2
ROLLING_MEAN_24H: int = 48
ROLLING_STD_24H: int = 48

# Winsorisation bounds (must match training)
PRICE_LOWER_PCT: float = 0.001
PRICE_UPPER_PCT: float = 0.999

# UK public holidays reference
UK_HOLIDAYS_2024_2026 = pd.to_datetime([
    "2024-01-01", "2024-03-29", "2024-04-01", "2024-05-06",
    "2024-05-27", "2024-08-26", "2024-12-25", "2024-12-26",
    "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-05",
    "2025-05-26", "2025-08-25", "2025-12-25", "2025-12-26",
    "2026-01-01", "2026-04-03", "2026-04-06", "2026-05-04",
    "2026-05-25", "2026-08-31", "2026-12-25", "2026-12-28",
])


# ---------------------------------------------------------------------------
# Parquet loading
# ---------------------------------------------------------------------------


def _drop_non_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Drop string/ArrowString columns — mirrors train_model.py cleaning."""
    str_cols = [
        c for c in df.columns
        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object
    ]
    if str_cols:
        df = df.drop(columns=str_cols)
    return df


def load_parquet(path: Path) -> pd.DataFrame:
    """Load features.parquet with DatetimeIndex (UTC).

    Parameters
    ----------
    path:
        Path to features.parquet.

    Returns
    -------
    pd.DataFrame
        Cleaned feature DataFrame indexed by UTC DatetimeIndex.
    """
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("features.parquet must have a DatetimeIndex.")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = _drop_non_numeric(df)
    logger.debug("Parquet loaded: %d rows × %d columns", *df.shape)
    return df


# ---------------------------------------------------------------------------
# Historical lookup
# ---------------------------------------------------------------------------


def get_historical_features(
    df: pd.DataFrame, dt: pd.Timestamp
) -> pd.DataFrame:
    """Look up a pre-computed feature row from parquet by datetime.

    Parameters
    ----------
    df:
        Full features.parquet DataFrame.
    dt:
        Settlement period datetime (UTC-aware).

    Returns
    -------
    pd.DataFrame
        Single-row feature DataFrame (target column excluded).

    Raises
    ------
    KeyError
        If ``dt`` is not in the parquet index.
    """
    if dt not in df.index:
        nearest = df.index[df.index.get_indexer([dt], method="nearest")]
        raise KeyError(
            f"Datetime {dt.isoformat()} not in parquet. "
            f"Nearest available: {[t.isoformat() for t in nearest]}"
        )
    row = df.loc[[dt]]
    if TARGET in row.columns:
        row = row.drop(columns=[TARGET])
    return row


# ---------------------------------------------------------------------------
# Live feature construction
# ---------------------------------------------------------------------------


def _winsorise_series(s: pd.Series) -> pd.Series:
    """Winsorise a price series to p0.1%–p99.9% — matches training."""
    lower = s.quantile(PRICE_LOWER_PCT)
    upper = s.quantile(PRICE_UPPER_PCT)
    return s.clip(lower, upper)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time encodings and calendar flags."""
    idx = df.index
    hour = idx.hour + idx.minute / 60.0
    dow = idx.dayofweek
    month = idx.month

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    df["is_weekend"] = (dow >= 5).astype(int)
    df["is_holiday"] = idx.normalize().isin(UK_HOLIDAYS_2024_2026).astype(int)
    df["is_morning_peak"] = ((idx.hour >= 7) & (idx.hour < 9)).astype(int)
    df["is_evening_peak"] = ((idx.hour >= 16) & (idx.hour < 19)).astype(int)
    df["is_overnight"] = ((idx.hour >= 23) | (idx.hour < 5)).astype(int)

    return df


def build_live_features(
    parquet_df: pd.DataFrame,
    snapshot: dict[str, Any],
    current_dt: datetime,
    recent_prices: pd.Series | None = None,
) -> pd.DataFrame:
    """Construct the 89-feature vector for the current settlement period.

    Uses real recent prices from Elexon (if provided) as the price lookback
    window for computing lag and rolling features. Falls back to parquet
    price history if recent_prices is None.

    Parameters
    ----------
    parquet_df:
        Full features.parquet DataFrame (used for non-price feature lookback
        and as fallback price history).
    snapshot:
        Raw live data from ``fetch_live_snapshot()`` in live.py.
    current_dt:
        Current settlement period start (UTC-aware datetime).
    recent_prices:
        Optional pd.Series of real recent half-hourly mid prices from Elexon
        (last 8 days), indexed by UTC DatetimeIndex. If provided, these
        replace the stale parquet prices for lag/rolling feature construction.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with 89 feature columns, ready for model.predict().
    """
    current_ts = pd.Timestamp(current_dt)
    if current_ts.tzinfo is None:
        current_ts = current_ts.tz_localize("UTC")

    # ── 1. Build price lookback series ────────────────────────────────────
    # Use real recent prices from Elexon if available, else fall back to parquet
    if recent_prices is not None and len(recent_prices) >= 10:
        price_lookback = recent_prices.copy()
        logger.info(
            "Using %d real recent prices for lag construction (last: %s, £%.2f/MWh)",
            len(price_lookback),
            price_lookback.index.max().isoformat(),
            price_lookback.iloc[-1],
        )
    else:
        logger.warning("No recent prices provided — falling back to parquet price history.")
        price_lookback = parquet_df["price_lag_1"].tail(400) if "price_lag_1" in parquet_df.columns else pd.Series(dtype=float)

    # Last known price = most recent period in the lookback
    last_known_price = float(price_lookback.iloc[-1]) if len(price_lookback) > 0 else 80.0

    # ── 2. Extract non-price lookback window from parquet ─────────────────
    # We still need parquet for non-price features (demand, generation history)
    # to compute demand lags and rolling stats
    lookback = parquet_df.tail(400).copy()

    # ── 3. Build the new live row ─────────────────────────────────────────
    new_row: dict[str, Any] = {
        **snapshot,
        "price_mid_gbp_mwh": last_known_price,
    }

    # Build a single-row DataFrame at the current timestamp
    live_df = pd.DataFrame([new_row], index=pd.DatetimeIndex([current_ts], tz="UTC"))
    live_df.index.name = lookback.index.name

    # Add settlement_period as int8 to match parquet dtype (integer 1–48)
    live_df["settlement_period"] = pd.array(
        [int(current_ts.hour * 2 + current_ts.minute // 30 + 1)], dtype="int8"
    )

    # ── 4. Concatenate with lookback for non-price feature computation ────
    shared_cols = [c for c in lookback.columns if c in live_df.columns and c != TARGET]
    live_only_cols = [c for c in live_df.columns if c not in lookback.columns and c != TARGET]
    combined = pd.concat([lookback[shared_cols], live_df[shared_cols + live_only_cols]], axis=0)
    combined = combined.sort_index()

    # ── 5. Inject real recent prices into combined for lag computation ─────
    # Replace the price_mid_gbp_mwh column with real Elexon prices where available
    if recent_prices is not None and len(recent_prices) >= 10:
        # Reindex recent prices onto combined's index, forward-fill gaps
        combined["price_mid_gbp_mwh"] = recent_prices.reindex(
            combined.index, method="ffill"
        )
        # For the current live row, use the last known price
        combined.loc[current_ts, "price_mid_gbp_mwh"] = last_known_price

    # ── 6. Winsorise price column for lag features ────────────────────────
    combined["price_winsorised"] = _winsorise_series(
        combined["price_mid_gbp_mwh"].fillna(last_known_price)
    )

    # ── 7. Price lags ─────────────────────────────────────────────────────
    for lag in PRICE_LAGS:
        combined[f"price_lag_{lag}"] = combined["price_winsorised"].shift(lag)

    # ── 8. Demand lags ────────────────────────────────────────────────────
    if "demand_mw" in combined.columns:
        for lag in DEMAND_LAGS:
            combined[f"demand_lag_{lag}"] = combined["demand_mw"].shift(lag)

    # ── 9. Rolling statistics ─────────────────────────────────────────────
    price_w = combined["price_winsorised"]
    combined["rolling_mean_1h"] = price_w.rolling(ROLLING_MEAN_1H, min_periods=1).mean()
    combined["rolling_mean_24h"] = price_w.rolling(ROLLING_MEAN_24H, min_periods=1).mean()
    combined["rolling_std_24h"] = price_w.rolling(ROLLING_STD_24H, min_periods=1).std().fillna(0)

    if "demand_mw" in combined.columns:
        demand = combined["demand_mw"]
        combined["demand_rolling_mean_1h"] = demand.rolling(ROLLING_MEAN_1H, min_periods=1).mean()
        combined["demand_rolling_mean_24h"] = demand.rolling(ROLLING_MEAN_24H, min_periods=1).mean()

    # ── 10. Price dynamics ────────────────────────────────────────────────
    combined["price_momentum_1p"] = price_w.diff(1).fillna(0)
    combined["price_momentum_4p"] = price_w.diff(4).fillna(0)
    combined["price_vs_daily_mean"] = price_w - combined["rolling_mean_24h"]
    combined["daily_price_range"] = (
        price_w.rolling(ROLLING_MEAN_24H, min_periods=1).max()
        - price_w.rolling(ROLLING_MEAN_24H, min_periods=1).min()
    )

    if "demand_mw" in combined.columns:
        demand = combined["demand_mw"]
        combined["demand_vs_daily_mean"] = demand - combined["demand_rolling_mean_24h"]
        combined["demand_momentum"] = demand.diff(1).fillna(0)

    # ── 11. Generation mix ratios ─────────────────────────────────────────
    total = combined.get("gen_total_mw", pd.Series(1.0, index=combined.index))
    total = total.replace(0, 1)

    for fuel, col in [
        ("gen_wind_mw", "wind_ratio"),
        ("gen_ccgt_mw", "gas_ratio"),
        ("gen_nuclear_mw", "nuclear_ratio"),
    ]:
        if fuel in combined.columns:
            combined[col] = combined[fuel] / total

    # ── 12. Supply-demand balance ─────────────────────────────────────────
    if "demand_mw" in combined.columns and "gen_total_mw" in combined.columns:
        combined["supply_demand_ratio"] = (
            combined["demand_mw"] / combined["gen_total_mw"].replace(0, 1)
        ).clip(upper=2.0)

    if "gen_renewable_total_mw" in combined.columns and "demand_mw" in combined.columns:
        combined["renewable_surplus"] = (
            combined["gen_renewable_total_mw"] - 0.5 * combined["demand_mw"]
        )

    # ── 13. Weather-derived features ──────────────────────────────────────
    if "temp_avg_c" in combined.columns:
        combined["hdd"] = (15.5 - combined["temp_avg_c"]).clip(lower=0)

    if "wind_speed_avg_kmh" in combined.columns:
        combined["wind_power_proxy"] = combined["wind_speed_avg_kmh"] ** 2

    if "wind_speed_edinburgh_kmh" in combined.columns:
        wind_48 = combined["wind_speed_edinburgh_kmh"].shift(48)
        combined["wind_surprise"] = combined["wind_speed_edinburgh_kmh"] - wind_48.fillna(
            combined["wind_speed_edinburgh_kmh"]
        )

    if "temp_london_c" in combined.columns and "temp_edinburgh_c" in combined.columns:
        combined["temp_spread"] = combined["temp_london_c"] - combined["temp_edinburgh_c"]

    if "carbon_actual" in combined.columns and "carbon_forecast" in combined.columns:
        combined["carbon_forecast_error"] = (
            combined["carbon_actual"] - combined["carbon_forecast"]
        )

    # ── 14. Time features (must run before interaction features) ──────────
    combined = _add_time_features(combined)

    # ── 15. Interaction features ──────────────────────────────────────────
    if "wind_ratio" in combined.columns and "demand_lag_1" in combined.columns:
        combined["wind_demand_interaction"] = combined["wind_ratio"] * combined["demand_lag_1"]

    if "hdd" in combined.columns and "is_evening_peak" in combined.columns:
        combined["cold_evening_peak"] = combined["hdd"] * combined["is_evening_peak"]

    # ── 16. Extract only the live row ─────────────────────────────────────
    live_row = combined.loc[[current_ts]].copy()

    # Drop columns not used as model features
    drop_cols = [
        TARGET, "price_mid_gbp_mwh", "price_winsorised",
        "price_ssp_gbp_mwh", "price_sbp_gbp_mwh", "settlement_date",
    ]
    live_row = live_row.drop(columns=[c for c in drop_cols if c in live_row.columns])

    logger.info(
        "Live feature vector built: %d features for %s (price_lag_1=£%.2f)",
        len(live_row.columns),
        current_ts.isoformat(),
        float(live_row["price_lag_1"].iloc[0]) if "price_lag_1" in live_row.columns else 0,
    )
    return live_row


# ---------------------------------------------------------------------------
# Fallback — latest parquet row
# ---------------------------------------------------------------------------


def get_latest_parquet_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Return the most recent feature row from parquet as a live fallback.

    Parameters
    ----------
    df:
        Full features.parquet DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, pd.Timestamp]
        ``(feature_row, timestamp)`` where ``timestamp`` is the period's datetime.
    """
    latest_ts = df.index.max()
    row = df.loc[[latest_ts]]
    if TARGET in row.columns:
        row = row.drop(columns=[TARGET])
    logger.warning(
        "Using parquet fallback — latest row: %s (may be stale).", latest_ts.isoformat()
    )
    return row, latest_ts
