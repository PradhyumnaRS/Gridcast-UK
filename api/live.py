"""
live.py — Live data fetching for the UK Electricity Price Forecasting API.

Fetches the current settlement period's raw data from three sources:
  - Elexon BMRS API     — prices, demand, generation mix, imbalance
  - Open-Meteo API      — weather for London, Birmingham, Edinburgh
  - Carbon Intensity API — carbon forecast, actual, and index label

All functions return a flat dictionary of raw field values keyed by the
same column names used in features.parquet. This ensures the live feature
vector is constructed identically to the training data.

If any individual API call fails, the caller (api/main.py) falls back
to the latest available row in features.parquet.
"""

from __future__ import annotations

import logging
from datetime import datetime, date, timedelta, timezone
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

_SESSION = requests.Session()
_TIMEOUT = 10  # seconds — fast timeout for live inference path


def _get(url: str, params: dict | None = None) -> Any:
    """Make a GET request and return parsed JSON.

    Parameters
    ----------
    url:
        Full URL to request.
    params:
        Optional query parameters.

    Returns
    -------
    Any
        Parsed JSON response body (may be a list or dict depending on endpoint).

    Raises
    ------
    RuntimeError
        If the request fails or returns a non-200 status.
    """
    try:
        resp = _SESSION.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"HTTP request failed: {url} — {exc}") from exc


# ---------------------------------------------------------------------------
# Current settlement period helper
# ---------------------------------------------------------------------------


def current_settlement_period() -> datetime:
    """Return the start of the current UTC half-hourly settlement period.

    Rounds down to the nearest 30-minute boundary.

    Returns
    -------
    datetime
        UTC-aware datetime for the current settlement period start.
    """
    now = datetime.now(tz=timezone.utc)
    minute = 0 if now.minute < 30 else 30
    return now.replace(minute=minute, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Elexon BMRS
# ---------------------------------------------------------------------------

_ELEXON_BASE = "https://data.elexon.co.uk/bmrs/api/v1"


def fetch_elexon_live() -> dict[str, Any]:
    """Fetch the latest generation mix, demand, and imbalance from Elexon BMRS.

    The generation/outturn/summary endpoint returns a top-level list of
    settlement period objects, each containing a ``data`` list of
    ``{"fuelType": str, "generation": float}`` items::

        [
          {
            "startTime": "2026-03-21T06:30:00Z",
            "settlementPeriod": 14,
            "data": [
              {"fuelType": "CCGT", "generation": 9036},
              {"fuelType": "WIND", "generation": 1517},
              ...
            ]
          },
          ...
        ]

    Returns
    -------
    dict[str, Any]
        Raw Elexon values keyed by features.parquet column names.

    Raises
    ------
    RuntimeError
        If the Elexon API is unreachable or returns unexpected data.
    """
    logger.debug("Fetching live Elexon generation mix...")

    # Response is a top-level LIST of settlement period objects
    url = f"{_ELEXON_BASE}/generation/outturn/summary"
    response = _get(url, params={"format": "json"})

    if not isinstance(response, list) or len(response) == 0:
        raise RuntimeError(
            f"Elexon generation/outturn/summary returned unexpected structure: {type(response)}"
        )

    # Most recent settlement period is first in the list
    latest = response[0]
    fuel_rows = latest.get("data", [])
    if not fuel_rows:
        raise RuntimeError("Elexon generation data is empty for latest settlement period.")

    logger.debug(
        "Elexon latest period: %s (SP %s), %d fuel types",
        latest.get("startTime"), latest.get("settlementPeriod"), len(fuel_rows),
    )

    # Map Elexon fuelType strings to features.parquet column names
    fuel_map = {
        "BIOMASS":  "gen_biomass_mw",
        "CCGT":     "gen_ccgt_mw",
        "COAL":     "gen_coal_mw",
        "INTELEC":  "gen_intelec_mw",
        "INTEW":    "gen_intew_mw",
        "INTFR":    "gen_intfr_mw",
        "INTGRNL":  "gen_intgrnl_mw",
        "INTIFA2":  "gen_intifa2_mw",
        "INTIRL":   "gen_intirl_mw",
        "INTNED":   "gen_intned_mw",
        "INTNEM":   "gen_intnem_mw",
        "INTNSL":   "gen_intnsl_mw",
        "INTVKL":   "gen_intvkl_mw",
        "NPSHYD":   "gen_npshyd_mw",
        "NUCLEAR":  "gen_nuclear_mw",
        "OCGT":     "gen_ocgt_mw",
        "OIL":      "gen_oil_mw",
        "OTHER":    "gen_other_mw",
        "PS":       "gen_ps_mw",
        "WIND":     "gen_wind_mw",
    }

    # Convert list of {"fuelType": ..., "generation": ...} into a flat lookup dict
    fuel_lookup = {row["fuelType"]: row.get("generation", 0.0) for row in fuel_rows}
    logger.debug("Fuel types received from Elexon: %s", sorted(fuel_lookup.keys()))

    result: dict[str, Any] = {}
    for elexon_key, col_name in fuel_map.items():
        result[col_name] = float(fuel_lookup.get(elexon_key, 0.0) or 0.0)

    # Derived generation totals
    renewable_fuels = {"gen_biomass_mw", "gen_wind_mw", "gen_npshyd_mw", "gen_intgrnl_mw"}
    fossil_fuels = {"gen_ccgt_mw", "gen_coal_mw", "gen_ocgt_mw", "gen_oil_mw"}

    result["gen_renewable_total_mw"] = sum(result.get(f, 0.0) for f in renewable_fuels)
    result["gen_fossil_total_mw"] = sum(result.get(f, 0.0) for f in fossil_fuels)
    result["gen_total_mw"] = sum(result.get(c, 0.0) for c in fuel_map.values())
    result["gen_renewable_fraction"] = (
        result["gen_renewable_total_mw"] / result["gen_total_mw"]
        if result["gen_total_mw"] > 0 else 0.0
    )

    # Interconnector net flow
    interconnector_cols = [
        "gen_intelec_mw", "gen_intew_mw", "gen_intfr_mw", "gen_intgrnl_mw",
        "gen_intifa2_mw", "gen_intirl_mw", "gen_intned_mw", "gen_intnem_mw",
        "gen_intnsl_mw", "gen_intvkl_mw",
    ]
    result["interconnector_net_mw"] = sum(result.get(c, 0.0) for c in interconnector_cols)

    # Demand — also returns a list
    try:
        demand_url = f"{_ELEXON_BASE}/demand/outturn/summary"
        demand_response = _get(demand_url, params={"format": "json"})
        if isinstance(demand_response, list) and len(demand_response) > 0:
            result["demand_mw"] = float(demand_response[0].get("demand", 0.0) or 0.0)
        elif isinstance(demand_response, dict):
            demand_rows = demand_response.get("data", [])
            result["demand_mw"] = float(demand_rows[0].get("demand", 0.0)) if demand_rows else 0.0
        else:
            result["demand_mw"] = result["gen_total_mw"]
    except RuntimeError:
        logger.warning("Could not fetch live demand — using gen_total_mw as proxy.")
        result["demand_mw"] = result["gen_total_mw"]

    # Net imbalance volume — best effort
    try:
        imb_url = f"{_ELEXON_BASE}/balancing/settlement/system-prices/latest"
        imb_response = _get(imb_url, params={"format": "json"})
        if isinstance(imb_response, list) and len(imb_response) > 0:
            imb = imb_response[0]
        elif isinstance(imb_response, dict):
            imb_rows = imb_response.get("data", [])
            imb = imb_rows[0] if imb_rows else {}
        else:
            imb = {}
        result["net_imbalance_vol_mwh"] = float(imb.get("netImbalanceVolume", 0.0) or 0.0)
        result["price_ssp_gbp_mwh"] = float(imb.get("systemSellPrice", 0.0) or 0.0)
        result["price_sbp_gbp_mwh"] = float(imb.get("systemBuyPrice", 0.0) or 0.0)
    except RuntimeError:
        logger.warning("Could not fetch live imbalance data — defaulting to 0.")
        result["net_imbalance_vol_mwh"] = 0.0
        result["price_ssp_gbp_mwh"] = 0.0
        result["price_sbp_gbp_mwh"] = 0.0

    logger.info(
        "Elexon live fetch complete — gen_total: %.0f MW, demand: %.0f MW, wind: %.0f MW",
        result["gen_total_mw"], result["demand_mw"], result["gen_wind_mw"],
    )
    return result


# ---------------------------------------------------------------------------
# Recent price history for lag feature construction
# ---------------------------------------------------------------------------


def fetch_recent_prices(n_days: int = 8) -> pd.Series:
    """Fetch the last ``n_days`` of half-hourly settlement prices from Elexon.

    Uses the /balancing/settlement/system-prices/{settlementDate} endpoint
    which returns all 48 settlement periods for a given date. Fetches the
    last ``n_days`` days to ensure we have at least 336 periods (1 week)
    for price_lag_336 construction.

    The mid price is computed as the mean of systemSellPrice and
    systemBuyPrice, matching the ``price_mid_gbp_mwh`` column used during
    training.

    Parameters
    ----------
    n_days:
        Number of days to fetch (default 8 — gives ~384 periods, enough
        for price_lag_336 with some buffer).

    Returns
    -------
    pd.Series
        Half-hourly mid prices indexed by UTC DatetimeIndex, sorted
        chronologically. Index name is ``settlement_period_start``.

    Raises
    ------
    RuntimeError
        If no price data can be fetched for any of the requested dates.
    """
    logger.info("Fetching recent %d days of settlement prices from Elexon...", n_days)

    all_records: list[dict] = []
    today = date.today()

    for i in range(n_days, 0, -1):
        target_date = (today - timedelta(days=i)).isoformat()
        url = f"{_ELEXON_BASE}/balancing/settlement/system-prices/{target_date}"
        try:
            response = _get(url, params={"format": "json"})
            rows = response.get("data", []) if isinstance(response, dict) else response
            if isinstance(rows, dict):
                rows = rows.get("data", [])
            all_records.extend(rows)
            logger.debug("Fetched %d periods for %s", len(rows), target_date)
        except RuntimeError as exc:
            logger.warning("Could not fetch prices for %s: %s", target_date, exc)

    if not all_records:
        raise RuntimeError("Could not fetch any recent settlement prices from Elexon.")

    df = pd.DataFrame(all_records)

    # Parse timestamp
    df["settlement_period_start"] = pd.to_datetime(df["startTime"], utc=True)

    # Compute mid price — matches price_mid_gbp_mwh from training
    df["price_mid_gbp_mwh"] = (
        df["systemSellPrice"].astype(float) + df["systemBuyPrice"].astype(float)
    ) / 2

    # Also store net imbalance volume for the lookback window
    df["net_imbalance_vol_mwh"] = df["netImbalanceVolume"].astype(float)

    # Deduplicate and sort
    df = df.drop_duplicates(subset=["settlement_period_start"]).sort_values("settlement_period_start")
    df = df.set_index("settlement_period_start")

    price_series = df["price_mid_gbp_mwh"]

    logger.info(
        "Recent prices fetched: %d periods (%s → %s), mean £%.2f/MWh",
        len(price_series),
        price_series.index.min().date(),
        price_series.index.max().date(),
        price_series.mean(),
    )
    return price_series


# ---------------------------------------------------------------------------
# Open-Meteo (forecast API for live data)
# ---------------------------------------------------------------------------

_WEATHER_BASE = "https://api.open-meteo.com/v1/forecast"

_LOCATIONS = {
    "london":     {"latitude": 51.51, "longitude": -0.13},
    "birmingham": {"latitude": 52.48, "longitude": -1.90},
    "edinburgh":  {"latitude": 55.95, "longitude": -3.19},
}


def fetch_weather_live() -> dict[str, Any]:
    """Fetch current weather conditions for London, Birmingham, and Edinburgh.

    Uses the Open-Meteo forecast API (no key required) rather than the
    archive API used during training. Returns the current hour's values.

    Returns
    -------
    dict[str, Any]
        Weather values keyed by features.parquet column names.

    Raises
    ------
    RuntimeError
        If weather data cannot be fetched for any location.
    """
    logger.debug("Fetching live weather from Open-Meteo...")

    result: dict[str, Any] = {}
    now_hour = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:00")

    for name, coords in _LOCATIONS.items():
        params = {
            **coords,
            "hourly": "temperature_2m,wind_speed_10m,cloud_cover",
            "timezone": "UTC",
            "wind_speed_unit": "kmh",
            "forecast_days": 1,
        }
        data = _get(_WEATHER_BASE, params=params)
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        try:
            idx = times.index(now_hour)
        except ValueError:
            idx = -1
            logger.warning(
                "Current hour %s not found in Open-Meteo response for %s — using last available.",
                now_hour, name,
            )

        result[f"temp_{name}_c"] = float(hourly["temperature_2m"][idx] or 0.0)
        result[f"wind_speed_{name}_kmh"] = float(hourly["wind_speed_10m"][idx] or 0.0)
        result[f"cloud_cover_{name}_pct"] = float(hourly["cloud_cover"][idx] or 0.0)

    # National averages
    result["temp_avg_c"] = (
        result["temp_london_c"] + result["temp_birmingham_c"] + result["temp_edinburgh_c"]
    ) / 3
    result["wind_speed_avg_kmh"] = (
        result["wind_speed_london_kmh"] + result["wind_speed_birmingham_kmh"] + result["wind_speed_edinburgh_kmh"]
    ) / 3
    result["cloud_cover_avg_pct"] = (
        result["cloud_cover_london_pct"] + result["cloud_cover_birmingham_pct"] + result["cloud_cover_edinburgh_pct"]
    ) / 3

    logger.info(
        "Weather live fetch complete — temp_avg: %.1f°C, wind_avg: %.1f km/h, cloud_avg: %.0f%%",
        result["temp_avg_c"], result["wind_speed_avg_kmh"], result["cloud_cover_avg_pct"],
    )
    return result


# ---------------------------------------------------------------------------
# Carbon Intensity API
# ---------------------------------------------------------------------------

_CARBON_BASE = "https://api.carbonintensity.org.uk"


def fetch_carbon_live() -> dict[str, Any]:
    """Fetch the current half-hour's carbon intensity data.

    Uses the /intensity endpoint which returns the current period.
    The carbon_index string label is dropped — carbon_forecast and
    carbon_actual (numeric) are used instead, matching training data.

    Returns
    -------
    dict[str, Any]
        Carbon values keyed by features.parquet column names.

    Raises
    ------
    RuntimeError
        If the Carbon Intensity API is unreachable.
    """
    logger.debug("Fetching live carbon intensity...")

    data = _get(f"{_CARBON_BASE}/intensity")
    rows = data.get("data", [])

    if not rows:
        raise RuntimeError("Carbon Intensity API returned no data.")

    intensity = rows[0].get("intensity", {})

    result: dict[str, Any] = {
        "carbon_forecast": float(intensity.get("forecast", 0.0) or 0.0),
        "carbon_actual": float(intensity.get("actual") or intensity.get("forecast") or 0.0),
        # carbon_index (string) deliberately excluded — dropped during training
    }

    # Carbon forecast error — engineered feature matching feature_engineering.py
    result["carbon_forecast_error"] = result["carbon_actual"] - result["carbon_forecast"]

    logger.info(
        "Carbon live fetch complete — forecast: %d, actual: %d gCO2/kWh",
        result["carbon_forecast"], result["carbon_actual"],
    )
    return result


# ---------------------------------------------------------------------------
# Combined live snapshot
# ---------------------------------------------------------------------------


def fetch_live_snapshot() -> tuple[dict[str, Any], pd.Series]:
    """Fetch a complete live data snapshot from all three APIs plus recent prices.

    Returns both the current period's raw feature values AND a price history
    Series for the last 8 days. The price history is used by features.py to
    compute lag and rolling features from real recent prices rather than
    stale parquet data.

    Returns
    -------
    tuple[dict[str, Any], pd.Series]
        - snapshot: flat dict of current period raw values
        - recent_prices: pd.Series of half-hourly mid prices (last 8 days),
          indexed by UTC DatetimeIndex

    Raises
    ------
    RuntimeError
        If any of the core API calls fail.
    """
    logger.info("Fetching live snapshot from all APIs...")

    elexon = fetch_elexon_live()
    weather = fetch_weather_live()
    carbon = fetch_carbon_live()
    recent_prices = fetch_recent_prices(n_days=8)

    snapshot = {**elexon, **weather, **carbon}
    logger.info("Live snapshot complete: %d raw fields, %d recent price periods",
                len(snapshot), len(recent_prices))
    return snapshot, recent_prices
