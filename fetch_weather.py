"""
fetch_weather.py
================
Fetches 2 years of hourly weather data from the Open-Meteo Archive API
for three UK locations, resamples to half-hourly, and saves to parquet.

API base : https://archive-api.open-meteo.com/v1/archive
No API key required.

Locations
---------
  London     (51.51, -0.13)  — southern England demand centre
  Birmingham (52.48, -1.90)  — Midlands, geographic centre of GB demand
  Edinburgh  (55.95, -3.19)  — Scotland, large share of UK wind capacity

Output columns
--------------
  settlement_period_start     — UTC half-hourly timestamp (matches Elexon)
  temp_london_c               — temperature at 2m, London (°C)
  temp_birmingham_c           — temperature at 2m, Birmingham (°C)
  temp_edinburgh_c            — temperature at 2m, Edinburgh (°C)
  temp_avg_c                  — national average temperature (mean of 3)
  wind_speed_london_kmh       — wind speed at 10m, London (km/h)
  wind_speed_birmingham_kmh   — wind speed at 10m, Birmingham (km/h)
  wind_speed_edinburgh_kmh    — wind speed at 10m, Edinburgh (km/h)
  wind_speed_avg_kmh          — average wind speed across 3 locations
  cloud_cover_london_pct      — cloud cover %, London
  cloud_cover_birmingham_pct  — cloud cover %, Birmingham
  cloud_cover_edinburgh_pct   — cloud cover %, Edinburgh
  cloud_cover_avg_pct         — average cloud cover across 3 locations

Usage
-----
  python fetch_weather.py
  python fetch_weather.py --start 2024-01-01 --end 2026-01-01
"""

import argparse
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
OUTPUT_DIR = Path(r"D:\Projects\UK_electricity_forecast\Data\raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOCATIONS = {
    "london":     {"latitude": 51.51, "longitude": -0.13},
    "birmingham": {"latitude": 52.48, "longitude": -1.90},
    "edinburgh":  {"latitude": 55.95, "longitude": -3.19},
}

VARIABLES = ["temperature_2m", "wind_speed_10m", "cloud_cover"]

MAX_RETRIES = 5
BACKOFF_BASE = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fetch one location
# ---------------------------------------------------------------------------

def fetch_location(name: str, lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """
    Fetch hourly weather for one location for the full date range.
    Returns a DataFrame indexed by UTC timestamp with columns:
      temperature_2m, wind_speed_10m, cloud_cover
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(VARIABLES),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
    }

    log.info("Fetching weather for %s (%s, %s) %s → %s", name, lat, lon, start, end)

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=60)
            if resp.status_code == 200:
                break
            elif resp.status_code == 429:
                wait = BACKOFF_BASE ** attempt
                log.warning("Rate limited. Sleeping %ds ...", wait)
                time.sleep(wait)
            else:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.RequestException as exc:
            wait = BACKOFF_BASE ** attempt
            log.warning("Request error: %s. Sleeping %ds ...", exc, wait)
            time.sleep(wait)
    else:
        raise RuntimeError(f"Failed to fetch weather for {name} after {MAX_RETRIES} retries")

    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"API error for {name}: {data.get('reason', data)}")

    hourly = data["hourly"]

    df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"], utc=True),
        "temperature_2m": hourly["temperature_2m"],
        "wind_speed_10m": hourly["wind_speed_10m"],
        "cloud_cover":    hourly["cloud_cover"],
    })

    log.info("Fetched %d hourly rows for %s", len(df), name)
    return df


# ---------------------------------------------------------------------------
# Resample hourly → half-hourly
# ---------------------------------------------------------------------------

def resample_to_halfhourly(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """
    Forward-fill hourly weather data onto a half-hourly UTC grid.

    Each hourly value at HH:00 applies to both settlement periods:
      HH:00 → HH:00 and HH:30

    Method: reindex onto the half-hourly grid and forward-fill.
    """
    df = df.set_index("time").sort_index()

    # Build the target half-hourly grid
    hh_index = pd.date_range(
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC") - pd.Timedelta(minutes=30),
        freq="30min",
    )

    # Reindex and forward-fill (each hour fills its own :00 and the next :30)
    df_hh = df.reindex(hh_index).ffill()

    # Any remaining NaNs at the very start (before first hourly value) — backfill
    df_hh = df_hh.bfill()

    df_hh.index.name = "settlement_period_start"
    return df_hh.reset_index()


# ---------------------------------------------------------------------------
# Main collection
# ---------------------------------------------------------------------------

def collect_weather(start: date, end: date) -> pd.DataFrame:
    """
    Fetch weather for all three locations, resample to half-hourly,
    and combine into a single wide DataFrame.
    """
    frames = {}

    for name, coords in LOCATIONS.items():
        raw = fetch_location(name, coords["latitude"], coords["longitude"], start, end)
        hh = resample_to_halfhourly(raw, start, end)
        frames[name] = hh
        time.sleep(1)  # polite delay between location requests

    # ── Build combined wide DataFrame ───────────────────────────────────────
    base = frames["london"][["settlement_period_start"]].copy()

    for name, df in frames.items():
        df = df.rename(columns={
            "temperature_2m": f"temp_{name}_c",
            "wind_speed_10m": f"wind_speed_{name}_kmh",
            "cloud_cover":    f"cloud_cover_{name}_pct",
        })
        base = base.merge(df, on="settlement_period_start", how="left")

    # ── National averages ────────────────────────────────────────────────────
    base["temp_avg_c"] = base[
        ["temp_london_c", "temp_birmingham_c", "temp_edinburgh_c"]
    ].mean(axis=1)

    base["wind_speed_avg_kmh"] = base[
        ["wind_speed_london_kmh", "wind_speed_birmingham_kmh", "wind_speed_edinburgh_kmh"]
    ].mean(axis=1)

    base["cloud_cover_avg_pct"] = base[
        ["cloud_cover_london_pct", "cloud_cover_birmingham_pct", "cloud_cover_edinburgh_pct"]
    ].mean(axis=1)

    # ── Coverage report ──────────────────────────────────────────────────────
    null_pct = base.isnull().mean().mul(100).round(1)
    log.info("Weather shape: %s", base.shape)
    log.info(
        "Null %% per column:\n%s",
        null_pct[null_pct > 0].to_string() if null_pct[null_pct > 0].any() else "  none",
    )

    return base.sort_values("settlement_period_start").reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch Open-Meteo weather data")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end",   default="2026-01-01")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    df = collect_weather(start, end)

    out_path = OUTPUT_DIR / "weather_raw.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Saved %s  (%d rows, %d columns)", out_path, *df.shape)

    print("\n─── Sample (first 5 rows) ───────────────────────────────────────")
    print(df.head().to_string(index=False))
    print("\n─── Stats ───────────────────────────────────────────────────────")
    print(df[["temp_avg_c", "wind_speed_avg_kmh", "cloud_cover_avg_pct"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
