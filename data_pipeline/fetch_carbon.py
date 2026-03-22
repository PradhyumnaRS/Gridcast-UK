"""
fetch_carbon.py
===============
Fetches 2 years of half-hourly carbon intensity data from the
Official Carbon Intensity API for Great Britain (NESO).

API base : https://api.carbonintensity.org.uk
No API key required.

Endpoint used
-------------
  GET /intensity/{from}/{to}
  Returns half-hourly carbon intensity forecast and actual values.
  Maximum window: 30 days. We chunk in 28-day windows to stay safe.

Timestamp note
--------------
  The API returns period-END timestamps. We subtract 30 minutes to
  convert to period-START to match our Elexon settlement_period_start.

Output columns
--------------
  settlement_period_start   — UTC half-hourly timestamp (matches Elexon)
  carbon_forecast           — forecast carbon intensity (gCO2/kWh)
  carbon_actual             — actual carbon intensity (gCO2/kWh)
  carbon_index              — descriptive index (low/moderate/high/very high)

Usage
-----
  python fetch_carbon.py
  python fetch_carbon.py --start 2024-01-01 --end 2026-01-01
"""

import argparse
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://api.carbonintensity.org.uk"
OUTPUT_DIR = Path(r"D:\Projects\UK_electricity_forecast\Data\raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_DAYS = 28
MAX_RETRIES = 5
BACKOFF_BASE = 2
REQUEST_DELAY = 0.5  # carbon intensity API recommends max 2 requests/min

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get(url: str, retries: int = MAX_RETRIES) -> dict:
    headers = {"Accept": "application/json"}
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = BACKOFF_BASE ** attempt
                log.warning("Rate limited (429). Sleeping %ds ...", wait)
                time.sleep(wait)
            else:
                log.error("HTTP %d for %s", resp.status_code, url)
                raise RuntimeError(f"HTTP {resp.status_code}: {url}")
        except requests.RequestException as exc:
            wait = BACKOFF_BASE ** attempt
            log.warning("Request error: %s. Sleeping %ds ...", exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} retries: {url}")


# ---------------------------------------------------------------------------
# Fetch carbon intensity
# ---------------------------------------------------------------------------

def _fmt(dt: datetime) -> str:
    """Format datetime as ISO-8601 UTC string for the API."""
    return dt.strftime("%Y-%m-%dT%H:%MZ")


def collect_carbon(start: date, end: date) -> pd.DataFrame:
    """
    Pull half-hourly carbon intensity in 28-day chunks.
    Returns DataFrame indexed by settlement_period_start (UTC).
    """
    log.info("Collecting carbon intensity %s → %s", start, end)
    records = []

    current = datetime(start.year, start.month, start.day)
    end_dt = datetime(end.year, end.month, end.day)
    total_days = (end_dt - current).days

    with tqdm(total=total_days, desc="carbon", unit="day") as pbar:
        while current < end_dt:
            chunk_end = min(current + timedelta(days=CHUNK_DAYS), end_dt)
            url = f"{BASE_URL}/intensity/{_fmt(current)}/{_fmt(chunk_end)}"

            try:
                data = _get(url)
                rows = data.get("data", [])
                records.extend(rows)
            except RuntimeError as exc:
                log.error("Skipping chunk %s–%s: %s", current, chunk_end, exc)

            pbar.update((chunk_end - current).days)
            time.sleep(REQUEST_DELAY)
            current = chunk_end

    if not records:
        raise ValueError("No carbon intensity data collected.")

    df = pd.DataFrame(records)
    log.info("Raw carbon shape: %s", df.shape)

    # ── Parse timestamps ─────────────────────────────────────────────────────
    # API returns period-END in 'to' field e.g. "2024-01-01T00:30Z"
    # Subtract 30 min to get period-START to match Elexon
    df["settlement_period_start"] = (
        pd.to_datetime(df["to"], utc=True) - pd.Timedelta(minutes=30)
    )

    # ── Extract intensity fields from nested dict ────────────────────────────
    df["carbon_forecast"] = df["intensity"].apply(lambda x: x.get("forecast"))
    df["carbon_actual"]   = df["intensity"].apply(lambda x: x.get("actual"))
    df["carbon_index"]    = df["intensity"].apply(lambda x: x.get("index"))

    # ── Select final columns ─────────────────────────────────────────────────
    df = df[["settlement_period_start", "carbon_forecast", "carbon_actual", "carbon_index"]]

    # ── Deduplicate ──────────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["settlement_period_start"], keep="first")
    if len(df) < before:
        log.info("Removed %d duplicate timestamps", before - len(df))

    # ── Coverage report ──────────────────────────────────────────────────────
    null_pct = df.isnull().mean().mul(100).round(1)
    log.info("Carbon intensity shape: %s", df.shape)
    log.info(
        "Null %% per column:\n%s",
        null_pct[null_pct > 0].to_string() if null_pct[null_pct > 0].any() else "  none",
    )

    return df.sort_values("settlement_period_start").reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch carbon intensity data")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end",   default="2026-01-01")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    df = collect_carbon(start, end)

    out_path = OUTPUT_DIR / "carbon_raw.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Saved %s  (%d rows, %d columns)", out_path, *df.shape)

    print("\n─── Sample (first 5 rows) ───────────────────────────────────────")
    print(df.head().to_string(index=False))
    print("\n─── Stats ───────────────────────────────────────────────────────")
    print(df[["carbon_forecast", "carbon_actual"]].describe().round(2).to_string())
    print("\n─── Carbon index distribution ───────────────────────────────────")
    print(df["carbon_index"].value_counts().to_string())


if __name__ == "__main__":
    main()
