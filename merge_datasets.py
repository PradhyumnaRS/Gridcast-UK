"""
merge_datasets.py
=================
Merges all three raw data sources into a single master dataset
aligned on half-hourly UTC settlement_period_start timestamps.

Sources
-------
  combined_raw.parquet  -- Elexon prices, demand, generation mix
  weather_raw.parquet   -- Open-Meteo temperature, wind speed, cloud cover
  carbon_raw.parquet    -- Carbon intensity forecast and actual (gCO2/kWh)

Output
------
  Data/processed/master.parquet

Join strategy
-------------
  Left join on settlement_period_start with combined_raw as the
  authoritative index. Every Elexon price row is preserved even if
  weather or carbon data is missing for that period. Missing values
  are handled in clean_data.py.

Usage
-----
  python merge_datasets.py
  python merge_datasets.py --raw-dir Data/raw --out-dir Data/processed
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_utc(df, source_name):
    """
    Ensure settlement_period_start is timezone-aware UTC.
    Logs a warning if the column had no timezone and localises it.
    """
    col = df["settlement_period_start"]
    if col.dt.tz is None:
        log.warning(
            "%s: settlement_period_start had no timezone -- localising to UTC.",
            source_name,
        )
        df["settlement_period_start"] = col.dt.tz_localize("UTC")
    else:
        df["settlement_period_start"] = col.dt.tz_convert("UTC")
    return df


def _report(df, label):
    """Log shape, duplicate count, and null percentages for a DataFrame."""
    dupes = df.duplicated(subset=["settlement_period_start"]).sum()
    null_pct = df.isnull().mean().mul(100).round(1)
    nulls = null_pct[null_pct > 0]
    log.info(
        "%s -- shape: %s | dupes: %d | nulls: %s",
        label,
        df.shape,
        dupes,
        nulls.to_dict() if not nulls.empty else "none",
    )


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_sources(raw_dir):
    """
    Load the three raw parquet files and validate their timestamps.

    Parameters
    ----------
    raw_dir : Path
        Directory containing combined_raw, weather_raw, carbon_raw parquets.

    Returns
    -------
    elexon, weather, carbon : DataFrames
    """
    log.info("Loading raw parquet files from %s ...", raw_dir)

    elexon  = pd.read_parquet(raw_dir / "combined_raw.parquet")
    weather = pd.read_parquet(raw_dir / "weather_raw.parquet")
    carbon  = pd.read_parquet(raw_dir / "carbon_raw.parquet")

    elexon  = _ensure_utc(elexon,  "elexon")
    weather = _ensure_utc(weather, "weather")
    carbon  = _ensure_utc(carbon,  "carbon")

    _report(elexon,  "Elexon combined")
    _report(weather, "Weather")
    _report(carbon,  "Carbon intensity")

    return elexon, weather, carbon


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge(elexon, weather, carbon):
    """
    Left-join all three sources on settlement_period_start.

    Elexon is the authoritative left table -- every price row is kept.
    Weather and carbon rows that do not match an Elexon timestamp are
    discarded. Elexon rows with no matching weather or carbon data
    receive NaN values which are imputed in clean_data.py.

    Parameters
    ----------
    elexon  : Elexon combined DataFrame (prices + demand + genmix)
    weather : Open-Meteo weather DataFrame
    carbon  : Carbon intensity DataFrame

    Returns
    -------
    Merged DataFrame
    """
    log.info("Merging Elexon + weather ...")
    df = elexon.merge(weather, on="settlement_period_start", how="left")

    log.info("Merging + carbon intensity ...")
    df = df.merge(carbon, on="settlement_period_start", how="left")

    log.info("Merged shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate(df):
    """
    Validate and clean the merged master dataset.

    Steps
    -----
    1. Remove any duplicate timestamps (keep first)
    2. Sort chronologically
    3. Report coverage -- missing periods vs expected
    4. Report null percentages per column
    5. Domain sanity checks on price column

    Parameters
    ----------
    df : Merged DataFrame

    Returns
    -------
    Validated, sorted DataFrame
    """
    log.info("Validating master dataset ...")

    # ── Duplicates ───────────────────────────────────────────────────────────
    dupes = df.duplicated(subset=["settlement_period_start"]).sum()
    if dupes > 0:
        log.warning("Found %d duplicate timestamps -- dropping.", dupes)
        df = df.drop_duplicates(subset=["settlement_period_start"], keep="first")

    # ── Sort ─────────────────────────────────────────────────────────────────
    df = df.sort_values("settlement_period_start").reset_index(drop=True)

    # ── Coverage ─────────────────────────────────────────────────────────────
    expected = pd.date_range(
        start=df["settlement_period_start"].min(),
        end=df["settlement_period_start"].max(),
        freq="30min",
        tz="UTC",
    )
    missing = len(expected) - len(df)
    log.info(
        "Coverage -- expected: %d | actual: %d | missing: %d",
        len(expected), len(df), missing,
    )

    # ── Null report ──────────────────────────────────────────────────────────
    null_pct = df.isnull().mean().mul(100).round(1)
    nulls = null_pct[null_pct > 0]
    if not nulls.empty:
        log.info("Null %% per column:\n%s", nulls.to_string())
    else:
        log.info("Null %% per column: none")

    # ── Price sanity checks ──────────────────────────────────────────────────
    n_negative = (df["price_mid_gbp_mwh"] < 0).sum()
    n_spike    = (df["price_mid_gbp_mwh"] > 500).sum()
    log.info(
        "Price sanity -- negative periods: %d | spikes >500 GBP/MWh: %d",
        n_negative, n_spike,
    )

    # ── Date range ───────────────────────────────────────────────────────────
    log.info(
        "Date range: %s --> %s",
        df["settlement_period_start"].min().date(),
        df["settlement_period_start"].max().date(),
    )

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge Elexon, weather, and carbon intensity datasets."
    )
    parser.add_argument(
        "--raw-dir",
        default="Data/raw",
        help="Directory containing raw parquet files (default: Data/raw)",
    )
    parser.add_argument(
        "--out-dir",
        default="Data/processed",
        help="Output directory for master.parquet (default: Data/processed)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────────
    elexon, weather, carbon = load_sources(raw_dir)

    # ── Merge ────────────────────────────────────────────────────────────────
    df = merge(elexon, weather, carbon)

    # ── Validate ─────────────────────────────────────────────────────────────
    df = validate(df)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = out_dir / "master.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Saved %s  (%d rows, %d columns)", out_path, *df.shape)

    # ── Summary print ────────────────────────────────────────────────────────
    print("\n--- Columns ({}) ---".format(len(df.columns)))
    for col in df.columns:
        print("  {}".format(col))

    print("\n--- Sample (first 3 rows, key columns) ---")
    key_cols = [
        "settlement_period_start",
        "price_mid_gbp_mwh",
        "demand_mw",
        "gen_wind_mw",
        "gen_ccgt_mw",
        "temp_avg_c",
        "wind_speed_avg_kmh",
        "carbon_actual",
    ]
    available = [c for c in key_cols if c in df.columns]
    print(df[available].head(3).to_string(index=False))


if __name__ == "__main__":
    main()