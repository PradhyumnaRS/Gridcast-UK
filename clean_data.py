"""
clean_data.py
=============
Cleans and prepares the master dataset for feature engineering.

Input
-----
  Data/processed/master.parquet

Cleaning steps
--------------
  1. Drop the 4 periods where Elexon price data is missing entirely
     (we never fabricate the target variable)
  2. Fill gen_intgrnl_mw nulls with 0 -- the Greenlink interconnector
     did not exist before late 2024, so zero is physically correct
  3. Forward-fill short gaps in carbon intensity (0.1% null)
     These are genuine data publication gaps, not structural absences
  4. Verify no nulls remain in columns used for modelling
  5. Log a final coverage and null report

Output
------
  Data/processed/master_clean.parquet

Usage
-----
  python clean_data.py
  python clean_data.py --in-dir Data/processed --out-dir Data/processed
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

def _null_report(df, label=""):
    """Log null percentages for any column with nulls."""
    null_pct = df.isnull().mean().mul(100).round(2)
    nulls = null_pct[null_pct > 0]
    if not nulls.empty:
        log.info("%s null %% per column:\n%s", label, nulls.to_string())
    else:
        log.info("%s null %% per column: none", label)


# ---------------------------------------------------------------------------
# Cleaning steps
# ---------------------------------------------------------------------------

def drop_missing_price_rows(df):
    """
    Drop any rows where price_mid_gbp_mwh is null.

    These are periods where Elexon published no settlement price.
    We never forward-fill the target variable -- it is better to
    lose 4 training examples than to invent a label.
    """
    before = len(df)
    df = df.dropna(subset=["price_mid_gbp_mwh"])
    dropped = before - len(df)
    if dropped > 0:
        log.info("Dropped %d rows with missing price (target variable).", dropped)
    else:
        log.info("No missing price rows found.")
    return df


def fill_greenlink(df):
    """
    Fill gen_intgrnl_mw nulls with 0.

    The Greenlink interconnector between GB and Ireland came online
    in late 2024. All periods before its commissioning genuinely had
    zero flow on this link. Filling with 0 is physically correct.
    """
    null_count = df["gen_intgrnl_mw"].isnull().sum()
    if null_count > 0:
        df["gen_intgrnl_mw"] = df["gen_intgrnl_mw"].fillna(0.0)
        log.info("Filled %d gen_intgrnl_mw nulls with 0.", null_count)
    else:
        log.info("gen_intgrnl_mw has no nulls.")
    return df


def fill_carbon(df):
    """
    Forward-fill carbon intensity nulls.

    The Carbon Intensity API has ~0.1% missing periods due to
    publication delays. These are short isolated gaps. Forward-fill
    is appropriate here because carbon intensity changes slowly and
    the previous period's value is a good approximation.

    Any remaining nulls at the very start of the series are
    back-filled as a fallback.
    """
    for col in ["carbon_forecast", "carbon_actual", "carbon_index"]:
        if col not in df.columns:
            continue
        null_count = df[col].isnull().sum()
        if null_count > 0:
            df[col] = df[col].ffill().bfill()
            log.info("Forward-filled %d nulls in %s.", null_count, col)
        else:
            log.info("%s has no nulls.", col)
    return df


def verify_modelling_columns(df):
    """
    Verify that all columns required for modelling have no nulls.

    These are the columns that will be used as features or as the
    target variable in feature_engineering.py and train_model.py.
    Any remaining nulls here would silently corrupt the model.
    """
    modelling_cols = [
        "price_mid_gbp_mwh",
        "demand_mw",
        "gen_wind_mw",
        "gen_ccgt_mw",
        "gen_nuclear_mw",
        "gen_biomass_mw",
        "gen_renewable_total_mw",
        "gen_fossil_total_mw",
        "gen_renewable_fraction",
        "gen_total_mw",
        "temp_avg_c",
        "wind_speed_avg_kmh",
        "cloud_cover_avg_pct",
        "carbon_actual",
    ]

    available = [c for c in modelling_cols if c in df.columns]
    null_pct = df[available].isnull().mean().mul(100).round(2)
    nulls = null_pct[null_pct > 0]

    if not nulls.empty:
        log.warning(
            "Nulls remain in modelling columns -- review before feature engineering:\n%s",
            nulls.to_string(),
        )
    else:
        log.info("All modelling columns are null-free.")

    return nulls.empty


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def clean(df):
    """
    Run all cleaning steps in order and return the cleaned DataFrame.

    Parameters
    ----------
    df : Raw master DataFrame from merge_datasets.py

    Returns
    -------
    Cleaned DataFrame ready for feature engineering
    """
    log.info("Starting cleaning. Input shape: %s", df.shape)
    _null_report(df, "Before cleaning --")

    # Step 1: Drop rows with missing target variable
    df = drop_missing_price_rows(df)

    # Step 2: Fill Greenlink interconnector with 0
    df = fill_greenlink(df)

    # Step 3: Forward-fill carbon intensity gaps
    df = fill_carbon(df)

    # Step 4: Verify modelling columns are clean
    verify_modelling_columns(df)

    # Step 5: Final sort
    df = df.sort_values("settlement_period_start").reset_index(drop=True)

    log.info("Cleaning complete. Output shape: %s", df.shape)
    _null_report(df, "After cleaning --")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Clean the master dataset for feature engineering."
    )
    parser.add_argument(
        "--in-dir",
        default="Data/processed",
        help="Directory containing master.parquet (default: Data/processed)",
    )
    parser.add_argument(
        "--out-dir",
        default="Data/processed",
        help="Output directory for master_clean.parquet (default: Data/processed)",
    )
    args = parser.parse_args()

    in_path  = Path(args.in_dir)  / "master.parquet"
    out_path = Path(args.out_dir) / "master_clean.parquet"

    log.info("Loading %s ...", in_path)
    df = pd.read_parquet(in_path)

    df = clean(df)

    df.to_parquet(out_path, index=False)
    log.info("Saved %s  (%d rows, %d columns)", out_path, *df.shape)

    print("\n--- Final shape: {} rows x {} columns ---".format(*df.shape))
    print("\n--- Key column sample ---")
    key_cols = [
        "settlement_period_start",
        "price_mid_gbp_mwh",
        "demand_mw",
        "gen_wind_mw",
        "temp_avg_c",
        "carbon_actual",
        "gen_intgrnl_mw",
    ]
    available = [c for c in key_cols if c in df.columns]
    print(df[available].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
