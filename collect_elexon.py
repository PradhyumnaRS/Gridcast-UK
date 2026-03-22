"""
collect_elexon.py
=================
Step 1 — Elexon Insights Solution API: historical data collection
Pulls 2 years of half-hourly settlement prices, demand outturn, and
generation-mix data from the Elexon Insights Solution REST API.

Endpoints used
--------------
  Prices  : GET /balancing/settlement/system-prices/{settlementDate}
  Demand  : GET /demand/outturn                       (INDO – half-hourly)
  Gen mix : GET /datasets/FUELHH                      (half-hourly by fuel type)

API base : https://data.elexon.co.uk/bmrs/api/v1
No API key required — all endpoints are publicly accessible.

Output
------
  data/raw/prices_raw.parquet
  data/raw/demand_raw.parquet
  data/raw/genmix_raw.parquet
  data/raw/combined_raw.parquet   ← aligned on settlement_period_start (UTC)

Usage
-----
  pip install requests pandas pyarrow tqdm
  python collect_elexon.py

  # Override date range:
  python collect_elexon.py --start 2022-01-01 --end 2024-01-01

  # Dry-run (fetch 7 days only, useful for testing):
  python collect_elexon.py --dry-run
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

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Polite rate-limiting: seconds to sleep between HTTP requests
REQUEST_DELAY = 0.25

# Retry config
MAX_RETRIES = 5
BACKOFF_BASE = 2  # seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level HTTP helper
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None, retries: int = MAX_RETRIES) -> dict:
    """
    GET request with exponential back-off on 429/5xx responses.
    Returns parsed JSON dict. Raises RuntimeError on persistent failure.
    """
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = BACKOFF_BASE ** attempt
                log.warning("Rate-limited (429). Sleeping %ds …", wait)
                time.sleep(wait)
            elif resp.status_code >= 500:
                wait = BACKOFF_BASE ** attempt
                log.warning("Server error %d. Sleeping %ds …", resp.status_code, wait)
                time.sleep(wait)
            else:
                log.error("HTTP %d for %s | %s", resp.status_code, url, resp.text[:200])
                raise RuntimeError(f"HTTP {resp.status_code}: {url}")
        except requests.RequestException as exc:
            wait = BACKOFF_BASE ** attempt
            log.warning("Request exception: %s. Sleeping %ds …", exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} retries: {url}")


# ---------------------------------------------------------------------------
# 1. Settlement system prices  (DISEBSP)
# ---------------------------------------------------------------------------
# Endpoint : /balancing/settlement/system-prices/{settlementDate}
# Returns  : all 48 settlement periods for a given calendar date.
# Key fields:
#   settlementDate, settlementPeriod,
#   systemSellPrice (SSP, £/MWh),
#   systemBuyPrice  (SBP, £/MWh),
#   netImbalanceVolume (NIV, MWh)  ← +ve = system long (surplus)
#
# Note: SSP and SBP are the two "imbalance" prices used to settle
# generators/suppliers who deviate from their contracted positions.
# For a forecasting target use (SSP + SBP) / 2  or just SBP as a proxy
# for "spot" cost of electricity.


def fetch_prices_for_date(d: date) -> list[dict]:
    """Return list of settlement-period dicts for one calendar date."""
    url = f"{BASE_URL}/balancing/settlement/system-prices/{d.isoformat()}"
    data = _get(url)
    # Response shape: {"data": [...], "meta": {...}}
    return data.get("data", [])


def collect_prices(start: date, end: date) -> pd.DataFrame:
    """
    Pull settlement system prices for every date in [start, end).
    Returns DataFrame indexed by settlement_period_start (UTC).
    """
    log.info("Collecting prices %s → %s", start, end)
    records: list[dict] = []
    days = (end - start).days

    for offset in tqdm(range(days), desc="prices", unit="day"):
        d = start + timedelta(days=offset)
        try:
            rows = fetch_prices_for_date(d)
            records.extend(rows)
        except RuntimeError as exc:
            log.error("Skipping prices %s: %s", d, exc)
        time.sleep(REQUEST_DELAY)

    if not records:
        raise ValueError("No price data collected — check API connectivity.")

    df = pd.DataFrame(records)
    log.info("Raw prices shape: %s", df.shape)
    log.debug("Columns: %s", df.columns.tolist())

    # ── Normalise column names to snake_case ────────────────────────────────
    df.columns = [_snake(c) for c in df.columns]

    # ── Parse settlement date + period → UTC timestamp ──────────────────────
    # UK settlement periods are numbered 1–48 (1–50 on clock-change days).
    # Period 1 starts at 00:00 local time; each period is 30 min.
    df = _add_utc_timestamp(df)

    # ── Select and rename the columns we care about ─────────────────────────
    keep = {
        "settlement_period_start": "settlement_period_start",
        "settlement_date": "settlement_date",
        "settlement_period": "settlement_period",
        "system_sell_price": "price_ssp_gbp_mwh",
        "system_buy_price": "price_sbp_gbp_mwh",
        "net_imbalance_volume": "net_imbalance_vol_mwh",
    }
    df = _select_rename(df, keep)

    # Mid-price as primary forecasting target
    df["price_mid_gbp_mwh"] = (
        df["price_ssp_gbp_mwh"] + df["price_sbp_gbp_mwh"]
    ) / 2

    return df.sort_values("settlement_period_start").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Demand outturn  (INDO)
# ---------------------------------------------------------------------------
# Endpoint : /demand/outturn
# Params   : from, to  (ISO-8601 datetime strings)
# Returns  : Initial National Demand Outturn — MW, half-hourly.
# Key fields:
#   startTime, settlementDate, settlementPeriod,
#   initialDemandOutturn  (MW)   ← total GB electricity demand


def collect_demand(start: date, end: date, chunk_days: int = 28) -> pd.DataFrame:
    """
    Pull half-hourly national demand outturn in monthly chunks.
    The API accepts a date-range but may time-out on large windows,
    so we chunk by `chunk_days`.
    """
    log.info("Collecting demand %s → %s", start, end)
    url = f"{BASE_URL}/demand/outturn"
    frames: list[pd.DataFrame] = []

    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        params = {
            "settlementDateFrom": current.isoformat(),
            "settlementDateTo": chunk_end.isoformat(),
            "format": "json",
            }
        try:
            data = _get(url, params=params)
            rows = data.get("data", [])
            if rows:
                frames.append(pd.DataFrame(rows))
        except RuntimeError as exc:
            log.error("Skipping demand chunk %s–%s: %s", current, chunk_end, exc)
        time.sleep(REQUEST_DELAY)
        current = chunk_end

    if not frames:
        raise ValueError("No demand data collected.")

    df = pd.concat(frames, ignore_index=True)
    df.columns = [_snake(c) for c in df.columns]
    log.info("Raw demand shape: %s", df.shape)

    df = _add_utc_timestamp(df)

    keep = {
        "settlement_period_start": "settlement_period_start",
        "initial_demand_outturn": "demand_mw",
        "transmission_system_demand": "transmission_demand_mw",
    }
    df = _select_rename(df, keep)
    return df.sort_values("settlement_period_start").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Half-hourly generation by fuel type  (FUELHH)
# ---------------------------------------------------------------------------
# Endpoint  : /datasets/FUELHH
# Params    : publishDateTimeFrom, publishDateTimeTo  (ISO-8601)
# Returns   : Generation output (MW) broken down by fuel type per
#             settlement period. One row per (settlementDate,
#             settlementPeriod, fuelType).
#
# Fuel types available (subject to API version):
#   CCGT, OCGT, OIL, COAL, NUCLEAR, WIND, PS (pumped storage),
#   NPSHYD (non-PS hydro), INTFR (French interconnector),
#   INTIRL (Irish), INTNED (Netherlands), INTEW (East–West),
#   BIOMASS, OTHER


def collect_genmix(start: date, end: date, chunk_days: int = 7) -> pd.DataFrame:
    """
    Pull half-hourly generation mix. FUELHH can return many rows per day
    so we use smaller chunks of 7 days to avoid timeout/pagination issues.
    """
    log.info("Collecting generation mix %s → %s", start, end)
    url = f"{BASE_URL}/datasets/FUELHH"
    frames: list[pd.DataFrame] = []

    current = start
    with tqdm(total=(end - start).days, desc="genmix", unit="day") as pbar:
        while current < end:
            chunk_end = min(current + timedelta(days=chunk_days), end)
            params = {
                "publishDateTimeFrom": _dt_str(current),
                "publishDateTimeTo": _dt_str(chunk_end),
                "format": "json",
            }
            try:
                data = _get(url, params=params)
                rows = data.get("data", [])
                if rows:
                    frames.append(pd.DataFrame(rows))
            except RuntimeError as exc:
                log.error("Skipping genmix %s–%s: %s", current, chunk_end, exc)
            time.sleep(REQUEST_DELAY)
            pbar.update((chunk_end - current).days)
            current = chunk_end

    if not frames:
        raise ValueError("No generation mix data collected.")

    df = pd.concat(frames, ignore_index=True)
    df.columns = [_snake(c) for c in df.columns]
    log.info("Raw genmix shape (long format): %s", df.shape)

    df = _add_utc_timestamp(df)

    # ── Pivot from long (one row per fuel type) to wide ─────────────────────
    # After pivoting each column is named "gen_<fuel>_mw"
    fuel_col = _find_col(df, ["fuel_type", "fueltype", "psrtype"])
    gen_col = _find_col(df, ["generation", "quantity", "output", "level"])

    if fuel_col is None or gen_col is None:
        log.warning(
            "Could not identify fuel/generation columns. Available: %s",
            df.columns.tolist(),
        )
        return df  # return long-format as fallback

    df[fuel_col] = df[fuel_col].str.upper().str.strip()
    df_wide = (
        df.pivot_table(
            index="settlement_period_start",
            columns=fuel_col,
            values=gen_col,
            aggfunc="mean",
        )
        .reset_index()
    )
    df_wide.columns.name = None
    df_wide.columns = ["settlement_period_start"] + [
        f"gen_{c.lower()}_mw" for c in df_wide.columns[1:]
    ]

    # ── Derived features: total renewable, total fossil, renewable fraction ─
    renewable_fuels = [c for c in df_wide.columns if any(
        k in c for k in ["wind", "solar", "hydro", "npshyd", "biomass"]
    )]
    fossil_fuels = [c for c in df_wide.columns if any(
        k in c for k in ["ccgt", "ocgt", "oil", "coal"]
    )]

    if renewable_fuels:
        df_wide["gen_renewable_total_mw"] = df_wide[renewable_fuels].sum(axis=1)
    if fossil_fuels:
        df_wide["gen_fossil_total_mw"] = df_wide[fossil_fuels].sum(axis=1)

    all_gen_cols = [c for c in df_wide.columns if c.startswith("gen_") and c.endswith("_mw")]
    df_wide["gen_total_mw"] = df_wide[all_gen_cols].sum(axis=1)
    if renewable_fuels:
        df_wide["gen_renewable_fraction"] = (
            df_wide["gen_renewable_total_mw"] / df_wide["gen_total_mw"].replace(0, float("nan"))
        )

    return df_wide.sort_values("settlement_period_start").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Combine all three datasets
# ---------------------------------------------------------------------------

def combine(
    prices: pd.DataFrame,
    demand: pd.DataFrame,
    genmix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join prices ← demand ← genmix on settlement_period_start.
    Prices is the authoritative index — every price row is kept even if
    demand/genmix data is missing (will surface as NaN → impute later).
    """
    log.info("Combining datasets …")
    df = prices.merge(demand, on="settlement_period_start", how="left")
    df = df.merge(genmix, on="settlement_period_start", how="left")

    # ── Coverage report ─────────────────────────────────────────────────────
    n = len(df)
    expected = (
        pd.date_range(df["settlement_period_start"].min(),
                      df["settlement_period_start"].max(),
                      freq="30min")
    )
    missing_periods = len(expected) - n
    null_pct = df.isnull().mean().mul(100).round(1)

    log.info("Combined shape: %s", df.shape)
    log.info("Expected periods: %d  |  Missing: %d", len(expected), missing_periods)
    log.info(
        "Null %% per column:\n%s",
        null_pct[null_pct > 0].to_string() if null_pct[null_pct > 0].any() else "  none",
    )

    return df.sort_values("settlement_period_start").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _snake(name: str) -> str:
    """camelCase / PascalCase → snake_case."""
    import re
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    return s.lower()


def _dt_str(d: date) -> str:
    """Date → ISO-8601 datetime string expected by the API."""
    return datetime(d.year, d.month, d.day).isoformat() + "Z"


def _add_utc_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a UTC settlement_period_start column from the combination of
    settlementDate + settlementPeriod that the API returns.

    Settlement period 1 = 00:00–00:30 local UK time.
    We store timestamps as UTC so downstream joins are unambiguous.
    The UK is UTC+0 (winter) or UTC+1 (BST, summer); pandas .tz_localize
    with ambiguous='infer' handles the clock-change correctly.
    """
    date_col = _find_col(df, ["settlement_date", "settlementdate", "start_time", "starttime"])
    period_col = _find_col(df, ["settlement_period", "settlementperiod"])

    if date_col and period_col:
        # Build naive local timestamp: date + (period-1)*30min
        df["_local_dt"] = pd.to_datetime(df[date_col]) + pd.to_timedelta(
            (df[period_col].astype(int) - 1) * 30, unit="min"
        )
        try:
            df["settlement_period_start"] = (
                df["_local_dt"]
                .dt.tz_localize("Europe/London", ambiguous="infer", nonexistent="shift_forward")
                .dt.tz_convert("UTC")
            )
        except Exception:
            # Fall back: treat as UTC (minor error on DST boundary days only)
            df["settlement_period_start"] = df["_local_dt"].dt.tz_localize("UTC")
        df.drop(columns=["_local_dt"], inplace=True)
    elif "start_time" in df.columns:
        df["settlement_period_start"] = pd.to_datetime(df["start_time"], utc=True)

    return df


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _select_rename(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Keep only the columns in mapping keys, renaming to mapping values."""
    available = {k: v for k, v in mapping.items() if k in df.columns}
    missing = set(mapping.keys()) - set(available.keys())
    if missing:
        log.debug("Columns not found (will be absent): %s", missing)
    return df[list(available.keys())].rename(columns=available)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Collect Elexon historical data")
    parser.add_argument(
        "--start",
        default=(date.today() - timedelta(days=365 * 2)).isoformat(),
        help="Start date YYYY-MM-DD (default: 2 years ago)",
    )
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="End date YYYY-MM-DD exclusive (default: today)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch only the first 7 days — useful for verifying connectivity",
    )
    parser.add_argument(
        "--skip-genmix",
        action="store_true",
        help="Skip generation mix (speeds up dry-runs)",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    if args.dry_run:
        end = min(start + timedelta(days=7), end)
        log.info("DRY-RUN mode: collecting %s → %s only", start, end)

    # ── Prices ───────────────────────────────────────────────────────────────
    prices = collect_prices(start, end)
    prices_path = OUTPUT_DIR / "prices_raw.parquet"
    prices.to_parquet(prices_path, index=False)
    log.info("Saved %s  (%d rows)", prices_path, len(prices))

    # ── Demand ───────────────────────────────────────────────────────────────
    demand = collect_demand(start, end)
    demand_path = OUTPUT_DIR / "demand_raw.parquet"
    demand.to_parquet(demand_path, index=False)
    log.info("Saved %s  (%d rows)", demand_path, len(demand))

    # ── Generation mix ───────────────────────────────────────────────────────
    if not args.skip_genmix:
        genmix = collect_genmix(start, end)
        genmix_path = OUTPUT_DIR / "genmix_raw.parquet"
        genmix.to_parquet(genmix_path, index=False)
        log.info("Saved %s  (%d rows)", genmix_path, len(genmix))
    else:
        log.info("Skipping generation mix (--skip-genmix).")
        genmix = pd.DataFrame(columns=["settlement_period_start"])

    # ── Combined ─────────────────────────────────────────────────────────────
    combined = combine(prices, demand, genmix)
    combined_path = OUTPUT_DIR / "combined_raw.parquet"
    combined.to_parquet(combined_path, index=False)
    log.info("Saved %s  (%d rows, %d columns)", combined_path, *combined.shape)

    # ── Quick sanity print ───────────────────────────────────────────────────
    print("\n─── Sample (first 5 rows of combined) ───────────────────────────")
    print(combined.head().to_string(index=False))
    print("\n─── dtypes ──────────────────────────────────────────────────────")
    print(combined.dtypes.to_string())
    print("\n─── Price stats (£/MWh) ─────────────────────────────────────────")
    price_cols = [c for c in combined.columns if "price" in c]
    if price_cols:
        print(combined[price_cols].describe().round(2).to_string())


if __name__ == "__main__":
    main()
