"""
predict.py — Single-period inference for UK Electricity Price Forecasting Engine.

Loads the trained XGBoost model and feature dataset, looks up the feature vector
for a given settlement period datetime, and predicts the next period's electricity
spot price (T+30 min forecast).

This script is designed to map directly onto the FastAPI inference endpoint —
the core ``predict_single`` function is importable and reusable by the API layer.

Usage
-----
    python predict.py --datetime "2025-10-01 14:00"
    python predict.py --datetime "2025-11-15 08:30" --output-json Outputs/prediction.json
    python predict.py --datetime "2025-12-01 18:00" --model Models/xgboost_price_forecaster.json

Output (terminal)
-----------------
    ════════════════════════════════════════════════════════════
    UK Electricity Price Forecast — T+30 min
    ════════════════════════════════════════════════════════════
      Input period  : 2025-10-01 14:00:00 UTC
      Forecast for  : 2025-10-01 14:30:00 UTC
      Predicted price: £87.42 /MWh
    ════════════════════════════════════════════════════════════

Output (JSON)
-------------
    {
      "input_datetime_utc": "2025-10-01T14:00:00+00:00",
      "forecast_datetime_utc": "2025-10-01T14:30:00+00:00",
      "predicted_price_gbp_mwh": 87.42,
      "model_path": "Models/xgboost_price_forecaster.json",
      "features_path": "Data/processed/features.parquet"
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    """Configure root logger with a human-readable format.

    Parameters
    ----------
    level:
        Logging level string, e.g. ``"INFO"`` or ``"DEBUG"``.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET: str = "price_next"
SETTLEMENT_PERIOD: str = "30min"  # half-hourly grid


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(path: Path) -> XGBRegressor:
    """Load a trained XGBoost model from disk.

    Parameters
    ----------
    path:
        Path to the saved XGBoost JSON model file.

    Returns
    -------
    XGBRegressor
        Loaded XGBoost regressor ready for inference.

    Raises
    ------
    SystemExit
        If the model file does not exist.
    """
    if not path.exists():
        logger.error("Model file not found: %s", path)
        sys.exit(1)

    model = XGBRegressor()
    model.load_model(str(path))
    logger.info("Model loaded from %s", path)
    return model


# ---------------------------------------------------------------------------
# Feature loading and lookup
# ---------------------------------------------------------------------------


def _drop_non_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop string / ArrowString columns that XGBoost cannot handle.

    Mirrors the cleaning step in ``train_model.py`` to ensure the feature
    matrix seen at inference matches the one used during training.

    Parameters
    ----------
    df:
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with all non-numeric columns removed.
    """
    str_cols = [
        c for c in df.columns
        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object
    ]
    if str_cols:
        logger.debug("Dropping non-numeric columns: %s", str_cols)
        df = df.drop(columns=str_cols)
    return df


def load_features(path: Path) -> pd.DataFrame:
    """Load and clean the feature parquet file.

    Parameters
    ----------
    path:
        Path to ``features.parquet``.

    Returns
    -------
    pd.DataFrame
        Cleaned feature DataFrame with DatetimeIndex (UTC).

    Raises
    ------
    SystemExit
        If the file does not exist or the index is not a DatetimeIndex.
    """
    if not path.exists():
        logger.error("Features file not found: %s", path)
        sys.exit(1)

    df = pd.read_parquet(path)
    logger.debug("Loaded features: %d rows × %d columns", *df.shape)

    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("Features index must be DatetimeIndex; got %s", type(df.index))
        sys.exit(1)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df = _drop_non_numeric_columns(df)
    return df


def lookup_feature_row(
    df: pd.DataFrame, dt: pd.Timestamp
) -> pd.DataFrame:
    """Look up a single feature row by its settlement period datetime.

    Parameters
    ----------
    df:
        Full feature DataFrame with DatetimeIndex.
    dt:
        Target settlement period (UTC-aware).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame containing the feature vector for ``dt``.

    Raises
    ------
    SystemExit
        If ``dt`` is not found in the index.
    """
    if dt not in df.index:
        # Show the 5 nearest available timestamps to help the user
        nearest = df.index[df.index.get_indexer([dt], method="nearest")]
        logger.error(
            "Datetime %s not found in features. Nearest available: %s",
            dt.isoformat(),
            [t.isoformat() for t in nearest],
        )
        sys.exit(1)

    row = df.loc[[dt]]
    logger.debug("Feature row found for %s", dt.isoformat())
    return row


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_single(
    model: XGBRegressor,
    feature_row: pd.DataFrame,
) -> float:
    """Run inference on a single feature row.

    Parameters
    ----------
    model:
        Loaded XGBoost regressor.
    feature_row:
        Single-row DataFrame containing the 89 feature columns.
        Must NOT contain the target column ``price_next``.

    Returns
    -------
    float
        Predicted price in £/MWh for the next settlement period (T+30 min).
    """
    # Drop target column if present (defensive — shouldn't be at inference time)
    if TARGET in feature_row.columns:
        feature_row = feature_row.drop(columns=[TARGET])

    prediction = model.predict(feature_row.to_numpy())
    return float(prediction[0])


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def build_result(
    input_dt: pd.Timestamp,
    predicted_price: float,
    model_path: Path,
    features_path: Path,
) -> dict[str, Any]:
    """Build a structured result dictionary for output.

    Parameters
    ----------
    input_dt:
        The settlement period datetime that was looked up.
    predicted_price:
        The model's predicted price in £/MWh.
    model_path:
        Path to the model file (for traceability).
    features_path:
        Path to the features file (for traceability).

    Returns
    -------
    dict[str, Any]
        Structured result with input/forecast datetimes and predicted price.
    """
    forecast_dt = input_dt + pd.Timedelta(minutes=30)
    return {
        "input_datetime_utc": input_dt.isoformat(),
        "forecast_datetime_utc": forecast_dt.isoformat(),
        "predicted_price_gbp_mwh": round(predicted_price, 4),
        "model_path": str(model_path),
        "features_path": str(features_path),
    }


def print_result(result: dict[str, Any]) -> None:
    """Print the prediction result to stdout in a readable format.

    Parameters
    ----------
    result:
        Result dictionary from ``build_result``.
    """
    print()
    print("═" * 60)
    print("  UK Electricity Price Forecast — T+30 min")
    print("═" * 60)
    print(f"  Input period   : {result['input_datetime_utc']}")
    print(f"  Forecast for   : {result['forecast_datetime_utc']}")
    print(f"  Predicted price: £{result['predicted_price_gbp_mwh']:.2f} /MWh")
    print("═" * 60)
    print()


def save_result(result: dict[str, Any], path: Path) -> None:
    """Save the prediction result to a JSON file.

    Parameters
    ----------
    result:
        Result dictionary from ``build_result``.
    path:
        Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    logger.info("Prediction saved to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Predict UK electricity spot price for a given settlement period.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datetime",
        type=str,
        required=True,
        help=(
            "Settlement period datetime to predict from, in ISO-8601 format. "
            "Examples: '2025-10-01 14:00', '2025-11-15 08:30'. "
            "Assumed UTC if no timezone given. Must exist in features.parquet."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("Models/xgboost_price_forecaster.json"),
        help="Path to the trained XGBoost model JSON file.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("Data/processed/features.parquet"),
        help="Path to the feature-engineered parquet file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save prediction result as JSON.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Single-period prediction pipeline.

    Parameters
    ----------
    argv:
        Optional argument list; defaults to ``sys.argv[1:]``.
    """
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    # ── 1. Parse and validate datetime ───────────────────────────────────
    try:
        dt = pd.Timestamp(args.datetime)
    except Exception as exc:
        logger.error("Could not parse datetime '%s': %s", args.datetime, exc)
        sys.exit(1)

    if dt.tzinfo is None:
        logger.debug("No timezone in input datetime — assuming UTC.")
        dt = dt.tz_localize("UTC")
    else:
        dt = dt.tz_convert("UTC")

    logger.info("Input settlement period: %s", dt.isoformat())

    # ── 2. Load model ─────────────────────────────────────────────────────
    model = load_model(args.model)

    # ── 3. Load features and look up row ──────────────────────────────────
    df = load_features(args.features)
    feature_row = lookup_feature_row(df, dt)

    # ── 4. Predict ────────────────────────────────────────────────────────
    predicted_price = predict_single(model, feature_row)
    logger.info("Predicted price: £%.4f /MWh", predicted_price)

    # ── 5. Build and output result ────────────────────────────────────────
    result = build_result(dt, predicted_price, args.model, args.features)
    print_result(result)

    if args.output_json:
        save_result(result, args.output_json)


if __name__ == "__main__":
    main()
