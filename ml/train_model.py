"""
train_model.py — XGBoost model training for UK Electricity Price Forecasting Engine.

Loads the feature-engineered parquet dataset, performs a strictly chronological
train/test split, runs Time Series Cross-Validation with an expanding window,
tunes hyperparameters via Optuna (Bayesian optimisation), trains a final XGBoost
regressor on the full training set, evaluates on the held-out test set, and
persists all artefacts (model, plots, metrics).

Usage
-----
    python train_model.py
    python train_model.py --features Data/processed/features.parquet \\
                          --model-out Models/xgboost_price_forecaster.json \\
                          --output-dir Outputs \\
                          --split-date 2025-09-01 \\
                          --cv-splits 5 \\
                          --n-trials 100 \\
                          --log-level DEBUG

Outputs
-------
    Models/xgboost_price_forecaster.json  — trained XGBoost model
    Outputs/feature_importance.png        — top-40 feature importance bar chart
    Outputs/test_predictions.png          — predictions vs actuals on test set
    Outputs/metrics.json                  — MAE, RMSE, and segment-level metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
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
    # Silence optuna's verbose trial logs at INFO; keep them at DEBUG.
    optuna_logger = logging.getLogger("optuna")
    optuna_logger.setLevel(logging.WARNING if numeric_level == logging.INFO else numeric_level)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

TARGET: str = "price_next"


def _drop_non_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any columns XGBoost cannot handle (string / ArrowString dtypes).

    The Carbon Intensity API returns ``carbon_index`` as a string label
    (e.g. "very low", "low", "moderate", "high", "very high"). This is
    redundant with the numeric ``carbon_actual`` column and is dropped rather
    than encoded, keeping the feature space clean.

    ``pd.api.types.is_string_dtype`` catches both legacy ``object`` dtype and
    the newer PyArrow-backed ``ArrowStringArray``.

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
        logger.info(
            "Dropping %d non-numeric column(s) (redundant with numeric equivalents): %s",
            len(str_cols),
            str_cols,
        )
        df = df.drop(columns=str_cols)
    return df


def load_features(path: Path) -> pd.DataFrame:
    """Load the feature-engineered parquet file and validate its schema.

    Parameters
    ----------
    path:
        Filesystem path to ``features.parquet``.

    Returns
    -------
    pd.DataFrame
        DataFrame with a DatetimeIndex (UTC), feature columns, and the
        ``price_next`` target column. All non-numeric columns are dropped.

    Raises
    ------
    SystemExit
        If the file is missing, the target column is absent, the index is not
        datetime, or null values are found.
    """
    logger.info("Loading features from %s", path)
    if not path.exists():
        logger.error("Features file not found: %s", path)
        sys.exit(1)

    df = pd.read_parquet(path)
    logger.info("Loaded DataFrame: %d rows × %d columns", *df.shape)

    # Validate index
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("DataFrame index must be a DatetimeIndex; got %s", type(df.index))
        sys.exit(1)
    if df.index.tz is None:
        logger.warning("DatetimeIndex has no timezone — localising to UTC.")
        df.index = df.index.tz_localize("UTC")

    # Validate target
    if TARGET not in df.columns:
        logger.error("Target column '%s' not found in DataFrame.", TARGET)
        sys.exit(1)

    # Drop string/ArrowString columns — XGBoost requires numeric dtypes.
    # carbon_index is a string label redundant with the numeric carbon_actual.
    df = _drop_non_numeric_columns(df)

    # Null check
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.error(
            "Null values detected in %d column(s):\n%s",
            (null_counts > 0).sum(),
            null_counts[null_counts > 0],
        )
        sys.exit(1)

    logger.info(
        "Date range: %s → %s", df.index.min().isoformat(), df.index.max().isoformat()
    )
    logger.info("Final feature shape after cleaning: %d rows × %d columns", *df.shape)
    logger.debug("Columns: %s", df.columns.tolist())
    return df


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------


def chronological_split(
    df: pd.DataFrame, split_date: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the DataFrame into train and test sets at a fixed date boundary.

    The split is strictly chronological — no shuffling. Rows strictly *before*
    ``split_date`` go into training; rows on or after go into test.

    Parameters
    ----------
    df:
        Full feature DataFrame with a DatetimeIndex.
    split_date:
        ISO-8601 date string, e.g. ``"2025-09-01"``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(train_df, test_df)``
    """
    boundary = pd.Timestamp(split_date, tz="UTC")
    train = df[df.index < boundary]
    test = df[df.index >= boundary]
    logger.info(
        "Train: %d rows (%s → %s)",
        len(train),
        train.index.min().date(),
        train.index.max().date(),
    )
    logger.info(
        "Test:  %d rows (%s → %s)",
        len(test),
        test.index.min().date(),
        test.index.max().date(),
    )
    if len(train) == 0 or len(test) == 0:
        logger.error("Split produced an empty train or test set. Check --split-date.")
        sys.exit(1)
    return train, test


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target variable.

    Parameters
    ----------
    df:
        DataFrame containing feature columns and the ``price_next`` target.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(X, y)`` where *X* contains all columns except ``price_next`` and *y*
        is the ``price_next`` series.
    """
    feature_cols = [c for c in df.columns if c != TARGET]
    logger.debug("Feature count: %d", len(feature_cols))
    return df[feature_cols], df[TARGET]


# ---------------------------------------------------------------------------
# Hyperparameter tuning with Optuna
# ---------------------------------------------------------------------------


def _make_objective(
    X_train: pd.DataFrame, y_train: pd.Series, n_splits: int
) -> Any:
    """Return an Optuna objective function for XGBoost hyperparameter search.

    Uses an expanding-window TimeSeriesSplit so the cross-validation respects
    temporal ordering. MAE is minimised. Training runs on GPU via
    ``device="cuda"`` for maximum speed.

    Parameters
    ----------
    X_train:
        Training feature matrix.
    y_train:
        Training target series.
    n_splits:
        Number of cross-validation folds.

    Returns
    -------
    Callable
        Optuna objective function ``(trial) -> float``.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "device": "cuda",       # GPU acceleration — RTX 3050
            "tree_method": "hist",  # required for GPU in XGBoost 2.x
            "random_state": 42,
        }

        fold_maes: list[float] = []
        for fold, (idx_tr, idx_val) in enumerate(tscv.split(X_train)):
            X_tr, X_val = X_train.iloc[idx_tr], X_train.iloc[idx_val]
            y_tr, y_val = y_train.iloc[idx_tr], y_train.iloc[idx_val]

            model = XGBRegressor(**params)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            preds = model.predict(X_val)
            fold_mae = mean_absolute_error(y_val, preds)
            fold_maes.append(fold_mae)
            logger.debug("  Fold %d MAE: %.3f", fold + 1, fold_mae)

        cv_mae = float(np.mean(fold_maes))
        logger.debug("Trial %d — CV MAE: %.3f", trial.number, cv_mae)
        return cv_mae

    return objective


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int,
    n_trials: int,
) -> dict[str, Any]:
    """Run Optuna Bayesian optimisation to find XGBoost hyperparameters.

    Parameters
    ----------
    X_train:
        Training feature matrix.
    y_train:
        Training target series.
    n_splits:
        Number of TimeSeriesSplit folds for cross-validation.
    n_trials:
        Number of Optuna trials to run.

    Returns
    -------
    dict[str, Any]
        Best hyperparameter dictionary found by Optuna.
    """
    logger.info(
        "Starting Optuna hyperparameter search: %d trials, %d CV folds",
        n_trials,
        n_splits,
    )
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    objective = _make_objective(X_train, y_train, n_splits)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    logger.info("Best CV MAE: %.4f £/MWh", study.best_value)
    logger.info("Best hyperparameters: %s", best_params)
    return best_params


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any],
) -> XGBRegressor:
    """Train the final XGBoost model on the full training set.

    Parameters
    ----------
    X_train:
        Training feature matrix.
    y_train:
        Training target series.
    params:
        Hyperparameter dictionary (from Optuna or defaults).

    Returns
    -------
    XGBRegressor
        Fitted XGBoost regressor.
    """
    model_params = {
        **params,
        "device": "cuda",       # GPU acceleration — RTX 3050
        "tree_method": "hist",  # required for GPU in XGBoost 2.x
        "random_state": 42,
    }
    logger.info("Training final model with params: %s", model_params)
    model = XGBRegressor(**model_params)
    model.fit(X_train, y_train)
    logger.info("Training complete.")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Compute evaluation metrics on the test set.

    Primary metric: MAE (£/MWh) — most interpretable for electricity markets.
    Secondary metric: RMSE — penalises large errors more heavily, relevant
    given spike events. MAPE is deliberately excluded: the price series
    contains negative and near-zero values which make MAPE undefined or
    misleading.

    Segment metrics are also reported separately for:
    - Negative price periods (excess renewable generation)
    - Spike periods (> £500/MWh)

    Parameters
    ----------
    model:
        Fitted XGBoost regressor.
    X_test:
        Test feature matrix.
    y_test:
        Test target series.

    Returns
    -------
    dict[str, Any]
        Dictionary of metric names to float values.
    """
    preds = model.predict(X_test)
    y_arr = y_test.values

    mae = mean_absolute_error(y_arr, preds)
    rmse = float(np.sqrt(mean_squared_error(y_arr, preds)))

    # Segment: negative prices
    neg_mask = y_arr < 0
    mae_neg = (
        float(mean_absolute_error(y_arr[neg_mask], preds[neg_mask]))
        if neg_mask.any()
        else None
    )

    # Segment: spike prices (> £500/MWh)
    spike_mask = y_arr > 500
    mae_spike = (
        float(mean_absolute_error(y_arr[spike_mask], preds[spike_mask]))
        if spike_mask.any()
        else None
    )

    metrics: dict[str, Any] = {
        "mae_gbp_mwh": round(mae, 4),
        "rmse_gbp_mwh": round(rmse, 4),
        "n_test_samples": int(len(y_arr)),
        "n_negative_price_periods": int(neg_mask.sum()),
        "mae_negative_price_periods_gbp_mwh": round(mae_neg, 4) if mae_neg is not None else None,
        "n_spike_periods_gt500": int(spike_mask.sum()),
        "mae_spike_periods_gbp_mwh": round(mae_spike, 4) if mae_spike is not None else None,
    }

    logger.info("── Test set evaluation ──────────────────────────────────")
    logger.info("  MAE:              %.3f £/MWh", mae)
    logger.info("  RMSE:             %.3f £/MWh", rmse)
    logger.info("  Negative periods: %d  →  MAE %.3f £/MWh", neg_mask.sum(), mae_neg or 0)
    logger.info("  Spike periods:    %d  →  MAE %.3f £/MWh", spike_mask.sum(), mae_spike or 0)
    logger.info("─────────────────────────────────────────────────────────")

    return metrics


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_feature_importance(
    model: XGBRegressor,
    feature_names: list[str],
    output_path: Path,
    top_n: int = 40,
) -> None:
    """Save a horizontal bar chart of the top-N feature importances.

    Parameters
    ----------
    model:
        Fitted XGBoost regressor.
    feature_names:
        Ordered list of feature column names.
    output_path:
        File path to save the PNG.
    top_n:
        Number of top features to display.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.28)))
    ax.barh(range(len(top_names)), top_vals[::-1], color="#1f77b4", edgecolor="white")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (gain)", fontsize=11)
    ax.set_title(
        f"Top {top_n} Feature Importances — XGBoost UK Electricity Price Forecaster",
        fontsize=12,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Feature importance plot saved: %s", output_path)


def plot_predictions(
    y_test: pd.Series,
    preds: np.ndarray,
    output_path: Path,
    n_periods: int = 2016,
) -> None:
    """Save a time-series overlay of actual vs predicted prices on the test set.

    Plots up to the first ``n_periods`` (≈ 6 weeks at half-hourly resolution)
    to keep the chart legible.

    Parameters
    ----------
    y_test:
        Actual target values with DatetimeIndex.
    preds:
        Predicted values (same length as ``y_test``).
    output_path:
        File path to save the PNG.
    n_periods:
        Maximum number of half-hourly periods to plot.
    """
    idx = y_test.index[:n_periods]
    actual = y_test.values[:n_periods]
    predicted = preds[:n_periods]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(idx, actual, label="Actual", linewidth=0.9, color="#2c7bb6", alpha=0.85)
    ax.plot(
        idx, predicted, label="Predicted", linewidth=0.9,
        color="#d7191c", alpha=0.75, linestyle="--",
    )
    ax.set_xlabel("Settlement Period (UTC)")
    ax.set_ylabel("Price (£/MWh)")
    ax.set_title(
        "UK Electricity Spot Price: Actual vs Predicted (Test Set — first 6 weeks)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Predictions plot saved: %s", output_path)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(model: XGBRegressor, path: Path) -> None:
    """Save the trained XGBoost model to disk in native JSON format.

    Parameters
    ----------
    model:
        Fitted XGBoost regressor.
    path:
        Destination file path (should end in ``.json``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    logger.info("Model saved: %s", path)


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    """Persist evaluation metrics to a JSON file.

    Parameters
    ----------
    metrics:
        Dictionary of metric names to values.
    path:
        Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved: %s", path)


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
        description="Train XGBoost model for UK electricity price forecasting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("Data/processed/features.parquet"),
        help="Path to the feature-engineered parquet file.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("Models/xgboost_price_forecaster.json"),
        help="Output path for the trained model (XGBoost JSON format).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Outputs"),
        help="Directory for plots and metrics JSON.",
    )
    parser.add_argument(
        "--split-date",
        type=str,
        default="2025-09-01",
        help="ISO-8601 date string for chronological train/test boundary.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of TimeSeriesSplit folds for cross-validation.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials for hyperparameter search.",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip Optuna tuning and train with sensible default hyperparameters.",
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

_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 800,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}


def main(argv: list[str] | None = None) -> None:
    """End-to-end model training pipeline.

    Parameters
    ----------
    argv:
        Optional argument list; defaults to ``sys.argv[1:]``.
    """
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    logger.info("═" * 60)
    logger.info("UK Electricity Price Forecasting Engine — Model Training")
    logger.info("═" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────
    df = load_features(args.features)

    # ── 2. Train / test split ─────────────────────────────────────────────
    train_df, test_df = chronological_split(df, args.split_date)

    X_train, y_train = get_X_y(train_df)
    X_test, y_test = get_X_y(test_df)

    feature_names: list[str] = X_train.columns.tolist()
    logger.info("Features used: %d", len(feature_names))

    # ── 3. Hyperparameter tuning ──────────────────────────────────────────
    if args.skip_tuning:
        logger.info("Skipping Optuna tuning; using default hyperparameters.")
        best_params: dict[str, Any] = _DEFAULT_PARAMS
    else:
        best_params = tune_hyperparameters(
            X_train, y_train, n_splits=args.cv_splits, n_trials=args.n_trials
        )

    # ── 4. Final model training ───────────────────────────────────────────
    model = train_final_model(X_train, y_train, best_params)

    # ── 5. Evaluation ─────────────────────────────────────────────────────
    metrics = evaluate(model, X_test, y_test)

    # Attach training metadata
    metrics["split_date"] = args.split_date
    metrics["n_train_samples"] = len(X_train)
    metrics["n_features"] = len(feature_names)
    metrics["best_params"] = best_params
    metrics["skip_tuning"] = args.skip_tuning
    metrics["cv_splits"] = args.cv_splits
    metrics["n_optuna_trials"] = 0 if args.skip_tuning else args.n_trials

    # ── 6. Persist artefacts ──────────────────────────────────────────────
    _ensure_dir(args.output_dir)

    save_model(model, args.model_out)
    save_metrics(metrics, args.output_dir / "metrics.json")
    plot_feature_importance(model, feature_names, args.output_dir / "feature_importance.png")

    preds_test = model.predict(X_test)
    plot_predictions(y_test, preds_test, args.output_dir / "test_predictions.png")

    logger.info("═" * 60)
    logger.info("Training pipeline complete.")
    logger.info("  Model   → %s", args.model_out)
    logger.info("  Metrics → %s", args.output_dir / "metrics.json")
    logger.info("  Plots   → %s", args.output_dir)
    logger.info("═" * 60)


if __name__ == "__main__":
    main()