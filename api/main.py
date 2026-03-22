"""
main.py — FastAPI application for the UK Electricity Price Forecasting Engine.

Endpoints
---------
  GET  /health                — model, parquet, and Redis status
  POST /predict/live          — predict next 30 min price using live API data
  POST /predict/historical    — predict price for a historical settlement period

Caching
-------
  Redis is used for both endpoints:
    - /predict/live       → TTL 5 minutes (one settlement period)
    - /predict/historical → TTL 24 hours (historical data never changes)

  If Redis is unavailable the API degrades gracefully — predictions are still
  served, just without caching.

Usage (development)
-------------------
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Environment variables
---------------------
  MODEL_PATH      — path to XGBoost model JSON (default: Models/xgboost_price_forecaster.json)
  FEATURES_PATH   — path to features.parquet (default: Data/processed/features.parquet)
  REDIS_URL       — Redis connection URL (default: redis://localhost:6379)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd
import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from xgboost import XGBRegressor
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.features import (
    build_live_features,
    get_historical_features,
    get_latest_parquet_features,
    load_parquet,
)
from api.live import current_settlement_period, fetch_live_snapshot
from api.schemas import HealthResponse, HistoricalRequest, PredictionResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

MODEL_PATH = Path(os.getenv("MODEL_PATH", "Models/xgboost_price_forecaster.json"))
FEATURES_PATH = Path(os.getenv("FEATURES_PATH", "Data/processed/features.parquet"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

LIVE_CACHE_TTL = 300          # 5 minutes — one settlement period
HISTORICAL_CACHE_TTL = 86400  # 24 hours

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="UK Electricity Price Forecasting API",
    description=(
        "XGBoost-based T+30 min electricity spot price forecaster. "
        "Trained on Elexon BMRS, Open-Meteo, and Carbon Intensity data "
        "for Great Britain, 2024–2025."
    ),
    version="1.0.0",
)

# CORS — allow React frontend on any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Application state — loaded once at startup
# ---------------------------------------------------------------------------

_model: XGBRegressor | None = None
_parquet_df: pd.DataFrame | None = None
_redis_client: redis.Redis | None = None


@app.on_event("startup")
async def startup() -> None:
    """Load model, features, and connect to Redis on startup."""
    global _model, _parquet_df, _redis_client

    # Load XGBoost model
    logger.info("Loading model from %s ...", MODEL_PATH)
    if not MODEL_PATH.exists():
        logger.error("Model file not found: %s", MODEL_PATH)
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    _model = XGBRegressor()
    _model.load_model(str(MODEL_PATH))
    logger.info("Model loaded successfully.")

    # Load features parquet
    logger.info("Loading features from %s ...", FEATURES_PATH)
    if not FEATURES_PATH.exists():
        logger.error("Features file not found: %s", FEATURES_PATH)
        raise RuntimeError(f"Features file not found: {FEATURES_PATH}")
    _parquet_df = load_parquet(FEATURES_PATH)
    logger.info("Features loaded: %d rows × %d columns.", *_parquet_df.shape)

    # Connect to Redis
    try:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        _redis_client.ping()
        logger.info("Redis connected: %s", REDIS_URL)
    except Exception as exc:
        logger.warning("Redis unavailable (%s) — caching disabled.", exc)
        _redis_client = None


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_get(key: str) -> dict | None:
    """Retrieve a cached value from Redis.

    Parameters
    ----------
    key:
        Cache key string.

    Returns
    -------
    dict | None
        Parsed JSON value if found, else None.
    """
    if _redis_client is None:
        return None
    try:
        value = _redis_client.get(key)
        return json.loads(value) if value else None
    except Exception as exc:
        logger.warning("Redis GET failed: %s", exc)
        return None


def _cache_set(key: str, value: dict, ttl: int) -> None:
    """Store a value in Redis with a TTL.

    Parameters
    ----------
    key:
        Cache key string.
    value:
        Dictionary to serialise and store.
    ttl:
        Time-to-live in seconds.
    """
    if _redis_client is None:
        return
    try:
        _redis_client.setex(key, ttl, json.dumps(value))
    except Exception as exc:
        logger.warning("Redis SET failed: %s", exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health() -> HealthResponse:
    """Return API health status including model, parquet, and Redis state."""
    redis_ok = False
    if _redis_client is not None:
        try:
            _redis_client.ping()
            redis_ok = True
        except Exception:
            pass

    parquet_loaded = _parquet_df is not None
    model_loaded = _model is not None

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        features_loaded=parquet_loaded,
        redis_connected=redis_ok,
        feature_count=int(_model.n_features_in_) if model_loaded else 0,
        parquet_row_count=len(_parquet_df) if parquet_loaded else 0,
        latest_parquet_datetime_utc=(
            _parquet_df.index.max().isoformat() if parquet_loaded else ""
        ),
    )


@app.post("/predict/live", response_model=PredictionResponse, tags=["prediction"])
async def predict_live() -> PredictionResponse:
    """Predict the next settlement period price using live data.

    Fetches current generation mix, weather, and carbon intensity from
    live APIs. Uses real recent Elexon prices (last 8 days) to construct
    lag and rolling features — ensuring predictions are based on fresh
    price history rather than stale parquet data.

    Falls back to the latest parquet row if any live API call fails.
    Results are cached in Redis for 5 minutes (one settlement period).
    """
    if _model is None or _parquet_df is None:
        raise HTTPException(status_code=503, detail="Model or features not loaded.")

    current_dt = current_settlement_period()
    cache_key = f"live:{current_dt.isoformat()}"

    # ── Cache check ───────────────────────────────────────────────────────
    cached = _cache_get(cache_key)
    if cached:
        logger.info("Cache HIT — live prediction for %s", current_dt.isoformat())
        return PredictionResponse(**{**cached, "cached": True})

    # ── Fetch live data ───────────────────────────────────────────────────
    data_source = "live_apis"
    try:
        # fetch_live_snapshot now returns (snapshot_dict, recent_prices_series)
        snapshot, recent_prices = fetch_live_snapshot()
        feature_row = build_live_features(
            _parquet_df, snapshot, current_dt, recent_prices=recent_prices
        )
        input_dt = current_dt
    except Exception as exc:
        logger.warning("Live API fetch failed (%s) — falling back to parquet.", exc)
        feature_row, input_dt = get_latest_parquet_features(_parquet_df)
        data_source = "parquet_fallback"

    # ── Predict ───────────────────────────────────────────────────────────
    try:
        prediction = float(_model.predict(feature_row.to_numpy())[0])
    except Exception as exc:
        logger.error("Model prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    input_ts = pd.Timestamp(input_dt)
    if input_ts.tzinfo is None:
        input_ts = input_ts.tz_localize("UTC")
    forecast_dt = input_ts + pd.Timedelta(minutes=30)

    result = {
        "input_datetime_utc": input_ts.isoformat(),
        "forecast_datetime_utc": forecast_dt.isoformat(),
        "predicted_price_gbp_mwh": round(prediction, 4),
        "source": "live",
        "data_source": data_source,
        "cached": False,
    }

    # ── Cache and return ──────────────────────────────────────────────────
    _cache_set(cache_key, result, LIVE_CACHE_TTL)
    logger.info(
        "Live prediction: £%.2f/MWh for %s (source: %s)",
        prediction, forecast_dt.isoformat(), data_source,
    )
    return PredictionResponse(**result)


@app.post("/predict/historical", response_model=PredictionResponse, tags=["prediction"])
async def predict_historical(request: HistoricalRequest) -> PredictionResponse:
    """Predict the price for a historical settlement period.

    Looks up the pre-computed feature row from features.parquet by datetime.
    The datetime must be an exact half-hourly UTC timestamp that exists in
    the dataset (2024-01-08 to 2025-12-31).

    Results are cached in Redis for 24 hours — historical data never changes.
    """
    if _model is None or _parquet_df is None:
        raise HTTPException(status_code=503, detail="Model or features not loaded.")

    # ── Parse datetime ────────────────────────────────────────────────────
    try:
        dt = pd.Timestamp(request.datetime_utc)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        else:
            dt = dt.tz_convert("UTC")
    except Exception:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid datetime format: '{request.datetime_utc}'. "
                   f"Use ISO-8601, e.g. '2025-10-01T14:00:00Z'.",
        )

    cache_key = f"historical:{dt.isoformat()}"

    # ── Cache check ───────────────────────────────────────────────────────
    cached = _cache_get(cache_key)
    if cached:
        logger.info("Cache HIT — historical prediction for %s", dt.isoformat())
        return PredictionResponse(**{**cached, "cached": True})

    # ── Feature lookup ────────────────────────────────────────────────────
    try:
        feature_row = get_historical_features(_parquet_df, dt)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # ── Predict ───────────────────────────────────────────────────────────
    try:
        prediction = float(_model.predict(feature_row.to_numpy())[0])
    except Exception as exc:
        logger.error("Model prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    forecast_dt = dt + pd.Timedelta(minutes=30)
    result = {
        "input_datetime_utc": dt.isoformat(),
        "forecast_datetime_utc": forecast_dt.isoformat(),
        "predicted_price_gbp_mwh": round(prediction, 4),
        "source": "historical",
        "data_source": "parquet_lookup",
        "cached": False,
    }

    # ── Cache and return ──────────────────────────────────────────────────
    _cache_set(cache_key, result, HISTORICAL_CACHE_TTL)
    logger.info(
        "Historical prediction: £%.2f/MWh for %s",
        prediction, forecast_dt.isoformat(),
    )
    return PredictionResponse(**result)

FRONTEND_BUILD = Path("frontend/build")
 
if FRONTEND_BUILD.exists():
    # Serve static assets (JS, CSS, images)
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_BUILD / "static")),
        name="static",
    )
 
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_react(full_path: str):
        """Serve React app for all non-API routes."""
        index = FRONTEND_BUILD / "index.html"
        return FileResponse(str(index))
