"""
schemas.py — Pydantic request/response models for the UK Electricity Price Forecasting API.

These models define the contract between the FastAPI endpoints and any client
(React frontend, CLI, or external consumers). All datetimes are ISO-8601 UTC strings.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared response model
# ---------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    """Returned by both /predict/live and /predict/historical endpoints."""

    input_datetime_utc: str = Field(
        ...,
        description="The settlement period used as model input (UTC ISO-8601).",
        examples=["2025-10-01T14:00:00+00:00"],
    )
    forecast_datetime_utc: str = Field(
        ...,
        description="The settlement period being forecast — T+30 min (UTC ISO-8601).",
        examples=["2025-10-01T14:30:00+00:00"],
    )
    predicted_price_gbp_mwh: float = Field(
        ...,
        description="Predicted electricity spot price in £/MWh.",
        examples=[87.42],
    )
    source: Literal["live", "historical"] = Field(
        ...,
        description="Whether the features came from live API calls or historical parquet.",
    )
    data_source: Literal["live_apis", "parquet_fallback", "parquet_lookup"] = Field(
        ...,
        description=(
            "live_apis — features fetched from Elexon/Open-Meteo/Carbon Intensity in real time. "
            "parquet_fallback — live APIs failed, latest parquet row used instead. "
            "parquet_lookup — historical endpoint, features looked up from parquet."
        ),
    )
    cached: bool = Field(
        ...,
        description="Whether this response was served from Redis cache.",
    )


# ---------------------------------------------------------------------------
# Historical endpoint request
# ---------------------------------------------------------------------------


class HistoricalRequest(BaseModel):
    """Request body for POST /predict/historical."""

    datetime_utc: str = Field(
        ...,
        description=(
            "Settlement period datetime to look up in features.parquet. "
            "Must be an exact half-hourly UTC timestamp that exists in the dataset. "
            "ISO-8601 format, e.g. '2025-10-01T14:00:00Z' or '2025-10-01 14:00'."
        ),
        examples=["2025-10-01T14:00:00Z"],
    )


# ---------------------------------------------------------------------------
# Health check response
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Returned by GET /health."""

    status: Literal["ok"] = "ok"
    model_loaded: bool = Field(..., description="Whether the XGBoost model is loaded.")
    features_loaded: bool = Field(..., description="Whether features.parquet is loaded.")
    redis_connected: bool = Field(..., description="Whether Redis is reachable.")
    feature_count: int = Field(..., description="Number of features the model expects.")
    parquet_row_count: int = Field(..., description="Number of rows in features.parquet.")
    latest_parquet_datetime_utc: str = Field(
        ..., description="Latest settlement period available in features.parquet."
    )
