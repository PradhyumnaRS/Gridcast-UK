# ⚡ Gridcast UK — Electricity Price Forecasting Engine

> Live UK electricity spot price forecaster — XGBoost · FastAPI · Redis · Docker · React

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-yellow)](https://huggingface.co/spaces/Pradh2430/gridcast-uk)
[![GitHub](https://img.shields.io/badge/GitHub-PradhyumnaRS%2FGridcast--UK-blue)](https://github.com/PradhyumnaRS/Gridcast-UK)

---

## Overview

Gridcast UK is an end-to-end MLOps project that forecasts the next UK electricity settlement period price (T+30 min) using real-time data from three live APIs.

**Key results on held-out test set (Sep–Dec 2025):**

| Metric | Value |
|---|---|
| MAE | £17.37/MWh |
| RMSE | £23.97/MWh |
| Naive baseline MAE | £20.94/MWh |
| Improvement over baseline | 17.1% |
| Directional accuracy | 66.1% |

---

## Architecture

```
Elexon BMRS API ──┐
Open-Meteo API  ──┼──► Feature Engineering ──► XGBoost Model ──► FastAPI ──► React Dashboard
Carbon Intensity ─┘         (89 features)        (T+30 min)      + Redis
```

### Stack

| Layer | Technology |
|---|---|
| Model | XGBoost (GPU accelerated, Optuna tuned) |
| Feature store | Parquet (34,747 rows × 89 features) |
| Inference API | FastAPI + Pydantic |
| Caching | Redis (5 min TTL for live, 24h for historical) |
| Containerisation | Docker + docker-compose |
| Frontend | React + Recharts |
| Deployment | HuggingFace Spaces |
| Data sources | Elexon BMRS, Open-Meteo, Carbon Intensity API |

---

## Features

### Live Prediction
- Fetches real-time generation mix from Elexon BMRS
- Fetches current weather for London, Birmingham, Edinburgh from Open-Meteo
- Fetches carbon intensity from the Carbon Intensity API
- Fetches last 8 days of real settlement prices for lag feature construction
- Falls back to latest parquet row if any API is unavailable

### Historical Lookup
- Look up predictions for any settlement period from Jan 2024 to Dec 2025
- Pre-computed features from the training dataset

### Dashboard
- Live price forecast with price tier classification (Negative / Low / Moderate / High / Spike)
- Predicted price history chart
- Historical settlement period lookup
- Model metadata panel

---

## Model

### Training Data
- **Source:** Elexon BMRS, Open-Meteo, Carbon Intensity API
- **Range:** January 2024 – August 2025 (training), September – December 2025 (test)
- **Frequency:** Half-hourly (48 settlement periods/day)
- **Target:** `price_next` — next settlement period mid price (£/MWh)

### Feature Engineering (89 features)
- **Price lags:** lag-1 through lag-9, lag-48 (24h), lag-336 (1 week) — validated via ACF/PACF + Mutual Information
- **Rolling statistics:** 1h mean, 24h mean, 24h std — windows selected via Spearman sensitivity analysis
- **Price dynamics:** momentum (30min, 2h), daily range, vs-daily-mean
- **Generation mix:** wind/gas/nuclear ratios, renewable surplus, supply-demand ratio
- **Weather-derived:** HDD (base 15.5°C), wind power proxy, wind surprise, temperature spread
- **Cyclical time encoding:** sin/cos pairs for hour, day-of-week, month
- **Calendar flags:** weekend, UK public holiday, morning peak, evening peak, overnight
- **Interactions:** wind-demand interaction, cold evening peak

### Hyperparameter Tuning
- **Method:** Bayesian optimisation (Optuna, TPE sampler)
- **Trials:** 100
- **CV:** TimeSeriesSplit with 5 expanding folds
- **Best params:** n_estimators=1232, max_depth=5, learning_rate=0.0048, reg_lambda=8.33

### Key Findings
- `daily_price_range` (volatility carry) was the single most important feature — consistent with volatility clustering in electricity markets
- Weather features (cloud cover, temperature spread) outranked price autocorrelation features, suggesting the model learned genuine supply-demand drivers
- Negative price periods (excess renewable generation) had higher MAE (£26.44/MWh) — structurally harder to forecast

---

## Project Structure

```
Gridcast-UK/
├── api/                        FastAPI backend
│   ├── main.py                 App, endpoints, Redis caching
│   ├── features.py             Real-time feature construction
│   ├── live.py                 Live data fetching (3 APIs)
│   └── schemas.py              Pydantic request/response models
├── data_pipeline/              Data collection scripts
│   ├── collect_elexon.py       Elexon BMRS prices, demand, generation
│   ├── fetch_weather.py        Open-Meteo weather (3 UK locations)
│   ├── fetch_carbon.py         Carbon Intensity API
│   ├── merge_datasets.py       Join all sources on settlement_period_start
│   └── clean_data.py           Null handling, validation
├── ml/                         Machine learning
│   ├── feature_engineering.py  89-feature pipeline with leakage prevention
│   ├── train_model.py          XGBoost training, Optuna tuning, evaluation
│   └── predict.py              CLI inference script
├── frontend/                   React dashboard
│   ├── src/App.js              Main dashboard component
│   └── src/App.css             Energy-themed styles
├── Data/processed/             Feature store
│   └── features.parquet        34,747 rows × 90 columns
├── Models/                     Trained model
│   └── xgboost_price_forecaster.json
├── Dockerfile                  Single-container deployment
├── docker-compose.yml          API + Redis local development
└── requirements.txt            Python dependencies
```

---

## Running Locally

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker Desktop

### 1. Install dependencies
```bash
pip install -r requirements.txt
cd frontend && npm install
```

### 2. Train the model (optional — pre-trained model included)
```bash
python ml/train_model.py --n-trials 100 --cv-splits 5
```

### 3. Start the API + Redis with Docker
```bash
docker-compose up --build
```

### 4. Start the React frontend
```bash
cd frontend && npm start
```

Open `http://localhost:3000`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Model, features, Redis status |
| POST | `/predict/live` | Live prediction using real-time APIs |
| POST | `/predict/historical` | Historical prediction from parquet |
| GET | `/docs` | Interactive Swagger UI |

### Example response
```json
{
  "input_datetime_utc": "2025-10-01T14:00:00+00:00",
  "forecast_datetime_utc": "2025-10-01T14:30:00+00:00",
  "predicted_price_gbp_mwh": 87.42,
  "source": "live",
  "data_source": "live_apis",
  "cached": false
}
```

---

## Limitations & Future Work

- **Distribution shift:** Model trained on 2024–2025 data. Performance degrades when prices move outside the training range (e.g. March 2026 price spike). A production system would retrain monthly with a rolling window.
- **Negative prices:** Higher MAE on negative price periods (£26.44/MWh) — structural regime change driven by renewable oversupply is inherently harder to forecast from lagged features.
- **No spike periods in test set:** The Jan 2025 £2,900/MWh spike event was in training data; Sep–Dec 2025 test set had no spikes above £500/MWh.
- **Future:** Ensemble with LSTM for sequential pattern capture, SHAP explainability dashboard, automated retraining pipeline.

---

## Author

**Pradhyumna R Shetty**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/pradhyumna-r-shetty)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Pradh2430-yellow)](https://huggingface.co/Pradh2430)
