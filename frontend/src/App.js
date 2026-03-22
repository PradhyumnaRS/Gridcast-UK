import React, { useState, useEffect, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine
} from 'recharts';
import { parseISO } from 'date-fns';
import { formatInTimeZone } from 'date-fns-tz';
import './App.css';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const API_BASE = process.env.REACT_APP_API_URL || '';
const REFRESH_INTERVAL = 30 * 60 * 1000; // 30 minutes — one settlement period

// ---------------------------------------------------------------------------
// Custom Tooltip for chart
// ---------------------------------------------------------------------------

const PriceTooltip = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="chart-tooltip">
      <p className="tooltip-time">{label}</p>
      <p className="tooltip-price">£{payload[0].value.toFixed(2)}<span>/MWh</span></p>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Price badge — colour coded by price level
// ---------------------------------------------------------------------------

const getPriceTier = (price) => {
  if (price < 0) return { label: 'NEGATIVE', cls: 'tier-negative' };
  if (price < 50) return { label: 'LOW', cls: 'tier-low' };
  if (price < 100) return { label: 'MODERATE', cls: 'tier-moderate' };
  if (price < 200) return { label: 'HIGH', cls: 'tier-high' };
  return { label: 'SPIKE', cls: 'tier-spike' };
};

// ---------------------------------------------------------------------------
// Helper — format a UTC ISO string for display, always in UTC
// ---------------------------------------------------------------------------

const fmtUTC = (isoString) =>
  formatInTimeZone(parseISO(isoString), 'UTC', 'dd MMM yyyy HH:mm');

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------

export default function App() {
  // Live prediction state
  const [livePrediction, setLivePrediction] = useState(null);
  const [liveLoading, setLiveLoading] = useState(false);
  const [liveError, setLiveError] = useState(null);

  // Historical prediction state
  const [historicalDatetime, setHistoricalDatetime] = useState('2025-10-01T14:00');
  const [historicalPrediction, setHistoricalPrediction] = useState(null);
  const [historicalLoading, setHistoricalLoading] = useState(false);
  const [historicalError, setHistoricalError] = useState(null);

  // Chart data — recent live predictions
  const [chartData, setChartData] = useState([]);

  // Last updated time
  const [lastUpdated, setLastUpdated] = useState(null);

  // ---------------------------------------------------------------------------
  // Fetch live prediction
  // ---------------------------------------------------------------------------

  const fetchLive = useCallback(async () => {
    setLiveLoading(true);
    setLiveError(null);
    try {
      const res = await fetch(`${API_BASE}/predict/live`, { method: 'POST' });
      if (!res.ok) throw new Error(`API error ${res.status}`);
      const data = await res.json();
      setLivePrediction(data);
      setLastUpdated(new Date());

      // Append to chart data (keep last 48 points = 24 hours)
      setChartData(prev => {
        const time = formatInTimeZone(parseISO(data.forecast_datetime_utc), 'UTC', 'HH:mm');
        const newPoint = {
          time,
          price: parseFloat(data.predicted_price_gbp_mwh.toFixed(2)),
          cached: data.cached,
        };
        const updated = [...prev, newPoint];
        return updated.slice(-48);
      });
    } catch (err) {
      setLiveError(err.message);
    } finally {
      setLiveLoading(false);
    }
  }, []);

  // Auto-refresh every settlement period
  useEffect(() => {
    fetchLive();
    const interval = setInterval(fetchLive, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [fetchLive]);

  // ---------------------------------------------------------------------------
  // Fetch historical prediction
  // ---------------------------------------------------------------------------

  const fetchHistorical = async () => {
    setHistoricalLoading(true);
    setHistoricalError(null);
    setHistoricalPrediction(null);
    try {
      const res = await fetch(`${API_BASE}/predict/historical`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ datetime_utc: historicalDatetime + ':00Z' }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `API error ${res.status}`);
      }
      const data = await res.json();
      setHistoricalPrediction(data);
    } catch (err) {
      setHistoricalError(err.message);
    } finally {
      setHistoricalLoading(false);
    }
  };

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  const tier = livePrediction ? getPriceTier(livePrediction.predicted_price_gbp_mwh) : null;
  const histTier = historicalPrediction ? getPriceTier(historicalPrediction.predicted_price_gbp_mwh) : null;

  return (
    <div className="app">
      {/* Background grid */}
      <div className="bg-grid" />

      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="logo">
            <span className="logo-bolt">⚡</span>
            <span className="logo-text">GRIDCAST</span>
          </div>
          <span className="logo-sub">UK Electricity Price Forecasting Engine</span>
        </div>
        <div className="header-right">
          {lastUpdated && (
            <span className="last-updated">
              Updated {formatInTimeZone(lastUpdated, 'UTC', 'HH:mm:ss')} UTC
            </span>
          )}
          <div className={`status-dot ${liveLoading ? 'pulsing' : 'active'}`} />
        </div>
      </header>

      <main className="main">

        {/* ── Live Prediction Card ── */}
        <section className="card card-live">
          <div className="card-header">
            <h2 className="card-title">LIVE FORECAST</h2>
            <button className="refresh-btn" onClick={fetchLive} disabled={liveLoading}>
              {liveLoading ? 'FETCHING...' : '↻ REFRESH'}
            </button>
          </div>

          {liveError && (
            <div className="error-banner">⚠ {liveError}</div>
          )}

          {livePrediction && (
            <div className="live-content">
              <div className="price-display">
                <span className="price-currency">£</span>
                <span className="price-value">
                  {livePrediction.predicted_price_gbp_mwh.toFixed(2)}
                </span>
                <span className="price-unit">/MWh</span>
              </div>

              <div className={`price-tier-badge ${tier.cls}`}>
                {tier.label}
              </div>

              <div className="forecast-meta">
                <div className="meta-item">
                  <span className="meta-label">INPUT PERIOD</span>
                  <span className="meta-value">
                    {fmtUTC(livePrediction.input_datetime_utc)} UTC
                  </span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">FORECASTING FOR</span>
                  <span className="meta-value highlight">
                    {fmtUTC(livePrediction.forecast_datetime_utc)} UTC
                  </span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">DATA SOURCE</span>
                  <span className={`meta-value ${livePrediction.data_source === 'live_apis' ? 'source-live' : 'source-fallback'}`}>
                    {livePrediction.data_source === 'live_apis' ? '● LIVE APIs' : '● PARQUET FALLBACK'}
                  </span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">CACHE</span>
                  <span className="meta-value">
                    {livePrediction.cached ? '✓ HIT' : '○ MISS'}
                  </span>
                </div>
              </div>
            </div>
          )}

          {!livePrediction && !liveError && !liveLoading && (
            <div className="empty-state">Fetching live data...</div>
          )}
          {liveLoading && !livePrediction && (
            <div className="loading-state">
              <div className="loading-bar" />
            </div>
          )}
        </section>

        {/* ── Chart ── */}
        <section className="card card-chart">
          <div className="card-header">
            <h2 className="card-title">PREDICTED PRICE HISTORY</h2>
            <span className="chart-subtitle">Last {chartData.length} forecasts (T+30 min)</span>
          </div>

          {chartData.length === 0 ? (
            <div className="empty-state">Predictions will appear here as they are fetched.</div>
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="priceGradient" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#00d4aa" />
                    <stop offset="100%" stopColor="#0088cc" />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis
                  dataKey="time"
                  tick={{ fill: '#7a9bb5', fontSize: 11, fontFamily: 'Space Mono' }}
                  axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: '#7a9bb5', fontSize: 11, fontFamily: 'Space Mono' }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={v => `£${v}`}
                  width={60}
                />
                <Tooltip content={<PriceTooltip />} />
                <ReferenceLine y={0} stroke="rgba(255,80,80,0.3)" strokeDasharray="4 4" />
                <Line
                  type="monotone"
                  dataKey="price"
                  stroke="url(#priceGradient)"
                  strokeWidth={2.5}
                  dot={{ fill: '#00d4aa', r: 3, strokeWidth: 0 }}
                  activeDot={{ fill: '#ffffff', r: 5, strokeWidth: 2, stroke: '#00d4aa' }}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </section>

        {/* ── Historical Lookup ── */}
        <section className="card card-historical">
          <div className="card-header">
            <h2 className="card-title">HISTORICAL LOOKUP</h2>
            <span className="chart-subtitle">2024-01-08 → 2025-12-31</span>
          </div>

          <div className="historical-form">
            <div className="input-group">
              <label className="input-label">SETTLEMENT PERIOD (UTC)</label>
              <input
                type="datetime-local"
                className="datetime-input"
                value={historicalDatetime}
                onChange={e => setHistoricalDatetime(e.target.value)}
                min="2024-01-08T00:00"
                max="2025-12-31T23:30"
                step="1800"
              />
            </div>
            <button
              className="lookup-btn"
              onClick={fetchHistorical}
              disabled={historicalLoading}
            >
              {historicalLoading ? 'LOOKING UP...' : 'GET PREDICTION →'}
            </button>
          </div>

          {historicalError && (
            <div className="error-banner">⚠ {historicalError}</div>
          )}

          {historicalPrediction && (
            <div className="historical-result">
              <div className="hist-price-row">
                <div className="hist-price">
                  <span className="price-currency">£</span>
                  <span className="hist-price-value">
                    {historicalPrediction.predicted_price_gbp_mwh.toFixed(2)}
                  </span>
                  <span className="price-unit">/MWh</span>
                </div>
                <div className={`price-tier-badge ${histTier.cls}`}>
                  {histTier.label}
                </div>
              </div>
              <div className="forecast-meta">
                <div className="meta-item">
                  <span className="meta-label">INPUT</span>
                  <span className="meta-value">
                    {fmtUTC(historicalPrediction.input_datetime_utc)} UTC
                  </span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">FORECAST FOR</span>
                  <span className="meta-value highlight">
                    {fmtUTC(historicalPrediction.forecast_datetime_utc)} UTC
                  </span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">CACHE</span>
                  <span className="meta-value">
                    {historicalPrediction.cached ? '✓ HIT' : '○ MISS'}
                  </span>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* ── Model Info Footer ── */}
        <section className="card card-info">
          <div className="info-grid">
            <div className="info-item">
              <span className="info-label">MODEL</span>
              <span className="info-value">XGBoost</span>
            </div>
            <div className="info-item">
              <span className="info-label">TEST MAE</span>
              <span className="info-value">£17.37/MWh</span>
            </div>
            <div className="info-item">
              <span className="info-label">TEST RMSE</span>
              <span className="info-value">£23.97/MWh</span>
            </div>
            <div className="info-item">
              <span className="info-label">FEATURES</span>
              <span className="info-value">89</span>
            </div>
            <div className="info-item">
              <span className="info-label">HORIZON</span>
              <span className="info-value">T+30 min</span>
            </div>
            <div className="info-item">
              <span className="info-label">TRAINING DATA</span>
              <span className="info-value">Jan 2024 – Aug 2025</span>
            </div>
            <div className="info-item">
              <span className="info-label">DATA SOURCES</span>
              <span className="info-value">Elexon · Open-Meteo · Carbon Intensity</span>
            </div>
            <div className="info-item">
              <span className="info-label">TUNING</span>
              <span className="info-value">Optuna · 100 trials</span>
            </div>
          </div>
        </section>

      </main>
    </div>
  );
}
