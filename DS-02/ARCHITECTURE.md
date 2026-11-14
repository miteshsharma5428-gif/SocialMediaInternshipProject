# System Architecture & Optimization Formulation (DS-02)

## 1. End-to-End Pipeline Architecture

    [M5 Hierarchical Sales Data: State -> Store -> Item]
            │
            ▼
    [Feature Engineering (`src/2_feature_engineering.py`)]
      ├── Historical Lags: 7, 14, and 28 days
      ├── Shifted Rolling Statistics: 7-day mean/std, 28-day mean
      └── Exogenous Signals: Promotional discounts, retail holidays, day-of-week
            │
            ▼
    [Forecasting & Hierarchical Reconciliation (`src/3_forecast_reconcile.py`)]
      ├── Statistical Baselines: Seasonal Naive & Holt-Winters Exponential Smoothing
      ├── Quantile Gradient Boosted Trees: Point + P10/P90 (80% Prediction Intervals)
      ├── Deep Neural Forecaster: Multi-Layer Perceptron (64x32 hidden layers)
      └── Bottom-Up Reconciliation: Item forecasts sum accurately to Store & State totals
            │
            ▼
    [Google OR-Tools Linear Optimizer (`src/4_optimize_inventory.py`)]
      ├── Objective: Minimize ($1.50 * Overstock_Units) + ($12.00 * Stockout_Units)
      └── Constraints: Satisfy 90th-percentile service demand within shelf capacity
            │
            ▼
    [Plotly Dash Interactive Web Dashboard (`app.py` - Port 8050)]

## 2. Data Schema (`data/forecast_results.csv`)
- `date`, `state_id`, `store_id`, `item_id`: Hierarchical identifiers and daily timestamp.
- `sales`, `sell_price`, `is_promo`, `is_holiday`: Observed demand and exogenous retail drivers.
- `forecast_point`, `forecast_lower_10`, `forecast_upper_90`: Reconciled point forecast and 80% prediction interval bounds.
- `reconciled_store_total`, `reconciled_state_total`: Bottom-up aggregated hierarchy totals.
