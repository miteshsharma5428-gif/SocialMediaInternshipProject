# DS-02: Nationwide Demand Forecasting and Inventory Optimization

## Overview
An end-to-end hierarchical retail forecasting and inventory replenishment system modeled on the Walmart M5 hierarchy (`State -> Store -> Item`). Combines leakage-free time-series feature engineering, multi-model forecasting with 80% quantile prediction intervals, bottom-up hierarchical reconciliation, Google OR-Tools linear cost optimization, and an interactive Plotly Dash web application.

## Key Results
- **Reconciled Ensemble Accuracy:** `5.43%` WAPE | `0.2773` RMSSE (45% error reduction over the `9.89%` Seasonal Naive baseline).
- **OR-Tools Optimization:** Solved to `OPTIMAL` status, recommending `4,946` total replenishment units across all locations to minimize holding and stockout costs.

## Setup & Execution
    cd DS-02
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

    python src/1_generate_data.py
    python src/2_feature_engineering.py
    python src/3_forecast_reconcile.py
    python src/4_optimize_inventory.py
    python app.py

Open `http://localhost:8050` in your browser.
