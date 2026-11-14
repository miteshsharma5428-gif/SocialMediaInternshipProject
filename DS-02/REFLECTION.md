# Individual Reflection — Project DS-02
- **Candidate:** Mitesh (`NEX-INT-2026-2196`)
- **Project:** Nationwide Demand Forecasting and Inventory Optimization

## What I Owned & Built
- Architected a modular 4-stage data science pipeline covering M5 hierarchical data simulation (`State -> Store -> Item`), leakage-free lag and rolling feature engineering, quantile gradient boosting (`P10–P90` prediction intervals), neural network ensembling, bottom-up reconciliation, Google OR-Tools replenishment optimization, and an interactive Plotly Dash web app.

## Technical Challenges & Solutions
- **Preventing Data Leakage:** Ensured all rolling averages and standard deviations were shifted by 1 day prior to window calculation so future demand never leaked into training features.
- **Probabilistic Inventory Sizing:** Connected the 90th-percentile upper prediction bound directly into Google OR-Tools (`GLOP`) constraints so stores maintain safety stock against demand spikes without over-ordering.

## Key Takeaways
- Learned how combining probabilistic machine learning forecasts with operations research solvers translates raw predictions into concrete, cost-minimizing supply chain decisions.
