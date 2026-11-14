import json
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

def optimize_replenishment():
    df = pd.read_csv("data/forecast_results.csv")
    with open("reports/model_comparison.json") as f:
        model_metrics = json.load(f)

    # Upcoming 7-day planning horizon per State -> Store -> Item
    horizon_7d = (
        df.sort_values("date")
        .groupby(["state_id", "store_id", "item_id"])
        .tail(7)
        .groupby(["state_id", "store_id", "item_id"])
        .agg(
            predicted_7d_demand=("forecast_point", "sum"),
            upper_90_demand=("forecast_upper_90", "sum"),
            avg_sell_price=("sell_price", "mean")
        )
        .reset_index()
    )

    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        raise RuntimeError("Could not initialize OR-Tools GLOP solver.")

    order_vars = {}
    shortage_vars = {}
    overstock_vars = {}

    HOLDING_COST_PER_UNIT = 1.50
    STOCKOUT_PENALTY_PER_UNIT = 12.00

    for idx, row in horizon_7d.iterrows():
        # Assume current warehouse inventory is 35% of weekly expected demand
        current_stock = 0.35 * row["predicted_7d_demand"]
        service_target = row["upper_90_demand"]
        max_shelf_capacity = row["upper_90_demand"] * 1.4

        q_order = solver.NumVar(0.0, max_shelf_capacity, f"order_{idx}")
        shortage = solver.NumVar(0.0, solver.infinity(), f"shortage_{idx}")
        excess = solver.NumVar(0.0, solver.infinity(), f"excess_{idx}")

        # Constraint 1: On-hand + Order + Shortage must meet 90th percentile service level
        solver.Add(current_stock + q_order + shortage >= service_target)

        # Constraint 2: Track excess inventory above expected 7-day demand
        solver.Add(current_stock + q_order - excess <= row["predicted_7d_demand"] * 1.10)

        order_vars[idx] = q_order
        shortage_vars[idx] = shortage
        overstock_vars[idx] = excess

    # Objective: Minimize Holding Cost + Stockout Penalty
    objective = solver.Objective()
    for idx in horizon_7d.index:
        objective.SetCoefficient(overstock_vars[idx], HOLDING_COST_PER_UNIT)
        objective.SetCoefficient(shortage_vars[idx], STOCKOUT_PENALTY_PER_UNIT)
    objective.SetMinimization()

    status = solver.Solve()
    status_str = "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE"

    recommendations = []
    for idx, row in horizon_7d.iterrows():
        opt_qty = int(np.ceil(order_vars[idx].solution_value()))
        current_stock = int(round(0.35 * row["predicted_7d_demand"]))
        safety_buffer = int(round(row["upper_90_demand"] - row["predicted_7d_demand"]))

        recommendations.append({
            "state_id": row["state_id"],
            "store_id": row["store_id"],
            "item_id": row["item_id"],
            "current_stock": current_stock,
            "predicted_7d_demand": round(row["predicted_7d_demand"], 1),
            "safety_buffer_units": max(0, safety_buffer),
            "optimal_order_qty": opt_qty
        })

    rec_df = pd.DataFrame(recommendations)
    rec_df.to_csv("data/ortools_recommendations.csv", index=False)

    final_metrics = {
        "project": "DS-02: Nationwide Demand Forecasting and Inventory Optimization",
        "solver_engine": "Google OR-Tools (GLOP Linear Optimizer)",
        "solver_status": status_str,
        "minimized_weekly_inventory_cost_usd": round(objective.Value(), 2),
        "total_recommended_order_units": int(rec_df["optimal_order_qty"].sum()),
        "model_accuracy_benchmarks": model_metrics,
        "store_replenishment_plan": recommendations
    }

    with open("reports/metrics_ds02.json", "w") as f:
        json.dump(final_metrics, f, indent=4)

    print(f"[STEP 4 COMPLETE] OR-Tools Status: {status_str} | Total Recommended Order: {final_metrics['total_recommended_order_units']:,} units")

if __name__ == "__main__":
    optimize_replenishment()
