import json
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

def calc_wape(y_true, y_pred):
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(y_true))

def calc_rmsse_hierarchical(train_df, test_df, pred_col):
    rmsse_list = []
    for (store, item), grp_test in test_df.groupby(["store_id", "item_id"]):
        grp_train = train_df[(train_df["store_id"] == store) & (train_df["item_id"] == item)]
        y_tr = grp_train["sales"].values
        y_te = grp_test["sales"].values
        y_hat = grp_test[pred_col].values

        scale = np.mean(np.diff(y_tr) ** 2)
        mse = np.mean((y_te - y_hat) ** 2)
        rmsse_list.append(np.sqrt(mse / max(scale, 1e-6)))
    return float(np.mean(rmsse_list))

def run_forecasting_pipeline():
    os.makedirs("reports", exist_ok=True)
    df = pd.read_csv("data/m5_features.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["state_id", "store_id", "item_id", "date"]).reset_index(drop=True)

    # Hold out final 28 days for testing
    cutoff_date = df["date"].max() - pd.Timedelta(days=28)
    train_mask = df["date"] <= cutoff_date
    test_mask = df["date"] > cutoff_date

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    # 1. Baseline A: Seasonal Naive (7-day lag)
    test_df["pred_naive"] = test_df["lag_7"].values

    # 2. Baseline B: Holt-Winters Exponential Smoothing
    hw_preds = []
    for (store, item), grp_train in train_df.groupby(["store_id", "item_id"]):
        grp_test = test_df[(test_df["store_id"] == store) & (test_df["item_id"] == item)]
        horizon = len(grp_test)
        try:
            hw_model = ExponentialSmoothing(
                grp_train["sales"].values,
                trend="add",
                seasonal="add",
                seasonal_periods=7
            ).fit()
            forecast_vals = np.maximum(0, hw_model.forecast(horizon))
        except Exception:
            forecast_vals = grp_test["rolling_mean_7"].values

        for idx, val in zip(grp_test.index, forecast_vals):
            hw_preds.append((idx, val))

    test_df["pred_exp_smoothing"] = test_df.index.map(pd.Series(dict(hw_preds)))

    # Encode categorical hierarchy columns
    df_encoded = pd.get_dummies(df, columns=["state_id", "store_id", "item_id"], drop_first=False)
    feature_cols = [c for c in df_encoded.columns if c not in ["date", "sales"]]

    X_train = df_encoded.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, "sales"].values
    X_test = df_encoded.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, "sales"].values

    # 3. Gradient Boosted Trees (Histogram Algorithm + Quantile Intervals)
    gbdt_point = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=42)
    gbdt_p10 = HistGradientBoostingRegressor(loss="quantile", quantile=0.10, max_iter=200, learning_rate=0.05, random_state=42)
    gbdt_p90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, max_iter=200, learning_rate=0.05, random_state=42)

    gbdt_point.fit(X_train, y_train)
    gbdt_p10.fit(X_train, y_train)
    gbdt_p90.fit(X_train, y_train)

    test_df["pred_gbdt"] = np.maximum(0, gbdt_point.predict(X_test))
    raw_p10 = np.maximum(0, gbdt_p10.predict(X_test))
    raw_p90 = np.maximum(0, gbdt_p90.predict(X_test))

    # 4. Deep Temporal Neural Network (MLP Architecture)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", max_iter=250, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    test_df["pred_neural"] = np.maximum(0, mlp.predict(X_test_scaled))

    # 5. Reconciled Ensemble (70% GBDT + 30% Neural)
    test_df["forecast_point"] = np.round(0.70 * test_df["pred_gbdt"] + 0.30 * test_df["pred_neural"], 1)
    test_df["forecast_lower_10"] = np.round(np.minimum(test_df["forecast_point"], raw_p10), 1)
    test_df["forecast_upper_90"] = np.round(np.maximum(test_df["forecast_point"], raw_p90), 1)

    # Bottom-Up Hierarchical Reconciliation (Item -> Store -> State)
    store_totals = test_df.groupby(["store_id", "date"])["forecast_point"].sum().reset_index()
    store_totals.rename(columns={"forecast_point": "reconciled_store_total"}, inplace=True)

    state_totals = test_df.groupby(["state_id", "date"])["forecast_point"].sum().reset_index()
    state_totals.rename(columns={"forecast_point": "reconciled_state_total"}, inplace=True)

    test_df = test_df.merge(store_totals, on=["store_id", "date"], how="left")
    test_df = test_df.merge(state_totals, on=["state_id", "date"], how="left")

    # Evaluate WAPE & RMSSE across all models
    models = {
        "Seasonal_Naive_Baseline": "pred_naive",
        "Exponential_Smoothing_Baseline": "pred_exp_smoothing",
        "LightGBM_HistGBDT": "pred_gbdt",
        "Deep_Neural_Network": "pred_neural",
        "Reconciled_Ensemble": "forecast_point",
    }

    accuracy_summary = {}
    for label, col in models.items():
        wape = calc_wape(y_test, test_df[col].values)
        rmsse = calc_rmsse_hierarchical(train_df, test_df, col)
        accuracy_summary[label] = {
            "WAPE": round(wape, 4),
            "WAPE_pct": f"{wape * 100:.2f}%",
            "RMSSE": round(rmsse, 4)
        }

    test_df.to_csv("data/forecast_results.csv", index=False)
    with open("reports/model_comparison.json", "w") as f:
        json.dump(accuracy_summary, f, indent=4)

    print("[STEP 3 COMPLETE] Model Evaluation (WAPE & RMSSE):")
    for m, scores in accuracy_summary.items():
        print(f"  - {m:32s} | WAPE: {scores['WAPE_pct']:7s} | RMSSE: {scores['RMSSE']}")

if __name__ == "__main__":
    run_forecasting_pipeline()
