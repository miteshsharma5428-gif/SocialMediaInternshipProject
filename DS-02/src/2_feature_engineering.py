import pandas as pd

def build_features():
    df = pd.read_csv("data/m5_raw_sales.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["state_id", "store_id", "item_id", "date"]).reset_index(drop=True)

    grouped = df.groupby(["store_id", "item_id"])["sales"]

    # 1. Historical lag features
    df["lag_7"] = grouped.shift(7)
    df["lag_14"] = grouped.shift(14)
    df["lag_28"] = grouped.shift(28)

    # 2. Shifted rolling averages & standard deviation (prevents data leakage)
    df["rolling_mean_7"] = grouped.transform(lambda x: x.shift(1).rolling(window=7).mean())
    df["rolling_std_7"] = grouped.transform(lambda x: x.shift(1).rolling(window=7).std())
    df["rolling_mean_28"] = grouped.transform(lambda x: x.shift(1).rolling(window=28).mean())

    # 3. Price discount percentage
    max_price = df.groupby(["store_id", "item_id"])["sell_price"].transform("max")
    df["price_discount_pct"] = ((max_price - df["sell_price"]) / max_price).round(3)

    # 4. Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["date"].dt.month

    # Remove first 28 warmup days where lags are empty
    df_features = df.dropna().reset_index(drop=True)
    df_features.to_csv("data/m5_features.csv", index=False)
    print(f"[STEP 2 COMPLETE] Created data/m5_features.csv ({len(df_features):,} rows, {len(df_features.columns)} features).")

if __name__ == "__main__":
    build_features()
