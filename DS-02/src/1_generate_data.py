import os
import numpy as np
import pandas as pd

def generate_m5_dataset():
    np.random.seed(42)
    os.makedirs("data", exist_ok=True)

    # 22 months of daily store-level sales
    dates = pd.date_range(start="2025-01-01", end="2026-10-31", freq="D")

    # M5 Hierarchy: (State, Store, Product SKU, Base Daily Demand, Regular Price)
    hierarchy = [
        ("CA", "CA_1_LosAngeles", "FOODS_001_OrganicMilk", 140, 4.99),
        ("CA", "CA_1_LosAngeles", "HOUSEHOLD_002_Detergent", 65, 14.99),
        ("CA", "CA_2_SanFrancisco", "FOODS_001_OrganicMilk", 125, 5.29),
        ("CA", "CA_2_SanFrancisco", "HOUSEHOLD_002_Detergent", 55, 15.49),
        ("TX", "TX_1_Austin", "FOODS_001_OrganicMilk", 110, 4.49),
        ("TX", "TX_1_Austin", "HOUSEHOLD_002_Detergent", 50, 13.99),
        ("NY", "NY_1_NewYork", "FOODS_001_OrganicMilk", 160, 5.49),
        ("NY", "NY_1_NewYork", "HOUSEHOLD_002_Detergent", 75, 15.99),
    ]

    # Major US retail holidays
    holidays = {
        "2025-01-01", "2025-02-14", "2025-05-26", "2025-07-04",
        "2025-09-01", "2025-11-27", "2025-11-28", "2025-12-25",
        "2026-01-01", "2026-02-14", "2026-05-25", "2026-07-04", "2026-09-07"
    }

    records = []
    for state, store, item, base_demand, base_price in hierarchy:
        for d in dates:
            date_str = d.strftime("%Y-%m-%d")
            is_holiday = 1 if date_str in holidays else 0
            is_weekend = 1 if d.dayofweek in [5, 6] else 0
            # Promotions run on the 1st and 15th of each month or on holidays
            is_promo = 1 if (d.day in [1, 15] or is_holiday == 1) else 0

            # 15% discount during promotions
            sell_price = round(base_price * (0.85 if is_promo else 1.0), 2)

            growth_trend = (d - dates[0]).days * 0.025
            weekend_boost = 1.25 if is_weekend else 1.0
            promo_boost = 1.35 if is_promo else 1.0
            holiday_boost = 1.40 if is_holiday else 1.0
            noise = np.random.normal(0, base_demand * 0.08)

            daily_sales = max(1, int(
                (base_demand + growth_trend) * weekend_boost * promo_boost * holiday_boost + noise
            ))

            records.append({
                "date": date_str,
                "state_id": state,
                "store_id": store,
                "item_id": item,
                "sales": daily_sales,
                "sell_price": sell_price,
                "is_promo": is_promo,
                "is_holiday": is_holiday
            })

    df = pd.DataFrame(records)
    df.to_csv("data/m5_raw_sales.csv", index=False)
    print(f"[STEP 1 COMPLETE] Created data/m5_raw_sales.csv ({len(df):,} rows across {df['state_id'].nunique()} states, {df['store_id'].nunique()} stores).")

if __name__ == "__main__":
    generate_m5_dataset()
