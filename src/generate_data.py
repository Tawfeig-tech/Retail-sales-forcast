"""Generate synthetic daily retail sales data for the forecasting project.

Mirrors the structure of a typical retail POS export (store, department,
date, units sold, promo/holiday flags) so the notebook can demonstrate
time-series feature engineering without relying on a licensed dataset.
"""
import numpy as np
import pandas as pd

RNG_SEED = 42
N_STORES = 5
N_DEPTS = 4
START_DATE = "2021-01-01"
END_DATE = "2023-12-31"

US_HOLIDAYS = [
    "01-01", "02-14", "05-31", "07-04", "09-06", "11-25", "11-26", "12-24", "12-25",
]


def generate_raw_sales(seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    rows = []
    for store in range(1, N_STORES + 1):
        store_base = rng.uniform(80, 220)
        store_trend = rng.uniform(0.01, 0.05)
        for dept in range(1, N_DEPTS + 1):
            dept_factor = rng.uniform(0.6, 1.6)
            base_price = rng.uniform(8, 45)

            for i, date in enumerate(dates):
                day_of_year = date.dayofyear
                weekday = date.weekday()

                trend = store_base * (1 + store_trend * (i / 365))
                seasonality = 1 + 0.35 * np.sin(2 * np.pi * day_of_year / 365) + (
                    0.4 if date.month in (11, 12) else 0
                )
                weekend_boost = 1.25 if weekday >= 5 else 1.0

                is_holiday = int(date.strftime("%m-%d") in US_HOLIDAYS)
                holiday_boost = 1.5 if is_holiday else 1.0

                is_promo = int(rng.random() < 0.12)
                promo_boost = rng.uniform(1.3, 1.8) if is_promo else 1.0

                noise = rng.normal(1.0, 0.08)

                units_sold = max(
                    0,
                    trend
                    * dept_factor
                    * seasonality
                    * weekend_boost
                    * holiday_boost
                    * promo_boost
                    * noise,
                )
                price = base_price * rng.uniform(0.95, 1.05) * (0.85 if is_promo else 1.0)

                rows.append(
                    {
                        "date": date,
                        "store_id": store,
                        "dept_id": dept,
                        "units_sold": round(units_sold),
                        "unit_price": round(price, 2),
                        "is_promo": is_promo,
                        "is_holiday": is_holiday,
                        "temperature_c": round(
                            15 + 12 * np.sin(2 * np.pi * (day_of_year - 30) / 365)
                            + rng.normal(0, 2),
                            1,
                        ),
                    }
                )

    df = pd.DataFrame(rows)
    df["revenue"] = (df["units_sold"] * df["unit_price"]).round(2)
    return df.sort_values(["store_id", "dept_id", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    data = generate_raw_sales()
    data.to_csv("data/raw/retail_sales.csv", index=False)
    print(f"Wrote {len(data):,} rows to data/raw/retail_sales.csv")
