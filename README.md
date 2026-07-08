# Retail Sales Forecasting

Data Science Portfolio

An end-to-end regression/time-series project that predicts daily unit sales
per store and department for a retail chain, using historical sales,
promotions, holidays, and weather. The work demonstrates the full workflow
from exploratory analysis through time-aware feature engineering to model
evaluation on a held-out future period.

## Project structure

```
retail-sales-forecasting/
├── data/
│   ├── raw/            # generated source CSV
│   └── processed/      # cleaned/engineered data
├── notebooks/
│   └── retail_sales_forecasting.ipynb
├── src/
│   └── generate_data.py
├── requirements.txt
└── README.md
```

## Dataset

The project uses synthetic daily sales data for 5 stores x 4 departments
over 3 years (~21,900 rows), generated via `src/generate_data.py`. Sales
are simulated from realistic business factors — store-level baseline
demand, yearly growth trend, seasonality, weekday/weekend effects, US
holidays, promotional pricing, and local temperature — plus random noise.
This synthetic approach avoids licensing concerns while remaining fully
reproducible.

## Methodology

- **EDA & visualization**: missing-value checks, daily/monthly revenue
  trends, seasonality by month, promo/holiday lift, correlation heatmap
- **Feature engineering**: calendar features (day of week, month, weekend
  flag) plus per-series lag features (`lag_7`, `lag_14`) and rolling means
  (`rolling_mean_7`, `rolling_mean_28`), computed within each store/department
  group to avoid leakage across series
- **Train/test split**: time-based — the last 90 days held out as test data
  (no random shuffling, since this is a forecasting problem)
- **Modeling**: Linear Regression vs. Random Forest vs. Gradient Boosting
- **Evaluation**: MAE, RMSE, R², actual-vs-predicted plots, feature importance

## Key results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 27.30 | 40.47 | 0.865 |
| Random Forest | 20.25 | 30.83 | 0.922 |
| **Gradient Boosting** | **18.98** | **28.07** | **0.935** |

Gradient Boosting was the best performer on the 90-day held-out test set.
Recent history (`lag_7`, `rolling_mean_7`/`rolling_mean_28`) dominated
feature importance, as expected for retail demand — recent sales are the
strongest signal for near-term forecasts. Promotions and holidays each
produced a clear, learnable lift in units sold (promo: +55%, holiday: +67%
vs. baseline average).

## Setup

```bash
pip install -r requirements.txt
python src/generate_data.py          # regenerates data/raw/retail_sales.csv
jupyter notebook notebooks/retail_sales_forecasting.ipynb
```
