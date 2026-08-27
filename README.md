# Assignment 1 — Predictive AI: Operational Forecasting

## Approach
- **Data:** 28 days (2026-08-01 to 2026-08-28) of daily operational data for one facility (`GMC-AHM-001`).
- **EDA:** loaded the CSV, checked dtypes/missing values (none), plotted `beds_occupied` over time, and computed mean-by-day-of-week to confirm a weekly pattern.
- **Features (6 total, no leakage):** `day_of_week`, `is_weekend`, `month`, `lag_1`, `lag_7`, `rolling_mean_7` (rolling mean uses `shift(1)` so today's value is never used to predict today — verified against a `_LEAKY` version for comparison).
- **Split:** time-based, no shuffling — first 70% of dates = train, next 15% = validation, last 15% = test (14 train / 3 val / 4 test rows per metric).
- **Baseline:** seasonal naive (`lag_7` — same day last week).
- **ML model:** Linear Regression, trained once per metric in a single loop over the 5 targets (lab tests, ED arrivals, OPD visits, beds occupied, ambulance calls).
- **Forecast:** 7 days beyond the last date (2026-08-29 → 2026-09-04), generated recursively via `forecast_target()` — each day's prediction feeds back in as `lag_1`/`lag_7`/`rolling_mean_7` for the next day.

## Results (test set, MAE / RMSE / R²)

| Metric           | Baseline MAE | Baseline RMSE | ML MAE | ML RMSE | ML R² | ML beat baseline? |
|------------------|-------------:|--------------:|-------:|--------:|------:|:------------------:|
| lab_tests        | 21.25        | 21.36         | 22.68  | 23.88   | 0.933 | No                 |
| ed_arrivals      | 4.75         | 4.97          | 4.19   | 4.41    | 0.967 | Yes                |
| opd_visits       | 20.00        | 21.21         | 7.59   | 8.80    | 0.995 | Yes                |
| beds_occupied    | 2.75         | 2.87          | 2.04   | 2.21    | 0.980 | Yes                |
| ambulance_calls  | 0.75         | 0.87          | 0.37   | 0.46    | 0.993 | Yes                |

**One sentence per metric:**
- **lab_tests:** ML did *not* beat the seasonal-naive baseline — a holiday-driven dip inside the tiny (4-row) test window skewed the comparison against the linear model, even though its R² is still reasonably high.
- **ed_arrivals:** ML beat the baseline, picking up the weekday/weekend swing slightly better than repeating last week's value.
- **opd_visits:** ML beat the baseline by a wide margin — OPD has the sharpest weekend/holiday drop, which the calendar features capture directly (R² = 0.995).
- **beds_occupied:** ML beat the baseline modestly, benefiting from the smoothing effect of `rolling_mean_7`.
- **ambulance_calls:** ML beat the baseline — this is the lowest-variance metric, so both models are close in absolute terms, but ML edges it out.

**Note on data size:** with only 28 days per facility and a 70/15/15 split, the test set is just 4 rows per metric, so these MAE/RMSE/R² numbers should be read as directional, not statistically robust. More historical days (the brief recommends ≥30, ideally several months) would give a much more reliable baseline-vs-ML comparison.

## Capacity alert
No breach predicted in the next 7 days — forecast `beds_occupied` peaks at 442.6 against a 500-bed capacity.

## Files
- `forcast_1.ipynb` — full pipeline, runs end-to-end (Part A → D, plus 7-day forecast and capacity alerts)
- `forecasts_next7.csv` — next-7-day forecast per metric
- `capacity_alerts.csv` — bed-occupancy forecast vs. 500-bed capacity