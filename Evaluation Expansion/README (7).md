# Expanded Model Evaluation — R², MAPE, MdAPE, and Price Bands

[← Project overview](../README.md) | [Previous: XGBoost modeling](../06_advanced_models/README.md)

**Notebook:** [`6_evaluation.ipynb`](6_evaluation.ipynb)  
**Assignment stage:** Week 8 — Evaluation Expansion (Phase 7 in this repository)  
**Prediction target:** `ClosePrice` in US dollars  
**Inputs:** `Completedf.csv` by default; optionally `Completedf_engineered.csv`  
**Outputs:** `metrics_summary.csv` (required), `price_band_metrics.csv` (additional), optional `evaluation_predictions.csv`

## Objective

This final phase provides a single place to re-train and evaluate the project’s models using **one chosen dataset**, a **common chronological test month**, and a consistent set of evaluation metrics. It addresses Week 8 by adding mean absolute percentage error (MAPE) and median absolute percentage error (MdAPE) beyond R², and comparing prediction quality across actual residential sale-price bands.

This is a **re-evaluation notebook**, not a spreadsheet of historical scores copied from Phases 3–6. The earlier saved metrics were obtained from different feature sets and in some cases different January 2024 populations; comparisons across those separate runs cannot establish which final model is best.

## Models evaluated

| Configuration | Implementation | Parameters used in this notebook |
| --- | --- | --- |
| Linear Regression | `sklearn.linear_model.LinearRegression` | Default intercept |
| Decision Tree | `DecisionTreeRegressor` | Depth 11 for base dataset or 20 for engineered dataset |
| Random Forest | `RandomForestRegressor` | 150 trees/depth 11 (base) or 100 trees/depth 20/min split 5 (engineered) |
| XGBoost (validation-selected settings) | `XGBRegressor` | Depth 3, learning rate 0.1, 200 estimators |
| XGBoost (log target) | `XGBRegressor` | Same settings; fit `log1p(ClosePrice)` and invert with `expm1` |
| XGBoost (optimized settings) | `XGBRegressor` | Depth 6, learning rate 0.05, 500 estimators |

The model families are the same as in the earlier phases, but the new evaluation may not reproduce the previous scores exactly because the input version, filtering, training window, and preprocessing must be consistent across models.

## Setup and required input

1. Open [`6_evaluation.ipynb`](6_evaluation.ipynb) with Python 3 in Jupyter or Google Colab.
2. Install packages if necessary: `pip install pandas numpy scikit-learn matplotlib xgboost jupyter`.
3. Obtain an authorized copy of the final cleaned data (`Completedf.csv`) or the engineered district dataset (`Completedf_engineered.csv`). **Neither is included in this public repository package.**
4. Put the CSV in the working directory used by the notebook, or change `DATA_FILE` to its full path. In Colab, the notebook can prompt for an upload when a file is missing.
5. Run every cell from top to bottom, and save the generated CSV files before ending the session.

In the settings cell, the default is:

```python
DATA_FILE = "Completedf.csv"
USE_SHARED_TRAINING_WINDOW = True
SHARED_TRAINING_MONTHS = 24
RANDOM_STATE = 42
MIN_PRICE_BAND_ROWS = 25
SAVE_PREDICTIONS = False
```

Change `DATA_FILE` to `"Completedf_engineered.csv"` to test the school-district features. This should be the *same exported file* for every model. The notebook detects whether school-district indicator columns are present and picks historical tree/forest settings accordingly.

## Methodology

### Shared chronological holdout

The last calendar month available in `CloseDate` is the test set. The preceding **24 calendar months** form the default common training window. Setting `USE_SHARED_TRAINING_WINDOW = False` instead uses model-specific window lengths drawn from earlier experiments; doing that changes training-set sizes and complicates direct algorithm comparisons.

Rows with invalid `CloseDate` values or nonpositive/missing `ClosePrice` are removed. The notebook checks that the test month has enough records and excludes obvious post-sale or target-derived features. It keeps only numeric and boolean predictors, removes infinite values, fits median imputers on **training data only**, and standardizes predictors only for Linear Regression. These choices improve consistency but do not establish that all upstream engineered fields were calculated without leakage.

### Evaluation metrics

| Metric | Interpretation | Preferred direction |
| --- | --- | --- |
| **R²** | Variation in prices explained relative to predicting the test-set average; may be negative | Higher |
| **MAE (USD)** | Mean absolute dollar error | Lower |
| **RMSE (USD)** | Square-root of mean squared dollar error; weighs large misses more heavily | Lower |
| **MAPE (%)** | Mean of absolute percentage errors for positive actual prices | Lower |
| **MdAPE (%)** | Median of absolute percentage errors; reflects typical relative error and is less sensitive to a few outliers | Lower |

For a sale with true price `$400,000` and predicted price `$360,000`, absolute percentage error is `|400000 - 360000| / 400000 × 100 = 10%`. Mean percentage error is sensitive to very low positive prices. The notebook excludes nonpositive target prices before evaluation but unusual low-dollar transactions still warrant review.

### Price-band analysis

Every test sale is assigned to a band using its **actual** closing price:

| Price band | Actual closing price |
| --- | --- |
| Under $300k | $0–$299,999.99 |
| $300k–$600k | $300,000–$599,999.99 |
| $600k–$1M | $600,000–$999,999.99 |
| $1M–$2M | $1,000,000–$1,999,999.99 |
| $2M+ | $2,000,000 or above |

For each model/band, the notebook records row count, median actual price, MAE, MAPE, MdAPE, percentage of predictions within 20% of true price, and an indicator for whether the band has at least 25 test sales. Groups with fewer than 25 records should not be used to make strong claims.

## Saved deliverables

### `metrics_summary.csv` — assignment deliverable

Automatically generated after model evaluation, **one row per model configuration**, with the following fields:

```text
Model,Data_File,Training_Months,Train_Rows,Test_Rows,Feature_Count,Train_Start,Test_Month,Hyperparameters,R2,MAE_USD,RMSE_USD,MAPE_Percent,MdAPE_Percent,MAPE_Valid_Rows,Negative_Predictions
```

The CSV is written using `metrics_summary.to_csv("metrics_summary.csv", index=False)`. The notebook's code computes actual values when run; the repository does not claim final metrics without the required dataset.

### `price_band_metrics.csv` — additional deliverable

Contains one row per observed price band/model combination, including counts, MAPE/MdAPE, MAE, and the proportion of predicted prices within ±20% of the actual value.

### `evaluation_predictions.csv` — optional

Set `SAVE_PREDICTIONS = True` to export record-level predictions and errors. These may be sensitive or restricted by your IDX/MLS data license; do not upload transaction-level exports publicly without permission.

## Plots and takeaways

The notebook includes an overall R² comparison, a MAPE/MdAPE comparison, and a price-band MdAPE heatmap. It prints the best model by R², the best model by MdAPE, and the lowest-/highest-error bands **based on results from the current dataset**. Since the supplied project materials do not include `Completedf.csv`, no real Week 8 scores or price-band findings are claimed in this README.

When interpreting the output, ask both whether a model generalizes across the overall market and whether its relative errors worsen in low-priced or high-priced homes. A good overall R² can coexist with poor performance in a particular price band.

## Limitations and recommended next steps

- **Do not reuse January 2024 as an untouched test set** after choosing configurations on earlier January scores. For a stronger study, select hyperparameters on past validation months and evaluate once on a new unseen future month.
- Comparison across the **original** and **engineered** feature sets requires separate reruns on a verified identical transaction population; the prior notebooks had different counts.
- The evaluation notebook uses stored engineered fields if present; the upstream feature-generation process still needs to be applied consistently to fresh properties.
- Non-numeric raw categories are excluded rather than re-encoded here. Use the prior cleaned/engineered dataset to make those features available in numeric form.
- Report the record counts in each price band to avoid overstating results in small groups.

## Completion checklist

- [x] Implement R², MAPE, and MdAPE for all model configurations.
- [x] Implement price-band error analysis and charts.
- [x] Export `metrics_summary.csv` and `price_band_metrics.csv` automatically when run.
- [ ] Run on the real authorized completed CSV and verify final outputs (data not included in this repository package).

[← Project overview](../README.md)
