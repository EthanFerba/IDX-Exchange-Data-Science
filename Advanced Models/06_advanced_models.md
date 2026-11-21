# Advanced Regression Models — XGBoost

**Notebook:** [`GxBoost.ipynb`](../GxBoost.ipynb)  
**Assignment stage:** Week 7 — Advanced Models (Phase 6 of this repository)  
**Input:** `Completedf_engineered.csv` (prepared IDX/CRMLS real-estate data with engineered school-district features)  
**Prediction target:** `ClosePrice` (US dollars)  
**Evaluation period:** January 1–31, 2024, as shown in the notebook's saved output  
**Scope:** Training, light hyperparameter tuning, and performance evaluation of XGBoost regression models

## 1. Project objective

The purpose of this phase is to determine whether gradient-boosted decision trees can provide accurate estimates of California residential single-family property closing prices. Earlier notebooks explored the IDX data, cleaned and prepared property records, trained a Linear Regression baseline, compared Decision Tree and Random Forest regressors, and added property- and school-district-based features.

This notebook introduces **XGBoost (`XGBRegressor`)** and tests how training-history length, tree depth, learning rate, and number of boosting rounds affect performance. It uses a chronological validation period to choose an initial configuration, evaluates that configuration on a held-out month, and then explores alternative ways of modeling the target and a deeper XGBoost model.

### Week 7 assignment checklist

| Requirement | Evidence in the notebook |
| --- | --- |
| Try gradient boosting, such as XGBoost or LightGBM | **Completed:** uses `XGBRegressor`; LightGBM is not implemented. |
| Light tuning of depth | **Completed:** `max_depth` = 3 or 5 in the initial search; 6 in a later manually selected experiment. |
| Light tuning of learning rate | **Completed:** 0.05 and 0.1. |
| Light tuning of `n_estimators` | **Completed:** 100 and 200, plus 500 in the later experiment. |
| Deliver notebook with test metrics | **Completed in the supplied notebook:** test R², MAE, MAPE, and median absolute percentage error are printed; the assignment's requested filename is `05_advanced_models.ipynb`, while the supplied file is `GxBoost.ipynb`. |

## 2. Input data and dependencies

The notebook begins with:

```python
rdf = pd.read_csv("Completedf_engineered.csv")
```

The input should be an **authorized prepared property dataset** with the target `ClosePrice`, a parseable `CloseDate`, numerical property features, and the geographic/engineered features prepared in the preceding phase. The saved first-cell preview reports **413 columns**, including `OfficialSchoolDistrict_...` indicator features. The notebook does **not** print a complete input row count.

> **Reproducibility warning:** The preceding feature-engineering notebook displays an engineered dataframe with **415 columns** and **7,069 January 2024 test transactions**, while this notebook's saved input preview shows **413 columns** and its split reports **3,238 January 2024 test transactions**. Its initial load also produces a pandas mixed-dtype warning for column 27, and its date split detects one missing `CloseDate`. Consequently, the saved `Completedf_engineered.csv` cannot be assumed identical to the exact dataframe from Phase 5. Reconcile the input export/version and those counts before claiming a controlled comparison with earlier models.

### Key Python libraries

- **pandas / NumPy:** data loading, transformations, missing/infinite-value handling, and result tables.
- **XGBoost:** `XGBRegressor`, which fits an ensemble of boosted decision trees.
- **scikit-learn:** R², mean absolute error, and mean squared error imports (RMSE/MSE are imported but not reported in this notebook).
- **Matplotlib:** scatter plot of actual vs. predicted closing prices.
- **itertools.product:** enumerates hyperparameter combinations.
- **Google Colab:** notebook originally includes `from google.colab import files`; this import is not required for model training and should be removed or guarded for local execution.

## 3. Variables excluded from model inputs

The notebook assembles a `columns_to_drop` list and keeps only entries present in the loaded dataframe:

```python
columns_to_drop = [
    "ClosePrice", "Log_ClosePrice", "CloseDate", "PurchaseContractDate",
    "DaysOnMarket", "ListingContractDate", "ListOfficeName",
    "ListAgentFullName", "PropertyType", "PropertySubType",
    "ListPrice", "Log_ListPrice"
]
columns_to_drop = [col for col in columns_to_drop if col in rdf.columns]
```

`ClosePrice` is the target and must not also be a predictor. Other excluded columns are identifiers, dates, administrative labels, or fields that are unavailable at prediction time or could reveal information too close to the final transaction. **This is a sensible attempt to reduce leakage, but it is not a guarantee**: all retained engineered and geographic inputs should be reviewed for availability at the specific intended prediction date.

## 4. Chronological train/test split

The `create_temporal_split(df, training_months_x)` function parses `CloseDate`, locates the first day of the most recent calendar month, and holds out that entire month as the final test set. The `X` calendar months immediately before that month form the training pool. Records with missing dates are not included in either selection.

The notebook compares `X` values of **3, 6, 12, and 24 months**:

| Training history | Training dates | Training-pool rows | Test dates | Test rows |
| --- | --- | ---: | --- | ---: |
| 3 months | 2023-10-01 to 2023-12-31 | 24,044 | 2024-01-01 to 2024-01-31 | 3,238 |
| 6 months | 2023-07-01 to 2023-12-31 | 47,079 | Same | 3,238 |
| 12 months | 2023-01-01 to 2023-12-31 | 54,273 | Same | 3,238 |
| 24 months | 2022-01-01 to 2023-12-31 | 58,713 | Same | 3,238 |

The saved output reports **one row with an unparseable/missing `CloseDate`**. All four final test sets cover January 2024, but they are smaller than in the earlier notebooks, so score comparisons across phases require reconciliation.

### Validation split inside each candidate training pool

The hyperparameter-search cell reserves the **most recent month inside the training pool** as a validation set; the remaining earlier months are used for fitting. With the saved date boundaries, this corresponds to the final training month (December 2023). This lets the notebook choose initial hyperparameters **without directly using January 2024 for that selection**.

After selecting a configuration, the notebook refits it on the complete selected training pool (including the month previously used for validation) and evaluates it once on January 2024.

## 5. Feature preparation for XGBoost

For each model-training loop, the notebook:

1. Separates `ClosePrice` from input features and drops the excluded columns.
2. Removes duplicate dataframe column names before and after encoding.
3. Converts remaining categorical fields to numeric indicators with `pd.get_dummies(..., dtype="int8")`.
4. Reindexes the validation or test inputs to match the model-training columns, inserting zeros for missing categories.
5. Replaces positive/negative infinity with `NaN`; XGBoost can route missing numeric values.

These steps address a practical issue from previous modeling work: duplicate feature names and inconsistent one-hot columns across data splits. A known limitation is that separately converting categories and reindexing cannot distinguish a truly unseen category from the all-zero reference state. A production pipeline should preserve a consistent fitted preprocessing schema and explicitly handle unknown categories.

## 6. Initial XGBoost hyperparameter search

The notebook defines:

| Parameter | Values tested | Purpose |
| --- | --- | --- |
| Training window | `3, 6, 12, 24` months | Tests the amount of historical information used. |
| `max_depth` | `3, 5` | Limits decision-tree complexity. |
| `learning_rate` | `0.05, 0.1` | Controls the contribution of each successive tree. |
| `n_estimators` | `100, 200` | Sets the number of boosting rounds/trees. |

This is **4 × 2 × 2 × 2 = 32** configurations. All fitted models use `objective="reg:squarederror"`, `tree_method="hist"`, `random_state=42`, and `n_jobs=-1`.

Each candidate is ranked by **validation R²**, and validation MAE is recorded as a second metric. The saved best-ranked configuration is:

| Selected parameter | Saved value |
| --- | ---: |
| Training history | 24 months |
| Maximum depth | 3 |
| Learning rate | 0.10 |
| Estimators | 200 |
| Validation R² | **0.762401** |
| Validation MAE | **$250,211.29** |

Validation R² was the selection criterion; a different model might be preferred if minimizing absolute dollar error were the primary business objective.

## 7. Final test results: selected model

The notebook refits the selected configuration on **58,713 training records from January 2022 through December 2023** and evaluates the **3,238 January 2024 records**. It reports **no duplicate feature columns** in either the final training or testing matrix.

| Metric | Result |
| --- | ---: |
| Test R² | **0.7910** |
| Test MAE | **$223,526.71** |
| Test MAPE | **23.55%** |
| Test MdAPE | **15.68%** |

**Interpretation:** An R² of 0.791 indicates that the model explains approximately 79.1% of the variation in closing prices in this specific test sample, relative to predicting the test-set mean. It does **not** mean that 79.1% of individual predictions are correct. MAE is the average absolute error measured in dollars, while MAPE and MdAPE summarize relative errors.

The saved scatter plot places actual closing prices on the horizontal axis and predicted closing prices on the vertical axis. It offers a quick visual inspection of agreement and outliers, though the current plot would be easier to interpret with axis labels, a diagonal perfect-prediction reference line, and a title.

## 8. Experiment: log-transformed target

The notebook fits a second XGBoost model with the same selected hyperparameters, but trains it to predict:

```python
y_train_log = np.log1p(y_train)
```

Predictions are converted back to dollars with `np.expm1()` before computing metrics. The comparison is therefore performed on the **same original dollar scale**:

| Target approach | Test R² | Test MAE |
| --- | ---: | ---: |
| Original target (`ClosePrice`) | 0.7910 | $223,526.71 |
| Log-transformed target, back-transformed | 0.7320 | $198,040.98 |

The log target **reduces MAE by about $25,486**, but **lowers R² by 0.059**. This is a tradeoff rather than a clear win for either metric. A log target changes the effective emphasis of the squared-error training objective and may improve average dollar error while worsening errors on some higher-price properties. Confirm that all target values are valid for `log1p`, and inspect errors by price range before selecting this approach.

## 9. Experiment: deeper XGBoost model

The notebook then manually fits a new configuration using all 24 months of training data:

```python
optimized_model = XGBRegressor(
    max_depth=6,
    learning_rate=0.05,
    n_estimators=500,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)
```

Its saved results are:

| Configuration | Test R² | Test MAE | Test MAPE | Test MdAPE |
| --- | ---: | ---: | ---: | ---: |
| Validation-selected XGBoost | 0.7910 | $223,526.71 | 23.55% | 15.68% |
| **Deeper XGBoost (manual experiment)** | **0.8189** | **$174,059.81** | **16.96%** | **11.49%** |

Compared with the validation-selected model, the deeper model improves observed test R² by **0.0279**, reduces test MAE by **$49,466.90**, and reduces MAPE by **6.59 percentage points**. The MdAPE improvement is **4.19 percentage points**.

**Important experimental distinction:** The deeper model is evaluated using the already-inspected January 2024 test set; it was *not* among the original 32 validation-selected configurations. Because this later configuration appears to be assessed after examining test results, its score is best treated as an **exploratory test result**, not an unbiased final performance claim. To establish a trustworthy final comparison, select this configuration using validation data or rolling time-based folds, then assess it on a separate untouched future month.

## 10. How this compares with previous phases

Earlier saved notebooks reported the following best R² values:

| Earlier model | Notebook | Reported best R² |
| --- | --- | ---: |
| Linear Regression | `First_Modelp2.ipynb` | 0.530972 |
| Decision Tree | `Base_Compare (1).ipynb` | 0.673584 |
| Random Forest, original features | `Base_Compare (1).ipynb` | 0.778343 |
| Random Forest, engineered features | `Feature_eng.ipynb` | **0.825943** |
| XGBoost, deeper manual experiment | `GxBoost.ipynb` | **0.8189** |

The engineered Random Forest's reported R² is numerically slightly higher than XGBoost's **0.8189**. **Do not interpret these numbers as an apples-to-apples leaderboard:** the older notebooks evaluated **7,069 January 2024 properties**, but this notebook evaluates **3,238**, and its loaded engineered file preview has **413 columns versus 415** in the Phase 5 output. The input datasets or exports must be reconciled, and both models must be rerun on identical train/validation/test records before concluding which algorithm generalizes better.

## 11. Model behavior: strengths, weaknesses, and interpretation

**Strengths observed or motivated by the approach**

- XGBoost can learn nonlinear relationships and interactions among living area, property condition/age, and location without requiring a manually specified linear equation.
- The initial search evaluates hyperparameters against a chronological validation month rather than selecting them using the final month directly.
- The notebook records both R² and dollar/percentage error metrics, highlighting that different objectives can rank models differently.
- The deeper experiment reduces measured error considerably relative to the initial XGBoost configuration on the saved test subset.

**Limitations and open questions**

- A high R² does not mean individual properties will be priced accurately. Large property-price differences and high-cost outliers may still be important to business users.
- MAPE can be unstable when actual values are close to zero. The helper function excludes zero-valued actual prices but does not investigate unusually small sales prices.
- The extra 500-tree configuration may be more computationally demanding than the 100–200-tree search configurations.
- A single January test month cannot establish robustness to seasonality, housing-market shifts, or changing geography.
- The school-district mapping uses **2024–25 boundaries** for earlier property transactions; historic district assignments may not be exact.
- Some upstream cleaning or imputations may have been performed before the temporal split. Learned transformations should be fit to training data only.
- The included notebook does not provide feature-importance, SHAP, per-region errors, or residual analyses; these would help explain model behavior.

## 12. Reproduce the notebook

### Prerequisites

- Python 3 and Jupyter Notebook or Google Colab.
- Libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`.
- Access to an authorized IDX/CRMLS property dataset prepared using the earlier cleaning and feature-engineering phases.

```bash
pip install pandas numpy matplotlib scikit-learn xgboost jupyter
```

### Reproduction steps

1. Run the earlier cleaning and feature-engineering notebooks using properly authorized input data and the appropriate school-district boundary file.
2. In `Feature_eng.ipynb`, **after feature creation and one-hot encoding**, explicitly save the intended dataframe:

   ```python
   rdf_features.to_csv("Completedf_engineered.csv", index=False)
   ```

   **Do not assume this newly exported version will reproduce the notebook's saved metrics** without first comparing the dataframe shape, columns, date range, and January 2024 row counts with the version originally used in `GxBoost.ipynb`.
3. Place `Completedf_engineered.csv` in the notebook's working directory, or update its `pd.read_csv(...)` path.
4. Run notebook cells in order. Verify `ClosePrice` numeric type, `CloseDate` conversion, duplicate column counts, and the printed temporal splits.
5. Review the 32-row validation leaderboard, the selected model's January 2024 R², and the log-target/500-tree experiment metrics.
6. Save results and generated figures with the exact input-data version, package versions, and random seed used.

### Suggested data-version verification

Before running training, add a cell like the following:

```python
print("Input shape:", rdf.shape)
print("Duplicate columns:", rdf.columns.duplicated().sum())
print("Missing CloseDate:", pd.to_datetime(rdf["CloseDate"], errors="coerce").isna().sum())
print("Date range:", pd.to_datetime(rdf["CloseDate"], errors="coerce").agg(["min", "max"]))
print("Target missing:", pd.to_numeric(rdf["ClosePrice"], errors="coerce").isna().sum())
```

To compare reliably with the earlier Random Forest, run both algorithms on **the same exported dataset, identical chronological splits, and identical prediction target**.

**Data-access note:** Real-estate listing and sales extracts may be subject to IDX/MLS licensing and distribution restrictions. Do not publish raw or derived property-level records without authorization; documentation and code can be shared separately.

## 13. Suggested next improvements

For a stronger final modeling report, standardize the input-data export between Phases 5 and 6, rerun the comparison using identical records, tune all final candidates on rolling or expanding chronological validation periods, and reserve a new untouched month for the final comparison. Report R², MAE, and RMSE together, examine errors by price range and geography, and add feature-importance or SHAP diagnostics. If model deployment is a future goal, package preprocessing and prediction in one reproducible pipeline that can transform new incoming listings consistently.

---

**Summary:** The notebook meets the Week 7 XGBoost and light hyperparameter-tuning requirements. It reports a validation-selected model with **R² = 0.7910** on January 2024 and a later deeper experiment with **R² = 0.8189, MAE = $174,059.81** on that notebook's **3,238-record** test sample. The saved dataset differs in visible ways from Phase 5, and the deeper model was evaluated on an already-used test set; both qualifications should accompany claims of improvement.
