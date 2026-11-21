# Model Comparison — Linear Regression, Decision Tree, and Random Forest

[← Project overview](../README.md) | [Next: Feature Engineering and School Districts](../05_feature_engineering/README.md)


**Notebook:** [`Base_Compare (1).ipynb`](Base_Compare%20%281%29.ipynb)  
**Project phase:** Phase 4 — Model comparison and model-behavior analysis  
**Source data:** Prepared IDX / CRMLS residential single-family sales records from [Phase 2](../02_data_cleaning/README.md)  
**Prediction target:** `ClosePrice` (sale price in dollars)  
**Evaluation metric:** Test-set R² (coefficient of determination)  
**Comparison month:** January 2024

## 1. Project objective

This notebook builds on the [Linear Regression baseline](../03_baseline_model/README.md) by testing two nonlinear approaches: `DecisionTreeRegressor` and `RandomForestRegressor`. The goal is to determine whether these methods better capture relationships between property characteristics and closing prices, compare their R² scores with the baseline, and describe how their behavior differs.

The assigned objectives are to:

1. Train **Decision Tree** and **Random Forest** regression models.
2. Compare their **test-set R²** scores against the existing Linear Regression baseline.
3. Document the **strengths and weaknesses** of the models.
4. Submit a Jupyter notebook called `04_model_comparison.ipynb` (the supplied file is currently called `Base_Compare (1).ipynb`).

This phase explores a small grid of hyperparameters and training-history lengths. The comparison is **preliminary** because the best configurations were selected using the same month described as the test set; see [Limitations](#7-limitations-and-interpretation).

## 2. Input data and environment

The notebook loads the data using:

```python
rdf = pd.read_csv("Completedf.csv")
```

The saved output reports **65,782 records and 46 columns**, with pandas reporting **0% null values** for each column. The preceding preparation phase filtered to residential, single-family properties, handled missing values, created selected indicator features, and added log-transformed numeric columns. Phase 4 does **not** repeat those cleaning operations.

**Reproducibility prerequisite:** The supplied Phase 2 notebook did not include the CSV export used in Phases 3 and 4. After confirming the finalized dataframe is `rdf` in that notebook, export it:

```python
rdf.to_csv("Completedf.csv", index=False)
```

Put the CSV where this notebook can read it or change the input path. **IDX/MLS extracts may be licensed. Do not publish raw or cleaned transaction-level data unless redistribution is permitted.** A public repository may provide code and documentation while requiring authorized users to supply their own eligible source data.

### Dependencies

The notebook imports pandas, NumPy, Matplotlib, and the following components from scikit-learn: `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, and `r2_score`.

A basic Python 3 / Jupyter setup is:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

Specific package versions are not recorded in the notebook. The Random Forest sweep fits **80 forests** (four training windows × five estimator counts × four depths), so it is notably more computationally intensive than the baseline.

## 3. Shared modeling setup

### Temporal train/test split

All three regressors use the same `create_temporal_split(df, training_months_x)` function. It identifies the **most recent full closing month represented in the dataset** as the test period, then selects the preceding *X* calendar months for training, without overlap. In the saved notebook, the test period is **January 1–31, 2024**, containing **7,069 sales**.

| Training window | Training period | Training rows | Test period | Test rows |
| --- | --- | ---: | --- | ---: |
| 3 months | Oct–Dec 2023 | 24,044 | Jan 2024 | 7,069 |
| 6 months | Jul–Dec 2023 | 47,079 | Jan 2024 | 7,069 |
| 12 months | Jan–Dec 2023 | 54,273 | Jan 2024 | 7,069 |
| 24 months | Jan 2022–Dec 2023 | 58,713 | Jan 2024 | 7,069 |

The row counts grow unevenly with the calendar window. This notebook does not establish why older months contribute fewer records; possibilities such as source coverage and filtering require separate investigation.

### Target selection and predictor exclusions

The models predict **raw `ClosePrice`**, not `Log_ClosePrice`. The notebook excludes:

```python
columns_to_drop = [
    "ClosePrice", "Log_ClosePrice", "CloseDate",
    "PurchaseContractDate", "DaysOnMarket", "ListingContractDate",
    "ListOfficeName", "ListAgentFullName", "PropertyType",
    "PropertySubType", "ListPrice", "Log_ListPrice"
]
```

This prevents direct target leakage through `Log_ClosePrice` and excludes several variables that may not be appropriate at the intended prediction moment. The exact availability of every remaining predictor at prediction time should still be reviewed separately.

For each split, the notebook selects only numerical columns and aligns the test columns to the training columns:

```python
X_train = train_set.drop(columns=columns_to_drop, errors="ignore")
X_test = test_set.drop(columns=columns_to_drop, errors="ignore")
X_train = X_train.select_dtypes(include="number")
X_test = X_test[X_train.columns]
```

**Important:** Remaining text/categorical columns are ignored rather than encoded here. The earlier flooring indicators are numeric and can be included. Numeric-coded identifiers or categories, if present, should be checked for appropriate interpretation.

## 4. Models and experiments

### A. Linear Regression — baseline

The first experiment re-runs `LinearRegression()` for all four window lengths, following the Phase 3 methodology. Linear Regression estimates a weighted additive relationship between the selected numeric property attributes and closing price. It is straightforward to reproduce and interpret, but may miss nonlinear effects and interactions.

| Window | Saved test R² |
| --- | ---: |
| 3 months | 0.526261 |
| **6 months** | **0.530972** |
| 12 months | 0.529234 |
| 24 months | 0.526381 |

The best recorded baseline is **R² = 0.530972** using six training months. The notebook also includes an actual-vs-predicted scatterplot with a diagonal reference line. **That plot is produced after the baseline loop**, so it reflects the last (24-month) model, not the highest-scoring six-month model.

### B. Decision Tree Regressor

`DecisionTreeRegressor` recursively divides the feature space into groups and predicts closing prices within terminal regions. The notebook tries four maximum tree depths (`3`, `5`, `7`, and `11`) for each training window, with `random_state=42`—**16 configurations** in total.

| Window | Best depth in that window | Best saved test R² |
| --- | ---: | ---: |
| 3 months | 7 | 0.612165 |
| **6 months** | **11** | **0.673584** |
| 12 months | 7 | 0.622406 |
| 24 months | 11 | 0.628956 |

The overall best Decision Tree configuration is **six months of training and maximum depth 11**, achieving **R² = 0.673584**.

The deeper tree does **not** always win: depth 7 outperforms depth 11 for the three-month and 12-month windows. This is a useful caution against assuming that a deeper tree will consistently perform better.

### C. Random Forest Regressor

`RandomForestRegressor` combines predictions from many trees trained with randomized sampling and feature selection. The notebook compares:

- **Training windows:** 3, 6, 12, 24 months.
- **Number of trees (`n_estimators`):** 50, 100, 150, 200, 300.
- **Maximum depths (`max_depth`):** 3, 5, 7, 11.
- **Random seed:** `random_state=42`.

This is **80 attempted configurations** in the saved run. The best results within each training window, reconstructed from the printed run summaries, are:

| Window | Best depth | Best number of trees | Approx. test R² |
| --- | ---: | ---: | ---: |
| 3 months | 11 | 300 | 0.7472 |
| 6 months | 11 | 100 | 0.7621 |
| 12 months | 11 | 200 | 0.7775 |
| **24 months** | **11** | **150** | **0.778343** |

The strongest recorded Random Forest is **R² = 0.778343**, trained with a **24-month history, 150 estimators, and maximum depth 11**. The exact R² comes from the notebook's saved ranked output; the estimator count of **150** comes from the corresponding printed loop result. The saved final comparison table labels that model as **50 estimators**, but that label is incorrect because of the recording bug described below.

## 5. Model comparison and interpretation

| Model | Best training window | Selected settings | Highest saved test R² |
| --- | ---: | --- | ---: |
| Linear Regression | 6 months | Default `LinearRegression()` | 0.530972 |
| Decision Tree | 6 months | `max_depth=11` | 0.673584 |
| **Random Forest** | **24 months** | **`n_estimators=150`, `max_depth=11`** | **0.778343** |

Within this January 2024 comparison, the tree-based models outperform the linear baseline. The strongest Random Forest improves the reported R² by about **0.2474** absolute points over Linear Regression. Its R² indicates that, on this selected test month, the model accounts for roughly **77.8% of the variance in observed closing prices** relative to predicting the test-set mean. It does **not** mean that individual house-price estimates are, on average, 77.8% accurate.

These saved scores support the conclusion that nonlinear modeling is promising. They do **not** yet demonstrate that the Random Forest will outperform on every future month or that 24 months is the universally best training history.

### Strengths and weaknesses

| Model | Strengths | Weaknesses / cautions |
| --- | --- | --- |
| **Linear Regression** | Fast, simple baseline; interpretable additive relationships; low computational cost | Cannot automatically capture complex nonlinear patterns/interactions; sensitive to unusual observations and correlated predictors |
| **Decision Tree** | Learns nonlinear thresholds and interactions; no feature scaling required; one tree can be inspected | May overfit or become unstable; performance changes with depth and training window; can generalize poorly to unseen ranges |
| **Random Forest** | Combines many trees to reduce single-tree instability; handles nonlinear relationships; best saved test R² of the three | More expensive to train; less directly interpretable; performance depends on tuning and representative past data; does not extrapolate well beyond observed response ranges |

The strength/weakness descriptions combine **observed results in this notebook** with **general algorithm properties**; the notebook does not measure runtime, training R², feature importance, or stability across different test months.

## 6. Code issue that must be corrected for reproducibility

### Incorrect destination list and estimator variable

In the supplied Random Forest loop, the results are appended like this:

```python
dt_results.append({
    "N value": n_val,
    "Training_Months": months,
    "Max_Depth": depth,
    "Training_Rows": len(X_train),
    "Test_R2": test_r2
})
```

There are two related bugs:

1. The list should be **`rf_results`**, not `dt_results`. Otherwise Decision Tree and Random Forest records are mixed together and `rf_results` remains empty.
2. The estimator variable should be the inner-loop variable **`n`**, not **`n_val`**. A clean run may raise `NameError` because `n_val` is not defined. In the saved output, all Random Forest rows show `N value = 50`, even though the loop evaluated 50, 100, 150, 200, and 300 trees. This is why the final table incorrectly reports 50 trees for the strongest model.

**Minimal change inside the existing forest loop:**

```python
rf_results.append({
    "N_Estimators": n,
    "Training_Months": months,
    "Max_Depth": depth,
    "Training_Rows": len(X_train),
    "Testing_Rows": len(X_test),
    "Test_R2": test_r2
})
```

After the loop, create the Random Forest result table from its own list:

```python
rf_results_df = (
    pd.DataFrame(rf_results)
    .sort_values("Test_R2", ascending=False)
    .reset_index(drop=True)
)
display(rf_results_df.head(10))
```

Then update the model comparison so it reads each model's results from the correct collection:

```python
best_lr = max(baseline_results, key=lambda x: x["Test_R2"])
best_dt = max(dt_results, key=lambda x: x["Test_R2"])
best_rf = max(rf_results, key=lambda x: x["Test_R2"])

comparison_df = pd.DataFrame([
    {
        "Model": "Linear Regression",
        "Training_Months": best_lr["Training_Months"],
        "Test_R2": best_lr["Test_R2"],
        "Settings": "Default"
    },
    {
        "Model": "Decision Tree",
        "Training_Months": best_dt["Training_Months"],
        "Test_R2": best_dt["Test_R2"],
        "Settings": f"max_depth={best_dt['Max_Depth']}"
    },
    {
        "Model": "Random Forest",
        "Training_Months": best_rf["Training_Months"],
        "Test_R2": best_rf["Test_R2"],
        "Settings": (
            f"max_depth={best_rf['Max_Depth']}, "
            f"n_estimators={best_rf['N_Estimators']}"
        )
    }
])
display(comparison_df)
```

Replace the original Random Forest result append, the incorrect DataFrame assembly, and the final comparison-selection code with the corrected versions above. **Rerun the notebook from a fresh kernel in order**, rather than relying on variables left over from earlier Colab executions. The supplied notebook has not been modified; these are recommended edits.

## 7. Limitations and interpretation

- **Model selection on the test set:** Hyperparameters (`max_depth`, `n_estimators`) and training-window length are chosen by looking at January 2024 scores, and January 2024 is also called the test set. This creates optimistic selection bias. For a more reliable estimate, tune on **earlier months** using chronological validation and reserve a later untouched month for final testing.
- **Single test month:** Results may change during different market conditions or seasons. Repeat evaluation across several successive held-out months (rolling-origin validation).
- **Upstream preprocessing:** Some missing-value imputation and feature preparation occurred before the temporal split in the earlier notebook. Any learned statistics (such as medians) should be calculated on the training period only and then applied to validation/test records to reduce leakage.
- **Incomplete feature coverage:** This notebook selects numeric predictors only. Any potentially useful string fields not encoded upstream are omitted; no systematic feature-importance analysis is included.
- **Evaluation breadth:** R² describes variance explained but not typical dollar error or error by price band. Add MAE/RMSE and residual analysis to understand practical pricing accuracy.
- **Depth / recency hypothesis:** The notebook's commentary suggests deeper trees and more recent windows may be useful. However, depth 11 does not always beat depth 7 for Decision Trees, and the best Random Forest score occurs with **24 months**, not three months. Treat these observations as hypotheses for repeated validation rather than settled conclusions.
- **No independently verified rerun:** Numerical results cited here were read from the notebook's saved outputs. The original IDX CSV was not supplied, and the Random Forest recording code needs correction, so these scores have not been reproduced independently in a fresh run.

## 8. How to reproduce the comparison

1. Obtain authorized IDX/CRMLS extracts, run the exploration and cleaning phases, and produce `Completedf.csv` with the expected schema.
2. Install the libraries listed above and open `Base_Compare (1).ipynb` in Jupyter or Colab.
3. Place `Completedf.csv` in the notebook's working directory or update its path.
4. Correct the Random Forest result-collection code, then restart the kernel and **Run All**.
5. Confirm the dataset shape, date boundaries, missing-value report, and common 7,069-row January 2024 test set against the saved run.
6. Inspect each model's R² summary and verify that the Random Forest estimator values are recorded correctly.
7. Compare model types using the same held-out month; for claims of generalization, rerun the experiment with training-only preprocessing, historical validation months, and a later untouched test month.

## 9. Conclusion and next steps

Among the configurations tested and recorded in this notebook, **Random Forest** produces the highest January 2024 R² (**0.778343**) versus **Decision Tree** (**0.673584**) and **Linear Regression** (**0.530972**). This is evidence that the relationship between property characteristics and closing prices benefits from nonlinear modeling in the current experiment.

Next steps are to fix the result tracking, add training/validation/test discipline across time, measure price prediction error in dollars, investigate feature importance and outliers, and use the corrected results as a comparison point for additional models (for example, gradient boosting). The optimal training-history length should remain an empirical choice rather than a fixed assumption.
