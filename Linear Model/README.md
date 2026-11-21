# Baseline Linear Regression Model — California Residential Sales

[← Project overview](../README.md) | [Next: Decision Tree and Random Forest Comparison](../04_model_comparison/README.md)


**Notebook:** [`First_Modelp2.ipynb`](First_Modelp2.ipynb)  
**Project phase:** Phase 3 — Baseline model development and evaluation  
**Data source:** Cleaned IDX / CRMLS residential single-family sales data from the [data cleaning phase](../02_data_cleaning/README.md)  
**Prediction target:** `ClosePrice`  
**Model:** scikit-learn `LinearRegression`  
**Evaluation metric:** Test-set coefficient of determination (R²)

## 1. Project objective

This notebook establishes the project's first predictive model for residential property closing prices. It uses the cleaned, feature-engineered dataset prepared in the previous phase to train an ordinary least-squares Linear Regression model. The purpose is to create a simple, interpretable baseline that future models—such as gradient boosting—can be compared against.

The assignment specifies three objectives:

1. Train a **Linear Regression** model as the first model.
2. Evaluate predictions using **R² on the test set**.
3. **Record baseline results** in the notebook.

In addition, the notebook carries forward the previous phase's requirement to explore the length of the training-history window. Four models are trained using 3, 6, 12, and 24 months of historical data, each evaluated on the **same most recent month**. This makes their scores directly comparable on that month's observed sales, although selecting the best window using the test set introduces a model-selection limitation discussed below.

## 2. Input data and dependencies

The notebook loads a single prepared CSV using:

```python
rdf = pd.read_csv("Completedf.csv")
```

The saved notebook reports **65,782 rows and 46 columns**. It is the dataset produced after the project's filtering, missing-value handling, and feature engineering steps. Relevant attributes include `ClosePrice`, `CloseDate`, `LivingArea`, geographic and property characteristics, and encoded flooring indicators. Unlike the earlier phases, this notebook does **not** import the raw 2022–2024 IDX extracts or rerun the cleaning steps: it expects the prepared CSV to exist beforehand.

**Important reproducibility gap:** The supplied data-cleaning notebook does not contain a `to_csv()` export of `Completedf.csv`. After confirming the final cleaned dataframe in that notebook is named `rdf`, save it explicitly before running Phase 3:

```python
rdf.to_csv("Completedf.csv", index=False)
```

Run the baseline notebook in the folder containing that file, or update the `pd.read_csv()` path. **Do not publish the CSV unless your IDX/CRMLS data license allows redistribution.** Sharing the notebooks and documentation alone does not make the source records publicly reproducible without authorized data access.

### Python packages

The notebook imports:

- **pandas:** CSV loading and DataFrame manipulation.
- **NumPy:** imported for numerical operations, although not directly used in the saved modeling cells.
- **scikit-learn:** `LinearRegression` for training and `r2_score` for evaluation.
- **Matplotlib:** actual-versus-predicted scatterplot.

A basic installation command is:

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

Package versions are not pinned. A Jupyter notebook or Google Colab environment can be used.

## 3. Modeling workflow

### Step 1 — Load and check the prepared dataset

The notebook reads `Completedf.csv`, prints its shape, displays the first records, and checks percentage missingness for each feature with:

```python
rdf.isna().mean() * 100
```

The saved output shows **0% missingness across all 46 columns**. This confirms the exported data was complete under pandas' missing-value definition when the notebook was executed; it does not independently confirm that every imputed value was correct, that every data type is appropriate, or that no invalid values remain.

### Step 2 — Split by closing month, not randomly

A reusable function, `create_temporal_split(df, training_months_x)`, converts `CloseDate` to datetime, finds the latest observed closing month, and places that entire month in the test set. The training set contains the **X full calendar months immediately before the test month**.

The function uses month boundaries and exclusive upper limits so the test data does not overlap the training period. It prints the date boundaries, counts, and the number of unparseable/missing closing dates for each split.

In the saved run, the latest month is **January 2024**:

| Training-window length | Training period | Training rows | Shared test period | Test rows |
| --- | --- | ---: | --- | ---: |
| 3 months | Oct 1–Dec 31, 2023 | 24,044 | Jan 1–31, 2024 | 7,069 |
| 6 months | Jul 1–Dec 31, 2023 | 47,079 | Jan 1–31, 2024 | 7,069 |
| 12 months | Jan 1–Dec 31, 2023 | 54,273 | Jan 1–31, 2024 | 7,069 |
| 24 months | Jan 1, 2022–Dec 31, 2023 | 58,713 | Jan 1–31, 2024 | 7,069 |

Notice that adding more calendar months does **not** proportionally increase row counts. The 12-month dataset contains only 7,194 more rows than the 6-month dataset, and the 24-month dataset contains only 4,440 more rows than the 12-month dataset. This may reflect differences in source coverage or how the previous phase filtered rows; the notebook does not investigate the cause.

### Step 3 — Define predictors and remove excluded columns

The target (`y_train`/`y_test`) is the original `ClosePrice`. Predictors are generated by dropping the following fields:

```python
columns_to_drop = [
    "ClosePrice", "Log_ClosePrice", "CloseDate",
    "PurchaseContractDate", "DaysOnMarket", "ListingContractDate",
    "ListOfficeName", "ListAgentFullName", "PropertyType",
    "PropertySubType", "ListPrice", "Log_ListPrice"
]
```

This excludes the target and its log transform, dates and transaction-related fields, agent/office information, property-type fields already used for filtering, and the listing-price fields. In particular, `Log_ClosePrice` must not be used to predict `ClosePrice`, because it is derived directly from the answer the model is trying to predict.

The notebook then keeps **only numeric columns**:

```python
X_train = X_train.select_dtypes(include="number")
X_test = X_test[X_train.columns]
```

This ensures scikit-learn receives numerical inputs in the same column order for both datasets. It also means any remaining object/string predictors are **silently omitted**, rather than encoded in this phase. Numeric-looking category codes, if present, may still be treated as continuous quantities; that should be reviewed before interpreting coefficients.

### Step 4 — Train the model and predict closing prices

For each training window, the notebook fits a fresh scikit-learn Linear Regression model:

```python
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Linear Regression fits coefficients that minimize the sum of squared differences between observed training closing prices and fitted closing prices. Its role here is **a reference point for future, more flexible models**, rather than a claim that property prices have a purely linear relationship with every feature.

The code does not scale the input features in this notebook. Standard scaling is not a strict requirement for ordinary, unregularized least-squares Linear Regression, although feature scales and correlations may matter for numerical stability and coefficient interpretation.

### Step 5 — Evaluate R² and record each run

The test score is calculated using:

```python
test_r2 = r2_score(y_test, predictions)
```

R² compares the model's prediction errors with the errors from always predicting the test set's mean closing price. A score of **1** indicates perfect prediction, **0** indicates the same squared-error performance as that mean baseline, and a **negative score** indicates worse squared-error performance than that baseline. Higher values indicate more of the variation in test closing prices is accounted for, **not** the percentage of individual sale prices predicted correctly.

For every window, the code saves `Training_Months`, `Training_Rows`, `Testing_Rows`, and `Test_R2` in `baseline_results`. These are converted into `results_df` and sorted by descending R² to compare performance.

## 4. Saved baseline results

The following values are taken directly from the executed notebook's results table:

| Rank | Training months | Training rows | Test rows | Test R² |
| ---: | ---: | ---: | ---: | ---: |
| 1 | **6** | **47,079** | **7,069** | **0.530972** |
| 2 | 12 | 54,273 | 7,069 | 0.529234 |
| 3 | 24 | 58,713 | 7,069 | 0.526381 |
| 4 | 3 | 24,044 | 7,069 | 0.526261 |

The **6-month window** achieved the highest observed test R² of approximately **0.5310**. For the January 2024 test transactions, this corresponds to explaining about **53.1% of the variation** in `ClosePrice` relative to a constant-mean prediction benchmark. The four scores are close together (a difference of approximately 0.0047 from best to worst), so this is evidence of a modest advantage on this particular test month, not proof that six months is always the optimal training history.

The results are a baseline rather than a final production estimate. The notebook does not calculate MAE/RMSE, perform repeated temporal validation, or report uncertainty around the scores.

## 5. Actual vs. predicted plot

The notebook ends with a scatterplot of actual `ClosePrice` values on the x-axis and model predictions on the y-axis. A red diagonal `y = x` reference line shows where perfectly accurate predictions would fall. Points farther from the line represent larger prediction errors; the line is **not** a regression fit through the scatter points.

**Which model is plotted?** The plot uses the variables `y_test` and `predictions` remaining after the `for` loop completes. Because the windows run in the order `3, 6, 12, 24`, the saved figure depicts predictions from the **24-month model**, not the highest-scoring 6-month model. To plot the best model, the corresponding predictions must be saved during the loop or the selected model must be fitted again.

The notebook contains the figure but does not save it to a separate image file.

## 6. Reproducing the experiment

1. Obtain authorized IDX/CRMLS source extracts and follow the [data exploration](../01_exploration/README.md) and [cleaning](../02_data_cleaning/README.md) notebooks.
2. Export the completed, feature-engineered `rdf` dataframe as `Completedf.csv`, preserving `CloseDate` and `ClosePrice`.
3. Place the CSV next to `First_Modelp2.ipynb`, or update the read path in the notebook.
4. Install the listed Python packages and run the cells **in order**.
5. Check that the input shape is **(65782, 46)** and that January 2024 contains **7,069** test records for the version of the data used here. Different authorized extracts, preprocessing rules, or added records may produce different counts and results.
6. Confirm that `results_df` reports all four windows and the values above; inspect the final scatterplot, remembering it visualizes the 24-month fit.

A small optional export makes future comparisons easier:

```python
results_df.to_csv("baseline_model_results.csv", index=False)
```

This optional CSV export is a recommendation, **not an operation performed by the supplied notebook**.

## 7. Limitations and recommended next steps

- **Test-set selection bias:** The same January 2024 holdout is used both to compare training windows and to report their performance. For a stronger final estimate, choose X using earlier rolling/expanding time-based validation periods and reserve an untouched later month for final testing. This is especially important before comparing more model types.
- **Upstream preprocessing leakage:** Some imputation values and categories in the preceding cleaning notebook were determined from the full dataset, including the test month. Learn imputation and encoding rules from each training split only, then apply them unchanged to its validation/test split.
- **Feature availability:** Ensure every predictor would be known at the moment a closing-price prediction is intended to be made. The code excludes several obvious potentially leaky features, but the remaining features still need this review.
- **Dropped categorical information:** Remaining text columns are omitted. A future pipeline can encode suitable categories consistently, handle unknown categories, and inspect how geography is represented.
- **Numeric feature meaning:** Postal codes or other categorical IDs may appear numeric without representing continuous relationships. Verify data types before treating them as linear numeric predictors.
- **Evaluation completeness:** Add mean absolute error (MAE), root mean squared error (RMSE), and residual analysis to express prediction errors in dollars and identify patterns the model misses.
- **Model comparison:** Compare this baseline against other approaches on equivalent time splits and data preparation. Record the metric, evaluation period, row counts, and feature-selection choices for every experiment.
- **Artifacts and repeatability:** Save model results, the selected model/preprocessing pipeline, and package versions; do not rely on variables remaining from the final loop iteration to represent the best model.

## 8. Conclusion

This phase successfully builds a **Linear Regression baseline**, evaluates it on the most recent month using **test-set R²**, and records scores across four historical training windows. In the saved run, **six months** achieved the highest R² (**0.530972**) for **January 2024** transactions. These results establish a useful benchmark for later predictive models, while leaving room to improve evaluation design, categorical feature handling, and overall prediction accuracy.

**Data usage note:** IDX/MLS sale records may be subject to licensing, access, and privacy restrictions. Reproduce this notebook with authorized extracts; do not commit restricted raw or cleaned sale records to a public repository.
