# Feature Engineering — Property Attributes and California School District Geography

**Notebook:** [`Feature_eng.ipynb`](../Feature_eng.ipynb)  
**Project stage:** Phase 5 in this repository / Week 6 of the assignment  
**Source transactions:** Prepared IDX/CRMLS California residential, single-family sales from [Phase 2](02_data_cleaning_and_preprocessing.md)  
**External geographic source:** [California School District Areas 2024–25, California Department of Education (ZIP shapefile)](https://data.ca.gov/dataset/california-school-district-areas-2024-25/resource/7dfaf005-58eb-45db-93b1-7aff091b2172)  
**Prediction target:** `ClosePrice` (dollars)  
**Evaluation:** R² on sales closed in January 2024

## 1. Purpose and assignment requirements

This notebook investigates whether additional property characteristics and geographic context can help predict the final closing price of a residential home. The previous modeling phases used existing numeric property attributes. Here, the data is enriched with derived property features and a school-district identifier obtained by geographically matching each property's coordinates to California Department of Education district boundaries.

The Week 6 instructions were to create features such as a bedroom/bathroom ratio or property age, spatially join properties to California School District Areas 2024–25, retrain models, and provide a table comparing the original and expanded feature sets.

**What the submitted notebook contains:**

- Four derived numeric/binary property variables: `PropertyAge`, `BathroomsPerBedroom`, `LogLotToLivingRatio`, and `NewerHome`.
- One categorical school-district variable, `OfficialSchoolDistrict`, spatially joined from the state boundary layer and then one-hot encoded.
- New Decision Tree and Random Forest experiments across four time-based training windows.
- A chart comparing the best saved original-feature and engineered-feature R² values.
- **No Linear Regression run with engineered features.** It appears in the comparison plot with the engineered result intentionally blank (`np.nan`).

**Important interpretation:** The higher scores are observed results from this notebook, but they do not isolate the benefit of feature engineering alone. The models' depth and other hyperparameters were also changed. See [Section 7](#7-how-to-interpret-the-comparison).

## 2. Inputs, packages, and data expectations

### Transaction data

The notebook starts by loading an already prepared CSV:

```python
rdf = pd.read_csv("Completedf.csv")
```

The saved output shows **65,782 properties and 46 columns**. The notebook checks percentage missing for each attribute and shows zero missing values in the prepared dataset. This file is the output of earlier cleaning and encoding—not a raw IDX export. See [Phase 2 documentation](02_data_cleaning_and_preprocessing.md).

**To reproduce this stage:** First run the earlier cleaning phase and save its final cleaned `rdf` dataframe:

```python
rdf.to_csv("Completedf.csv", index=False)
```

Save `Completedf.csv` in the working directory used by `Feature_eng.ipynb`, or change the CSV path. This project involves IDX/MLS records; confirm redistribution rights before placing any transaction data in a public repository.

### School district shapefile

Download the **2024–25** district areas ZIP/shapefile from the linked California Department of Education resource. The notebook expects a local file named exactly:

```text
DistrictAreas2425.zip
```

If the downloaded ZIP has a different filename, rename it or edit `zip_path` in the notebook. The code extracts the archive into `school_districts_extracted/` when it cannot find an existing `.shp` there and reads the first located shapefile. The shapefile and its supporting components (`.shx`, `.dbf`, `.prj`, etc.) need to remain together. The notebook does **not** download the boundary data automatically.

The source represents **district boundaries effective for the 2024–25 academic year**, whereas the modeled property sales span January 2022–January 2024. These boundaries are therefore a retrospective approximation to the geographic districts at each historical sale date, not a verified historical point-in-time match.

### Environment

Python 3 in Jupyter or Google Colab is expected. The notebook imports pandas, NumPy, Matplotlib, GeoPandas, `zipfile`, `pathlib`, and scikit-learn.

```bash
pip install pandas numpy matplotlib geopandas scikit-learn jupyter
```

The saved notebook does not pin package versions. `geopandas` must be able to read shapefiles and perform a spatial join with a supported geometry/spatial-index backend. These experiments can be computationally intensive, especially after district encoding expands the predictors.

## 3. Chronological modeling setup

The notebook reuses `create_temporal_split(df, training_months_x)` from earlier modeling phases. It converts `CloseDate` to datetime, identifies the most recent month present (**January 2024**), and uses that month as the test data. Each training set contains the *X* calendar months immediately before it.

| Training history | Training dates | Training properties | Test dates | Test properties |
| --- | --- | ---: | --- | ---: |
| 3 months | Oct–Dec 2023 | 24,044 | Jan 2024 | 7,069 |
| 6 months | Jul–Dec 2023 | 47,079 | Jan 2024 | 7,069 |
| 12 months | Jan–Dec 2023 | 54,273 | Jan 2024 | 7,069 |
| 24 months | Jan 2022–Dec 2023 | 58,713 | Jan 2024 | 7,069 |

The target is the raw `ClosePrice`. The model excludes the following columns from predictors:

```python
columns_to_drop = [
    "ClosePrice", "Log_ClosePrice", "CloseDate",
    "PurchaseContractDate", "DaysOnMarket",
    "ListingContractDate", "ListOfficeName",
    "ListAgentFullName", "PropertyType",
    "PropertySubType", "ListPrice", "Log_ListPrice"
]
```

It then keeps numerical predictor columns using `select_dtypes(include="number")` and aligns the test set to the training-set column list. This means numeric engineered attributes and district indicators are used, while remaining text columns are not used directly.

**Evaluation design caveat:** The models' settings and history windows are selected using January 2024 R², which is also reported as the test score. It is more accurate to describe these as *selected January 2024 evaluation results* than as an unbiased estimate of performance on a future, untouched month.

## 4. New property-level features

The notebook creates four derived features in a copy of the original prepared dataframe, `rdf_features`.

| Feature | Construction | Purpose and considerations |
| --- | --- | --- |
| `PropertyAge` | `max(listing year − YearBuilt, 0)`; missing or invalid results filled with `0` | Encodes the approximate age of the property at listing. Zero can also represent missing/invalid data, not only a newly built home. |
| `BathroomsPerBedroom` | `BathroomsTotalInteger / BedroomsTotal` | Captures bathroom availability relative to bedrooms. Zero bedroom counts are first replaced with missing to prevent division by zero, then filled with `0`. |
| `LogLotToLivingRatio` | `log1p(LotSizeSquareFeet / LivingArea)` | Compares lot size with indoor living area, with a log transform to reduce the effect of extreme ratios. Zero living area is handled by replacing the denominator with missing and later filling the ratio with `0`. |
| `NewerHome` | `1` if `PropertyAge <= 5`, otherwise `0` | A binary indicator for properties five years old or newer according to the derived age. Invalid/missing ages imputed as `0` will also be classified as newer. |

The assignment mentions a bed/bath ratio as an example; the actual notebook computes the **inverse direction**, **bathrooms per bedroom**, and documentation should reflect that name and formula exactly.

The code uses **listing year** rather than closing year for `PropertyAge` because it derives the year from `ListingContractDate`. The full listing/closing dates are then excluded from the predictors.

### Improvements worth considering

- Keep a separate `PropertyAgeMissing` indicator when `YearBuilt` or the listing date is unavailable. Do not automatically interpret unknown age as zero years old.
- Verify whether zero bedroom counts represent studios, data errors, or meaningful missing categories, rather than treating every zero-derived bathroom ratio equivalently.
- Check for negative/invalid lot and living-area measurements before taking ratios and logarithms.
- Document the exact intended prediction time (for example, when a listing first appears) and ensure every retained feature would actually be available at that time.

These are **recommended changes**, not changes already made to the submitted notebook.

## 5. Geographic feature: official school districts

### Spatial joining approach

The geographic feature workflow is:

1. Extract and read a district boundary shapefile using **GeoPandas**.
2. Locate the source district name/type columns and rename them `DistrictName` and `DistrictType`.
3. Retain districts whose type contains `high`, `secondary`, or `unified` (case insensitive). **Elementary-only districts are deliberately excluded.**
4. Convert the district geometry to geographic coordinates in `EPSG:4326` (latitude/longitude).
5. Check that property coordinates fall within broad California latitude/longitude ranges; create point geometries using `Longitude` as X and `Latitude` as Y.
6. Use `gpd.sjoin(..., how="left", predicate="within")` to match points inside qualifying district polygons.
7. Detect multiple matches, select the first district name alphabetically for any duplicate property index, and assign unmatched records the string `Unknown`.

The resulting `OfficialSchoolDistrict` is a geographical label derived from the public boundary layer. It is **not a measure of school quality, school ratings, attendance eligibility, or a guarantee that a given household can enroll in a particular school**. High/unified boundaries and this selection strategy should be described precisely when interpreting results.

### Spatial matching results in the saved output

| Metric | Saved value |
| --- | ---: |
| Property records before join | 65,782 |
| Property records after join | 65,782 |
| Invalid coordinates (outside broad bounding box) | 14 |
| Properties with multiple district matches | 0 |
| Properties assigned a district other than `Unknown` | 65,713 |
| Total records remaining `Unknown` | 69 |

The match rate is approximately **99.9%** of the input properties. The 69 unknowns comprise **14 records flagged as invalid coordinates** and **55 additional records without a matched qualifying district**. Possible reasons for otherwise valid coordinates not matching include the district-type filter, boundary edge behavior, or incomplete coverage; the notebook does not investigate their causes.

### One-hot encoding

`pd.get_dummies()` converts the district name to binary indicator columns using:

```python
school_dummies = pd.get_dummies(
    rdf_features["OfficialSchoolDistrict"],
    prefix="OfficialSchoolDistrict",
    drop_first=True,
    dtype=int,
)

rdf_features = pd.concat(
    [rdf_features.drop(columns=["OfficialSchoolDistrict"]), school_dummies],
    axis=1,
)
```

The dataset changes shape as follows:

| Stage | Rows | Columns |
| --- | ---: | ---: |
| Prepared dataset | 65,782 | 46 |
| After property features and district-name field | 65,782 | 51 |
| After district one-hot encoding | 65,782 | 415 |

The encoder adds **365 binary columns** after removing the original district-name field. `drop_first=True` omits one category as the reference; it does **not** necessarily mean the `Unknown` category is omitted.

**Deployment note:** To score data from a new source, reuse the exact same property-feature formulas, school-district boundary version, and one-hot category layout learned from training. When training/serving, fit the category vocabulary on training data and reindex new predictions to the training columns. Creating dummy columns independently on different batches can result in misaligned predictors. In this notebook, encoding is performed on the full data **before** the chronological split, which means the category vocabulary incorporates the held-out month; a strict evaluation pipeline should avoid this.

## 6. Models retrained with engineered attributes

The notebook first reruns models using the original prepared data, then creates and evaluates the expanded `rdf_features` dataset. The score tables below reproduce **saved notebook outputs**; the code has not been rerun or repaired for this documentation.

### Models evaluated on original features in *this notebook*

- **Linear Regression:** four windows, best **R² = 0.530972** at **6 months**.
- **Decision Tree:** depths `[3, 5, 7, 11]` and four windows, best **R² = 0.673584** at **6 months**, `max_depth=11`.
- **Random Forest:** `n_estimators=[50,100,200]`, `max_depth=11`, and four windows, best **R² = 0.777501** at **12 months**, `n_estimators=200` (identified from the printed run summaries).

**Why this original forest differs from Phase 4:** The [Phase 4](04_model_comparison.md) forest sweep also tried **150 and 300 trees** and found a higher **R² = 0.778343** using **24 months, 150 trees, depth 11**. The `0.777501` value is the best original-feature forest in this **new notebook's smaller search**, and is the value its final chart uses. These two runs should not be described as contradictory results.

### Models evaluated on engineered features

| Model | Best training window | Key settings | Best saved January 2024 R² |
| --- | ---: | --- | ---: |
| Decision Tree | 12 months | `max_depth=20` | **0.734518** |
| Random Forest | 12 months | `n_estimators=100`, `max_depth=20`, `min_samples_split=5` | **0.825943** |
| Linear Regression | — | Not run after feature engineering | **Not available** |

The feature-engineered Decision Tree uses depths `[7, 11, 20]` and all four training windows. The engineered Random Forest specifies `[50,100,200]` trees, `max_depth=20`, and `min_samples_split=[3,4,5]`. **However, due to the loop indentation problem described in Section 8, only the models with `min_samples_split=5` are evaluated and recorded.** The other split settings are fitted but not scored in the saved output.

## 7. How to interpret the comparison

The submitted notebook's final chart uses this comparison:

| Model | Best original-feature R² (within this notebook) | Best engineered-feature R² | Difference |
| --- | ---: | ---: | ---: |
| Linear Regression | 0.530972 | *Not evaluated* | — |
| Decision Tree | 0.673584 | 0.734518 | **+0.060934** |
| Random Forest | 0.777501 | 0.825943 | **+0.048442** |

Using the stronger forest from the earlier Phase 4 sweep (0.778343) instead produces a difference of **+0.047600** to the engineered forest score. Whichever reference you use, **label it clearly**.

The strongest saved engineered forest explains about **82.6% of the variation in closing prices** on the selected January 2024 evaluation data relative to a constant-mean predictor. This is not the percentage of individual house prices predicted accurately.

The results are promising, particularly because geographic location can capture patterns not fully represented by latitude/longitude alone. **Nevertheless, the comparison is not a controlled feature-ablation test:**

- The original Decision Tree's winning depth was **11**, while the engineered tree's was **20**.
- The original forest's depth was **11**, while the engineered forest's was **20**, and the engineered forest also varies `min_samples_split`.
- The one-hot-encoded district variables add hundreds of predictors; improvements may be due to some new variables, changed model capacity, or both.
- All configurations are compared/selected against the same January 2024 month, potentially yielding optimistic selected scores.
- School-district geometries are effective in **2024–25**, after the sales dates, so any historical predictive claim needs an explicit point-in-time boundary caveat.

To measure the incremental value of the new features, compare original vs engineered columns **at matched hyperparameters, identical training rows, and the same held-out evaluation month**, ideally with multiple earlier chronological validation months and a separate final untouched test month. It may also be useful to test property features alone, district features alone, and both together.

### Model behavior and tradeoffs

| Model | Benefit | Potential weakness |
| --- | --- | --- |
| Decision Tree | Learns nonlinear thresholds and feature interactions; can be inspected | Sensitive to tree depth and sample changes; high depths can overfit |
| Random Forest | Averages many trees, generally reducing single-tree instability; best saved evaluation score | More computation and memory with 365 district dummy columns; less directly interpretable; may not extrapolate to unprecedented prices |
| Geographic district encoding | Gives local geographic context beyond raw coordinates | High cardinality; small districts may have few training examples; 2024–25 boundaries are not contemporaneous with 2022–24 sales |

The notebook **does not** show feature importances, training scores, runtime measurements, MAE/RMSE, or performance by district, so this table mixes observed results with general modeling considerations rather than reporting unmeasured outcomes.

## 8. Code issues to fix before claiming a reproducible model sweep

### Issue A — Random Forest results are appended to the wrong list

Both the original and engineered forest loops initialize `rf_results = []` but append dictionaries to **`dt_results`**. This mixes the Random Forest results with Decision Tree results. They also store `"N value": n_vals` (the **whole list**) instead of the current `n`. Consequently, the displayed result tables do not correctly show the estimator count for each row.

For each forest loop, use `rf_results.append(...)`, store the scalar `n`, then build `rf_results_df` rather than `dt_results_df`.

### Issue B — The engineered forest evaluates only the last `min_samples_split`

In the feature-engineered forest cell, the prediction, R² calculation, recording, and print statements are **outside** the innermost `for s in splits` loop. The cell therefore fits models at `s=3,4,5` but retains/evaluates **only the last fitted model (`s=5`)** for each tree count and training window.

An illustrative corrected core loop is:

```python
rf_results = []
for months in window_options:
    train_set, test_set = create_temporal_split(
        rdf_features, training_months_x=months
    )
    y_train, y_test = train_set["ClosePrice"], test_set["ClosePrice"]
    X_train = train_set.drop(columns=columns_to_drop, errors="ignore")
    X_test = test_set.drop(columns=columns_to_drop, errors="ignore")
    X_train = X_train.select_dtypes(include="number")
    X_test = X_test[X_train.columns]

    for n in [50, 100, 200]:
        for depth in [20]:
            for split_size in [3, 4, 5]:
                model = RandomForestRegressor(
                    n_estimators=n,
                    max_depth=depth,
                    min_samples_split=split_size,
                    random_state=42,
                )
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                score = r2_score(y_test, predictions)
                rf_results.append({
                    "Training_Months": months,
                    "Training_Rows": len(X_train),
                    "N_Estimators": n,
                    "Max_Depth": depth,
                    "Min_Samples_Split": split_size,
                    "Test_R2": score,
                })

rf_results_df = pd.DataFrame(rf_results).sort_values(
    "Test_R2", ascending=False
)
display(rf_results_df.head(10))
```

This corrects the reporting logic; **it is recommended code, not code already executed in the submitted notebook**. It evaluates **36 forest configurations** (4 windows × 3 estimator counts × 1 depth × 3 split settings) rather than only recording 12. The saved best score `0.825943` belongs to the `min_samples_split=5` fits shown in the output; it remains a reported result until the corrected notebook is rerun.

### Issue C — A proper before/after comparison should control more variables

The old and engineered models were not compared under identical hyperparameters. To support the assignment's *old vs new features* question more strongly, add a table for each model at a **fixed configuration**, varying only the feature set. Then optionally add a separately labeled *best tuned result* table.

### Issue D — Clean categorical schema and temporal validation

The district dummy columns are generated on the complete dataset before splitting. Fit the encoder using training data, handle unseen districts at prediction time, and preserve predictor column order. For validation, perform window/hyperparameter selection using earlier dates and reserve the final month as a true untouched test month, if enough time periods are available.

### Issue E — Potentially ambiguous feature values

`PropertyAge` fills invalid values with 0 and `NewerHome` turns 0 into 1. This can mislabel unknown properties as new. Consider explicit missingness flags and validate `YearBuilt`, `BedroomsTotal`, `LivingArea`, and `LotSizeSquareFeet` before calculating derived features.

## 9. Suggested rerun and repository workflow

1. Run the previous cleaning notebook and save the validated `Completedf.csv` (subject to IDX/MLS permissions).
2. Download the specified **2024–25** district shapefile ZIP and make its local filename match `DistrictAreas2425.zip`.
3. Open `Feature_eng.ipynb` with the working directory containing these two inputs.
4. Install dependencies and run cells from the top; verify the cleaned dataset shape is **(65782, 46)**, that the geographic join retains **65,782** properties, and that district one-hot encoding produces **415** total columns using the same underlying data and boundary file.
5. Fix forest result tracking and indentation. Rerun the comparison before publishing definitive tuning tables. Scores could vary across changed code, data versions, or libraries.
6. Preserve both tables: **original vs engineered with identical model settings**, and **best tuned score by model**. Report which data period and features were used.
7. To improve confidence, add a chronological validation period, error metrics such as MAE and RMSE, and feature-ablation or feature-importance analysis.

The current notebook has **not** been edited by this documentation. The recorded results are from its saved outputs, and the suggested fixes above are separate.

## 10. Key takeaway

This phase expands a prepared 46-column real estate dataset by engineering property-age, bathroom-density, lot-relative-size, and newer-home variables and matching approximately **99.9%** of properties to selected 2024–25 California high-school/unified district boundaries. Encoding the matched districts yields a **415-column** dataset. The best saved engineered Random Forest has an R² of **0.825943**, compared with **0.777501** in this notebook's original-feature forest sweep, and the engineered Decision Tree reaches **0.734518** versus **0.673584** originally. The apparent gains warrant follow-up with controlled hyperparameters and strict chronological validation before attributing them specifically to the engineered features.

---

### Documentation references

- [Exploratory analysis — Phase 1](01_exploration.md)
- [Data cleaning and preprocessing — Phase 2](02_data_cleaning_and_preprocessing.md)
- [Baseline Linear Regression — Phase 3](03_baseline_model.md)
- [Decision Tree / Random Forest comparison — Phase 4](04_model_comparison.md)
- [California School District Areas 2024–25 (official dataset)](https://data.ca.gov/@california-department-of-education/california-school-district-areas-2024-25)
