# Data Cleaning, Preprocessing, and Temporal Splitting — California Residential Sales

[← Project overview](../README.md) | [Next: Baseline Linear Regression](../03_baseline_model/README.md)


**Notebook:** [`Data_Cleaning_and_Analysis_Phase_revised.ipynb`](Data_Cleaning_and_Analysis_Phase_revised.ipynb)  
**Project phase:** Data cleaning and preparation for modeling  
**Data source:** IDX / CRMLS sold-property extracts  
**Target variable:** `ClosePrice`  
**Status:** Preprocessing and split design; **no predictive model is trained in this notebook**.

## 1. Project objective

This notebook builds on the earlier [exploratory data analysis](../01_exploration/README.md) by preparing IDX/CRMLS residential sales data for future machine-learning experiments. The goals are to reduce incomplete or inconsistent records, handle missing values using rules appropriate to each field, convert selected text/boolean variables to numerical features, examine whether skewed numerical features might benefit from log transforms, and create a chronological train/test split. This is intended to produce a more consistent dataset and a repeatable way to evaluate how much historical data a model should use.

**Assigned requirements:**

1. Decide whether to drop, impute, or flag missing values.
2. Encode categorical variables as numeric where necessary.
3. Normalize numerical features if needed.
4. Use the **most recent month** for testing and the **X months immediately before it** for training; experiment with different values of X instead of fixing the training window in advance.

The notebook implements each category of preparation to some degree. It **does not yet encode every remaining categorical predictor, perform conventional standardization, train models, or select an optimal value of X**. Those steps belong in the subsequent modeling workflow.

## 2. Input data and environment

The notebook was run in a Python 3 / Google Colab-style environment. It loads these two CSV files:

- `CRMLSSold20220101_20231231_filled(first 2 years).csv` — file labeled as covering 2022 and 2023.
- `CRMLSSold202401_filled(january_2024).csv` — file labeled as covering January 2024.

These are combined using `pd.concat(..., ignore_index=True)`. The saved output shows a combined source dataframe (`mdf`) with **175,786 rows and 80 columns**. As in the EDA phase, the population is filtered to `PropertyType == "Residential"` and `PropertySubType == "SingleFamilyResidence"`.

Imports include pandas, NumPy, Matplotlib, Seaborn, and Plotly Express. For a comparable environment:

```bash
pip install pandas numpy matplotlib seaborn plotly jupyter
```

The source files are not included here. IDX/MLS data may have access and redistribution restrictions; obtain authorized extracts rather than publishing the raw data. The notebook's current `pd.read_csv` paths point to `/content/`; change them when running locally. A mixed-type `DtypeWarning` is shown on CSV import, so input types should be reviewed when reproducing the notebook.

## 3. Processing workflow

### Step 1 — Select fields and restrict the dataset

The code first selects 40 columns from the original 80, including the target (`ClosePrice`), property dimensions, beds/baths, location, dates, property descriptors, garage and parking details, and flooring. It then filters for **Residential / SingleFamilyResidence** properties and converts `CloseDate` and `ListingContractDate` to datetime values.

A later second column-selection step removes nine of the original 40 columns, leaving **31 retained fields** before flooring encoding. Examples removed at this stage include association-fee fields, subdivision, buyer office association, total building area, and lot-size dimensions. The resulting feature set reflects preliminary judgment, **not validated model-based feature selection**.

### Step 2 — Remove potential duplicates and invalid or incomplete records

Potential duplicate records are removed with:

```python
rdf = rdf.loc[
    ~rdf.duplicated(subset=["ClosePrice", "CloseDate", "ListAgentFullName"])
].reset_index(drop=True)
```

This treats rows with the same closing price, closing date, and listing agent as potential duplicates. **It is a heuristic, not a unique transaction identifier**: two legitimate sales could share these values. A listing/transaction key would be a safer long-term duplicate rule.

The notebook then drops rows missing `Latitude`, `LivingArea`, or `ListAgentFullName` and retains only rows whose `CloseDate` is later than `ListingContractDate`. It also excludes records with `ParkingTotal > 10` and removes records with missing parking counts **because of how the comparison is ordered** (see cautions below). Later, it drops missing observations in `LotSizeAcres`, `LotSizeSquareFeet`, `YearBuilt`, `BathroomsTotalInteger`, `City`, `PurchaseContractDate`, and `NewConstructionYN`.

**Why:** These decisions prioritize complete fields needed for property-level analysis and eliminate some obviously implausible combinations. **Tradeoff:** Every row deletion shrinks the dataset and may introduce selection bias if missingness is systematic.

### Step 3 — Impute selected missing values

The notebook uses different strategies depending on the feature:

| Field(s) | Rule implemented | Rationale / limitation |
| --- | --- | --- |
| `LotSizeAcres`, `LotSizeSquareFeet`, `YearBuilt`, `BathroomsTotalInteger`, `City`, `PurchaseContractDate`, `NewConstructionYN` | Drop rows when missing | Keeps these features observed; may remove many otherwise valid transactions. |
| `Latitude`, `LivingArea`, `ListAgentFullName` | Drop rows when missing | Ensures these fields are present, even though not all are necessarily appropriate final model inputs. |
| `ParkingTotal` | Filter `<= 10`, then attempt median imputation | **Missing values have already been filtered out**, so the subsequent `fillna(median)` does not recover them. |
| `GarageSpaces` | Infer zero for some cases when `AttachedGarageYN` is false, then fill remaining missing with `0` | Missing garage count is assumed to be zero; may be inaccurate. |
| `AttachedGarageYN` | Set to `1` when garage spaces are positive and attached-garage status is missing, then fill remaining missing with `0` | Convenient proxy, but **having a garage is not proof it is attached**. |
| `BasementYN` | Fill missing/blank with `False` and convert to `0` | Very large missingness; **unknown is not necessarily the same as no basement**. |
| `Stories` | Fill with the dataset median | Keeps rows without needing a story count. |
| `Levels` | Fill with most frequent category (mode) | Handles missing categories without removing rows. |
| `HighSchoolDistrict` | Fill missing with `"Unknown"` | Preserves the distinction between unknown and named districts. |
| `Flooring` | Later add `Flooring_Missing` indicator | Preserves the missingness signal rather than dropping all records with unknown flooring. |

These operations are sufficient to produce a final zero-missing-value summary for the retained dataset after flooring is encoded, **but they should not be assumed to preserve the true meaning of every missing observation**.

### Step 4 — Encode boolean and multi-label features

Three yes/no fields are converted to `0`/`1` integers:

```python
boolean_cols = ["AttachedGarageYN", "NewConstructionYN", "BasementYN"]
for col in boolean_cols:
    rdf[col] = rdf[col].fillna(False).replace("", False).astype(int)
```

`Flooring` can list more than one material, such as `Carpet,Wood`. The code splits the comma-separated string, finds distinct materials, and creates an indicator column for each. The saved output contains `Flooring_Bamboo`, `Flooring_Brick`, `Flooring_Carpet`, `Flooring_Concrete`, `Flooring_Laminate`, `Flooring_Stone`, `Flooring_Tile`, `Flooring_Vinyl`, and `Flooring_Wood`, plus `Flooring_Missing` and `Flooring_SeeRemarks`. It then removes the original `Flooring` string column.

This is **multi-hot encoding** (several flooring indicators can equal `1` for one property), rather than mutually exclusive one-hot encoding. Other retained object columns—including `City`, `CountyOrParish`, `PostalCode`, `HighSchoolDistrict`, `Levels`, `ListOfficeName`, and `ListAgentFullName`—**remain categorical** in this notebook and will require appropriate treatment before use in most numerical ML models.

### Step 5 — Log-transform selected numerical columns

The notebook creates the following additional fields using `np.log1p(value)`:

- `Log_ClosePrice`
- `Log_ListPrice`
- `Log_LivingArea`
- `Log_LotSizeSquareFeet`
- `Log_LotSizeArea`

`log1p(x)` computes `ln(1 + x)`, which compresses large positive values and may reduce right-skew. **This is a logarithmic transformation, not standard scaling or min-max normalization.** It must be applied to valid values greater than `-1`; values at or below `-1` can produce undefined/nonfinite results. If `Log_ClosePrice` is modeled as the target, predictions must be back-transformed with `expm1` for dollar-scale interpretation, and the untransformed price must not also be used as an input feature.

The saved notebook shows **65,782 rows and 41 columns after flooring encoding**. The five additional log columns bring the expected schema to **46 columns**; the notebook does not include a separate displayed `rdf.shape` after the log transforms.

### Step 6 — Create a chronological train/test split

The function `create_temporal_split(df, training_months_x)`:

1. Copies the supplied dataframe and ensures `CloseDate` is a datetime.
2. Finds the most recent `CloseDate` and identifies the **first day of that calendar month**.
3. Uses that entire latest month as the **test set**.
4. Uses the `X` full calendar months immediately before that month as the **training set**.
5. Prints the period boundaries, number of records in each set, and any missing close dates.
6. Returns the two separate dataframes.

The notebook tries `window_options = [3, 6, 12, 24]`. Saved results:

| Training window (X) | Training period | Training rows | Test period | Test rows |
| --- | --- | ---: | --- | ---: |
| 3 months | Oct–Dec 2023 | 24,044 | Jan 2024 | 7,069 |
| 6 months | Jul–Dec 2023 | 47,079 | Jan 2024 | 7,069 |
| 12 months | Jan–Dec 2023 | 54,273 | Jan 2024 | 7,069 |
| 24 months | Jan 2022–Dec 2023 | 58,713 | Jan 2024 | 7,069 |

**Interpretation:** All four experiments use the same January 2024 holdout, so future model performance can be compared across different amounts of training history. However, the notebook only reports row counts and date ranges; it does **not** evaluate a model or choose the best training window. The comparatively small added 2022 population in the 24-month window is worth investigating for data coverage or selection effects.

## 4. Results and current status

- Initial combined source: **175,786 rows × 80 columns**.
- Reduced after first selection: **175,786 rows × 40 columns**.
- After duplicate removal: **86,898 rows** in the single-family filtered set (as shown in the saved dataframe output).
- After dropping missing latitude/living area/agent: **86,727 rows × 40 columns**.
- After enforcing chronological listing/closing dates: **85,923 rows × 40 columns**.
- Final cleaned, flooring-encoded table: **65,782 rows × 41 columns**.
- After creating five log features: **65,782 rows × 46 expected columns**.
- Final month-based test population: **7,069 January 2024 transactions**, unchanged across the four training-window experiments.

These figures come from the notebook's **saved outputs**, not a fresh execution against the original CSVs. Other IDX extracts or versions may return different counts.

## 5. How to reproduce this notebook

1. Clone or download the repository and open `Data_Cleaning_and_Analysis_Phase_revised.ipynb` in Jupyter or Colab.
2. Obtain the two authorized CSVs identified in **Input data and environment**.
3. Place the CSVs in the notebook's expected `/content/` directory (Colab), or update both `pd.read_csv(...)` paths to accessible local paths.
4. Install the required packages and run notebook cells **in order**, from the beginning. The notebook expects its working dataframes (`mdf`, `reducedmdf`, `rdf`) to be built in sequence.
5. Inspect intermediate `.shape`, `.dtypes`, and missing-value-percentage outputs; confirm any substantial differences before proceeding.
6. Verify the final `rdf` has the expected rows and columns, the flooring indicators are present, and the log transforms contain finite values.
7. Run `create_temporal_split` for `[3, 6, 12, 24]` and check the printed time ranges and row counts.
8. In the later modeling notebook, fit any learned transformations (e.g., numeric scalers, imputers, encoders) on the **training partition only**, then apply them to the test partition. Train the same model configuration on each X-month window and compare test metrics such as MAE and RMSE before choosing a preferred window. For a more robust selection method, use earlier rolling time splits for tuning and reserve January 2024 for final evaluation.

A short sanity-check cell that can be added **after** feature creation:

```python
print("Final shape:", rdf.shape)
print("Closing dates:", rdf["CloseDate"].min(), "to", rdf["CloseDate"].max())
print("Remaining nulls:", int(rdf.isna().sum().sum()))
print("Duplicate column labels:", rdf.columns.duplicated().sum())
log_cols = [col for col in rdf.columns if col.startswith("Log_")]
print("Nonfinite log values:", int((~np.isfinite(rdf[log_cols])).sum().sum()))
```

## 6. Limitations and recommended improvements

The notebook provides a workable preprocessing starting point, but the following issues should be considered before production or final model evaluation:

1. **Temporal leakage:** Medians, modes, and the set of flooring labels are determined from the full cleaned dataset *before* the train/test split. Fit learned preprocessing rules on training data, then apply them to the test data, to avoid information leaking from January 2024 into training.
2. **Not fully model-ready yet:** Several categorical columns are still text. Choose encoding strategies for them; high-cardinality agent/office identifiers may need special care.
3. **Potential target or future-information leakage:** `ClosePrice` is the target and must be excluded from model inputs. Also decide when the prediction would be made. `DaysOnMarket`, contract dates, or other fields may be unavailable at prediction time; `Log_ClosePrice` is target-derived and must never appear among predictors of the original target.
4. **Missing-value semantics:** Missing basement status is treated as `0`, and a positive garage-space count may be used to infer an attached garage. Validate these assumptions; consider separate unknown indicators or documented business rules.
5. **Parking filter ordering:** `rdf = rdf[rdf['ParkingTotal'] <= 10]` removes missing values before the median fill, so the median-imputation line does not actually impute those missing observations. Reorder or explicitly justify dropping them.
6. **Duplicate detection:** Matching agent, closing date, and price may exclude distinct properties. Prefer a stable listing identifier or validated transaction key if available.
7. **Log vs. scaling:** Log transforms address skew but do not necessarily center features or make scales comparable. Fit a scaler on the training set if the chosen model needs one.
8. **Coverage and representativeness:** The 24-month training split adds relatively few 2022 records compared with 2023. Check monthly record counts, missing-data patterns, location mix, and the effect of row filtering before interpreting time-window comparisons.
9. **Datetime validation:** The notebook converts `PurchaseContractDate` using `errors="coerce"` after earlier dropping missing values; invalid date strings could still become `NaT`. Recheck date quality after conversion.
10. **Reproducibility:** Version-pinned dependencies, a data dictionary, tracked row counts after each filter, and a reusable preprocessing function or pipeline would make this more dependable for newly received data.

## 7. Next phase

The next notebook can train a baseline model and compare models trained on 3, 6, 12, and 24 months of history. For each window, use the same January 2024 evaluation period, apply consistent training-only preprocessing, record error metrics, and select X based on clearly stated validation criteria. This will turn the **configurable split function** into an evidence-based selection of a training window.

**Data privacy and licensing:** IDX/MLS exports may be restricted and can contain identifiable agent or transaction fields. Do not commit raw extracts, personal contact details, or unauthorized derived datasets to a public repository.
