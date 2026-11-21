# California Residential Home Price Prediction | IDX Data Science Project

An end-to-end data science project exploring **California single-family home sales** and predicting final closing prices (`ClosePrice`) from property attributes and location. The workflow starts with raw IDX/CRMLS data, progresses through cleaning and geographic feature engineering, then compares several regression models and evaluates prediction errors by home-price range.

**Repository guide:** This page is the high-level project overview. **Each phase has its own detailed [`README.md`](01_exploration/README.md)** with methods, decisions, saved findings, limitations, and steps to reproduce the notebook.

> **Data availability:** IDX/MLS sale records may have licensing restrictions. The original sales CSVs and cleaned transaction-level exports are **not included**. Obtain authorized data and follow each notebook's input instructions. Previous notebook scores reflect saved outputs, not a fresh end-to-end run of this packaged repository.

## Project at a glance

| Item | Summary |
| --- | --- |
| **Business goal** | Understand which home characteristics are related to sale prices and build a reproducible model for estimated residential closing price |
| **Prediction target** | `ClosePrice` (USD) |
| **Property population** | `PropertyType = Residential` and `PropertySubType = SingleFamilyResidence` |
| **Geography** | California; property latitude/longitude and, later, school-district boundaries |
| **Time period** | Source filenames indicate January 2022–January 2024; verify the observed dates in `CloseDate` |
| **Algorithms** | Linear Regression, Decision Tree, Random Forest, and XGBoost |
| **Evaluation** | Chronological holdout; R², MAE, RMSE, MAPE, MdAPE, and errors by price band |

## Explore the project, phase by phase

Each link leads to **the detailed phase README**. Notebook links open the original work.

| Phase | Goal | Detailed README | Jupyter notebook |
| --- | --- | --- | --- |
| **1. Data Exploration** (Week 2) | Explore prices, home features, locations, and potential data issues | [Read Phase 1](https://github.com/EthanFerba/IDX-Exchange-Data-Science/blob/main/EDA/README.md) | [EDA notebook](https://github.com/EthanFerba/IDX-Exchange-Data-Science/blob/main/EDA/EDA_Phase.ipynb) |
| **2. Data Cleaning** | Address missing/invalid values, encode features, define chronological splits | [Read Phase 2](02_data_cleaning/README.md) | [Cleaning notebook](02_data_cleaning/Data_Cleaning_and_Analysis_Phase_revised.ipynb) |
| **3. Baseline Model** | Build Linear Regression and establish a test R² benchmark | [Read Phase 3](03_baseline_model/README.md) | [Baseline notebook](03_baseline_model/First_Modelp2.ipynb) |
| **4. Model Comparison** | Compare Decision Tree and Random Forest against Linear Regression | [Read Phase 4](04_model_comparison/README.md) | [Comparison notebook](04_model_comparison/Base_Compare%20%281%29.ipynb) |
| **5. Feature Engineering** (Week 6) | Add property-derived features and California school-district spatial data | [Read Phase 5](05_feature_engineering/README.md) | [Feature notebook](05_feature_engineering/Feature_eng.ipynb) |
| **6. Advanced Models** (Week 7) | Train and tune XGBoost, including alternative target transformations | [Read Phase 6](06_advanced_models/README.md) | [XGBoost notebook](06_advanced_models/GxBoost.ipynb) |
| **7. Expanded Evaluation** (Week 8) | Re-evaluate all models with one data version; compare MAPE/MdAPE and price bands | [Read Phase 7](07_evaluation/README.md) | [Evaluation notebook](07_evaluation/6_evaluation.ipynb) |

## How the project fits together

```text
Authorized IDX/CRMLS sale extracts
          |
          v
1. EDA: inspect distributions and single-family sales
          |
          v
2. Cleaning: handle missing values, encode fields, split by closing month
          |                `Completedf.csv`  [export required]
          v
3. Linear Regression: establish a benchmark
          |
          v
4. Decision Tree & Random Forest: compare non-linear approaches
          |
          v
5. Feature Engineering: property features + spatial school-district match
          |                `Completedf_engineered.csv` [export required]
          v
6. XGBoost: test gradient boosting and target transformation
          |
          v
7. Evaluation: re-train models on ONE chosen CSV, compare errors
                   |-- `metrics_summary.csv`
                   `-- `price_band_metrics.csv`
```

## Results so far (from saved notebook outputs)

The exploration phase starts with **175,786 sales records / 80 columns**, and filtering to residential single-family homes yields **86,973 records**. After cleaning, a saved dataframe contains **65,782 records / 46 columns**. The school-district feature-engineering phase expands that to **415 columns** and matches districts to most retained properties.

The early baseline Linear Regression achieved **R² ≈ 0.531** with a six-month training window. Tree-based models scored higher in later exploratory comparisons: the strongest reported engineered Random Forest achieved **R² ≈ 0.826**, and a separately run optimized XGBoost experiment achieved **R² ≈ 0.819**.

**These last two scores are not a valid head-to-head ranking.** The XGBoost notebook used a different engineered dataset version and evaluated only **3,238 January 2024 properties**, while the feature-engineering Random Forest notebook reported **7,069** test properties. The Week 8 evaluation notebook is intended to resolve this issue by using one dataset and the same test rows for all models. It does **not** have final real-data MAPE/MdAPE results yet; run it with an authorized completed CSV to generate them.

## Getting started

1. Clone or download this repository, then choose a phase using the links above.
2. Use Python 3 in Jupyter Notebook or Google Colab. Install common dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Obtain the permitted IDX/CRMLS CSV exports and update the file paths in the first notebooks. The supplied code was largely written for Google Colab and may refer to `/content/`.
4. Run Phase 2 to prepare a consistent cleaned dataset. Save the final dataframe to `Completedf.csv` and put it where downstream notebooks expect it; the provided cleaning notebook **does not automatically save this export**.
5. For Phase 5, download the [California School District Areas 2024–25 boundary layer](https://data.ca.gov/dataset/california-school-district-areas-2024-25/resource/7dfaf005-58eb-45db-93b1-7aff091b2172). Save the engineered output as `Completedf_engineered.csv` for Phase 6 or Phase 7.
6. Run the notebooks in phase order or navigate directly to the Week 8 notebook with one complete, prepared dataset. Phase 7 defaults to `Completedf.csv`; you can switch it to `Completedf_engineered.csv` in its settings cell.

**Note on historical reproducibility:** The notebooks contain exploratory code and some known issues (particularly in the Random Forest result-logging loops); each phase README identifies its limitations. Running every phase unmodified from a fresh session is **not guaranteed** to reproduce the saved outputs without the noted corrections and matching data versions.

## Repository organization

```text
IDX_Real_Estate_Prediction_Repository/
├── README.md                       # Overview and links to all phases
├── requirements.txt                # Python libraries
├── 01_exploration/
│   ├── README.md                   # Detailed EDA documentation
│   └── EDA_Phase.ipynb
├── 02_data_cleaning/
│   ├── README.md
│   └── Data_Cleaning_and_Analysis_Phase_revised.ipynb
├── 03_baseline_model/
│   ├── README.md
│   └── First_Modelp2.ipynb
├── 04_model_comparison/
│   ├── README.md
│   └── Base_Compare (1).ipynb
├── 05_feature_engineering/
│   ├── README.md
│   └── Feature_eng.ipynb
├── 06_advanced_models/
│   ├── README.md
│   └── GxBoost.ipynb
└── 07_evaluation/
    ├── README.md
    └── 6_evaluation.ipynb
```

## Limitations and future improvements

This is an exploratory research workflow, not a deployed home appraisal service. Historical model selection sometimes used the same held-out month for tuning and reporting, input dataset versions differed across phases, and some source-field transformations may need adjustment to avoid leakage. Improving the project would mean using **consistent feature pipelines**, training-only learned transformations, chronological cross-validation, and a genuinely unseen final test month. Errors should be reported both overall and by geography and price band to identify uneven model behavior.

**Navigation:** Start with [Phase 1 — Exploratory Data Analysis](01_exploration/README.md), or skip to the [final evaluation methodology](07_evaluation/README.md).
