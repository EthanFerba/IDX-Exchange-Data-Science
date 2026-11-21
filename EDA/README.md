# Exploratory Data Analysis — California Residential Sales

[← Project overview](../README.md) | [Next: Data Cleaning and Preparation](../02_data_cleaning/README.md)


**Notebook:** [`EDA_Phase.ipynb`](EDA_Phase.ipynb)  
**Project phase:** Week 2 — Data Exploration  
**Data source:** IDX / CRMLS sold-property extracts  
**Primary target variable:** `ClosePrice`

## Objective

The goal of this notebook is to understand the structure and quality of the available property-sales data before developing a predictive model. This phase examines the distribution of closing prices, explores how selected property characteristics relate to sale price, and identifies possible data-quality issues that may affect later preprocessing and modeling.

The assignment requires at least six months of data and focuses on residential single-family properties. This notebook uses two extracts whose filenames indicate coverage from January 2022 through January 2024 (approximately **25 calendar months**). The exact observed date range should be confirmed using `CloseDate`; the notebook does not currently verify it.

## Data sources

Two CSV extracts are loaded from Google Colab's `/content/` directory:

1. `CRMLSSold20220101_20231231_filled(first 2 years).csv` — labeled as January 2022 through December 2023.
2. `CRMLSSold202401_filled(january_2024).csv` — labeled as January 2024.

The notebook combines the files with `pd.concat(..., ignore_index=True)`. The combined dataset has **175,786 rows and 80 columns**, as reported by the saved notebook output. The data files themselves are not included with this documentation. Anyone reproducing the notebook must obtain authorized access to equivalent extracts and place them at the configured input paths.

## Tools and environment

The notebook was written for Python 3 in a Jupyter / Google Colab environment. It imports:

- **pandas** for loading and manipulating tabular data
- **NumPy** for numerical operations
- **Matplotlib** and **Seaborn** for static visualizations
- **Plotly Express** for an interactive geographic plot

Install the packages if needed:

```bash
pip install pandas numpy matplotlib seaborn plotly
```

Versions are not pinned in the notebook, so package changes may affect appearance or compatibility.

## Analysis workflow

### 1. Load and inspect the data

Load the two source CSVs, concatenate them into `mdf`, and inspect the first rows, dataset shape, column names, and numerical summary statistics with `head()`, `shape`, `columns`, and `describe()`.

The CSV import produces a mixed-type warning for two columns. This does not stop the saved notebook, but data types should be reviewed if these columns are used later.

### 2. Select relevant columns

Create `reducedmdf` by keeping 45 fields potentially useful for analysis, including sale prices, location, property dimensions, number of bedrooms and bathrooms, listing information, year built, and property type. This selection reduces the number of columns but does not change the row count: **175,786 rows and 45 columns**.

Column selection at this stage is for exploration, not final feature selection. Some columns that are currently retained may be excluded from prediction models later, particularly if they would not be known at prediction time.

### 3. Restrict the property population

Filter the data to the required property categories:

- `PropertyType == "Residential"`
- `PropertySubType == "SingleFamilyResidence"`

The filtered dataframe, `rdf`, contains **86,973 rows and 45 columns** in the saved output. All charts are based on this residential single-family population or a further visualization-specific subset.

### 4. Create exploratory visualizations

| Visualization | How it is constructed | Purpose |
| --- | --- | --- |
| **Closing price distribution** | Histogram and KDE of `ClosePrice`, excluding values at or above its 99th percentile for this plot | Examine typical price ranges and skew without the most expensive transactions dominating the axis |
| **Days on market vs. price change** | Scatterplot and fitted trend line of `DaysOnMarket` against `PriceDropPercent` | Explore whether listing duration is associated with selling below or above the original list price |
| **Year built vs. closing price** | Hexbin density plot of `YearBuilt` and `ClosePrice` | Find concentrations of transactions across construction years and price ranges |
| **Geographic sales map** | Interactive Plotly scatter map, colored by `ClosePrice` and sized by `LivingArea` | Explore geographic variation in property prices and sizes |
| **Living area vs. closing price** | Scatterplot with fitted regression line and 95% confidence interval | Explore the relationship between interior living area and sale price |

**Visualization filters:** The closing-price histogram excludes the top 1% of `ClosePrice` values for readability. The other four plots use `market_df`, which keeps rows below the 95th percentile of **all three** variables: `ClosePrice`, `LivingArea`, and `LotSizeSquareFeet`. Rows missing one of those three values also fall out of this subset. These filters help display common-range properties, but the resulting charts do not represent the full distribution of all filtered sales.

**Price-change definition:**

```text
PriceDropPercent = ((OriginalListPrice - ClosePrice) / OriginalListPrice) * 100
```

A positive value indicates a closing price below the original list price; a negative value indicates a sale above it. Missing or zero original list prices and implausible extreme values must be checked before interpreting the relationship.

## Preliminary observations

- **Closing prices are right-skewed.** The histogram contains a dense group of lower-to-mid-priced sales and a longer tail of higher-priced properties, even after removing the highest 1% for plotting.
- **Living area shows a positive overall association with closing price.** The fitted line rises with square footage, although there is substantial variation at similar sizes. This visualization alone does not establish predictive performance or causation.
- **The price-change scatterplot contains extreme negative percentages.** These may reflect unusual transactions, incorrect listing prices, or other source-data issues. They make the overall trend difficult to interpret without validation.
- **Property year and location may be useful to investigate further.** The density plot and interactive map show how transactions can be grouped by construction year, location, price, and size. Additional analysis would be required to isolate their effects.

These are exploratory interpretations, not results from a trained predictive model.

## How to reproduce

1. Download or clone the repository and open `EDA_Phase.ipynb` in Jupyter or Google Colab.
2. Obtain the two authorized IDX/CRMLS source CSVs listed above.
3. In Colab, upload both CSVs into `/content/`. If running locally, update the `pd.read_csv(...)` paths to point to your data directory.
4. Install the packages listed above, then run the notebook from top to bottom.
5. Confirm that the combined dataframe and single-family filtered dataframe have the expected columns and plausible date ranges. Row counts may differ if a different extract or refresh is used.
6. Review the generated static charts and interactive Plotly map. Rendering the map may depend on the notebook environment and Plotly version.

To verify the actual transaction period, add the following check after `rdf` is created:

```python
close_dates = pd.to_datetime(rdf["CloseDate"], errors="coerce")
print("Earliest close:", close_dates.min())
print("Latest close:", close_dates.max())
print("Months represented:", close_dates.dt.to_period("M").nunique())
```

## Limitations and remaining Week 2 requirements

The current notebook is a useful first EDA pass, but the original assignment also requests **distributions** of `LivingArea`, bedrooms, bathrooms, and lot size. Their corresponding fields are available (`LivingArea`, `BedroomsTotal`, `BathroomsTotalInteger`, and `LotSizeSquareFeet`), but separate distribution plots are **not yet included**. A living-area scatterplot is not a substitute for a univariate distribution.

Recommended follow-up checks before modeling include:

- Add histograms for the four missing required variables.
- Summarize missing values, invalid data types, and duplicate listing identifiers before dropping IDs needed for duplicate checks.
- Investigate nonpositive prices and living areas, extreme lot sizes and property sizes, unusual `DaysOnMarket` values, invalid coordinates, and zero `OriginalListPrice` values.
- Convert `CloseDate` to a datetime type and examine monthly coverage or possible seasonal patterns.
- Define and document consistent outlier-handling rules rather than relying only on plotting cutoffs.
- Avoid including future-known or transaction-outcome variables as predictors when developing a model intended to predict closing price before a sale.

**Data-access note:** IDX/MLS extracts may be subject to licensing or sharing restrictions and can contain contact details or other sensitive fields. Do not publicly commit the raw CSVs or identifying data unless their use and publication are authorized.
