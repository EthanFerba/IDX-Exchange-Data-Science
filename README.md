# IDX-Exchange-Data-Science
# Exploratory Data Analysis — California Residential Sales

## Project Overview

This notebook focuses on exploring residential real estate data acquired from the IDX database. The primary goal of this phase was to gain a better understanding of the dataset, examine important property characteristics, and identify potential relationships that could help with future predictive modeling.

The analysis focuses on `ClosePrice`, which serves as the target variable for the larger project. Additional attributes, including living area, bedrooms, bathrooms, lot size, year built, and days on market, were explored to better understand how different property characteristics relate to residential sale prices.

## Data Collection and Preparation

The original assignment required a minimum of six months of data. To provide a broader view of the housing market, two datasets covering January 2022 through January 2024 were loaded and combined using pandas.

The combined dataset contained **175,786 records and 80 attributes**. Initial exploration was conducted using pandas functions to examine the structure, available columns, and descriptive statistics.

After reviewing the available attributes, the dataset was reduced to 45 columns considered potentially useful for further analysis. The records were then filtered to include only properties with `PropertyType` equal to `Residential` and `PropertySubType` equal to `SingleFamilyResidence`.

This filtering process resulted in **86,973 residential single-family property records**.

## Exploratory Analysis

Several visualizations were developed to better understand the data.

The closing-price histogram was used to investigate the distribution of residential sale prices. The plot showed a right-skewed distribution, indicating that higher-priced properties extend the upper end of the market. The most extreme 1% of closing prices were excluded from this visualization to improve readability.

Additional plots investigated the relationship between living area and closing price, days on market and changes from original listing price, and year built and closing price. An interactive geographic scatter map was also created to examine property values across different locations.

For several of these plots, the upper 5% of closing prices, living areas, and lot sizes were excluded to reduce the influence of extreme values.

## Preliminary Findings

The analysis suggests that living area has a positive relationship with closing price, although properties with similar square footage can still sell at substantially different prices. The data also contains several unusual values, particularly in the price-change analysis, that should be investigated before modeling.

These preliminary observations helped identify possible predictors of closing price and highlighted areas that may require additional cleaning and preprocessing.

## Next Steps

The next phase will focus on evaluating missing values, investigating outliers, correcting inconsistent data types, and preparing the selected variables for predictive modeling.

Additional distributions for bedrooms, bathrooms, living area, and lot size should also be added to satisfy the original exploration requirements.

The findings in this notebook are exploratory and do not represent the performance of a trained predictive model.
