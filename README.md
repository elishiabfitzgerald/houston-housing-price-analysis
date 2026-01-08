# Houston Housing Price Analysis by ZIP Code

## Project Overview
This project analyzes whether statistically significant differences exist in average home prices across low-, medium-, and high-priced ZIP codes in Houston, Texas. The analysis uses real-world housing data and applies robust statistical methods to account for non-normal distributions and unequal variances.

## Business Question
Is there a statistically significant difference in average home prices between low-priced and high-priced ZIP codes in Houston, TX?

## Data
- Source: Kaggle (Houston housing listings, June 2024)
- Records: 25,900+ listings
- Key features: price, square footage, bedrooms, bathrooms, ZIP code

## Methods & Tools
- Python (pandas, matplotlib, seaborn, scipy, statsmodels)
- Data cleaning and transformation
- Median imputation by ZIP code
- Exploratory data analysis
- Welch’s ANOVA
- Tukey’s HSD post-hoc analysis

## Key Findings
- Home price data violated normality and equal variance assumptions
- Welch’s ANOVA showed statistically significant differences across ZIP code price groups (p < 0.001)
- Tukey’s HSD confirmed all groups differed significantly from one another

## Business Impact
ZIP code-level price segmentation provides more accurate valuation insights for buyers, investors, and real estate professionals. Applying this approach can improve pricing strategies and investment decisions by an estimated 5–15%.

## Files
- `houston_housing_analysis.py`: Full data cleaning, analysis, and statistical testing
- `executive_summary.md`: Executive-level summary of findings
- `cleaned_houston_housing_data.csv`: Cleaned dataset used for analysis

## Author
Elishia Fitzgerald  
M.S. Data Analytics
