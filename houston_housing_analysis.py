"""
Houston Housing Price Analysis by ZIP Code

Author: Elishia Fitzgerald
Description:
This script analyzes Houston housing data to determine whether statistically
significant differences exist in average home prices across ZIP code price groups.
The analysis applies robust statistical methods including Welch’s ANOVA and
Tukey’s HSD to account for non-normal distributions and unequal variances.
"""

#!/usr/bin/env python
# coding: utf-8

# -----------------------------
# 1. Import Required Libraries
# -----------------------------

import warnings
warnings.filterwarnings('ignore')
import pandas as pd # manipulation and analysis
import seaborn as sns #vizualization
import matplotlib.pyplot as plt #vizualization
from scipy.stats import shapiro #normality test
from scipy.stats import levene #levene's test
import statsmodels.api as sm #Welchs ANOVA
from statsmodels.formula.api import ols
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd #Tukey HSD test

# -----------------------------
# 2. Load and Inspect Dataset
# -----------------------------

# Load dataset from local project directory

file_path = "houston_housing_market_2024_light.json"
df = pd.read_json(file_path, encoding="utf-8")

# Initial data inspection (used during exploratory analysis)
df.head()
df.shape
df.columns
df.info()

# -----------------------------
# 3. Data Cleaning and Preparation
# -----------------------------

drop = ['zpid', 'id', 'rawHomeStatusCd', 'marketingStatusSimplifiedCd', 'imgSrc', 'hasImage', 'detailUrl', 'statusType', 'statusText', 'brokerName',
        'countryCurrency', 'isUndisclosedAddress', 'isZillowOwned', 'variableData', 'hdpData', 'isSaved', 'isUserClaimingOwner',
        'isUserConfirmedClaim', 'pgapt', 'sgapt', 'shouldShowZestimateAsPrice', 'has3DModel', 'hasVideo', 'isHomeRec', 'hasAdditionalAttributions',
        'isFeaturedListing', 'isShowcaseListing', 'list', 'relaxed', 'carouselPhotos', 'hasOpenHouse', 'openHouseStartDate', 'openHouseEndDate',
        'openHouseDescription', 'lotAreaString','providerListingId', 'builderName', 'streetViewURL', 'streetViewMetadataURL', 'isPropertyResultCDP',
        'flexFieldText', 'flexFieldType', 'info3String','info6String', 'info2String', 'availabilityDate']

df = df.drop(columns=drop)

df.rename(columns={
    'price': 'Price$',                 
    'unformattedPrice': 'PriceUnformatted',          
    'address': 'FullAddress',
    'addressStreet': 'Street',
    'addressCity': 'City',
    'addressState': 'State',
    'addressZipcode': 'Zipcode',
    'beds': 'Bedroom#',
    'baths': 'Bathroom#',
    'area': 'Area(sqft)',
    'latLong': 'Lat&Long',
    'zestimate': 'ZillowPriceEstimate'
}, inplace=True)

df.info()

# -----------------------------
# 4. Exploratory Data Analysis
# -----------------------------

df.describe()

df[['PriceUnformatted', 'Bedroom#', 'Bathroom#', 'Area(sqft)', 'ZillowPriceEstimate']].hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()

sns.scatterplot(data=df, x='Area(sqft)', y='PriceUnformatted')
plt.title('Price vs Living Area')
plt.show()

sns.heatmap(df[['PriceUnformatted', 'Bedroom#', 'Bathroom#', 'Area(sqft)', 'ZillowPriceEstimate']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Group ZIP codes into price tiers to support comparative market analysis

zip_medians = df.groupby('Zipcode')['PriceUnformatted'].median()

# Define ZIP code price tiers based on quartile distribution

low = zip_medians.quantile(0.25)
medium = zip_medians.quantile(0.50)
high = zip_medians.quantile(0.75)

df['ZipMedianPrice'] = df['Zipcode'].map(zip_medians)

def assign_price_group(median_price):
    if median_price <= low:
        return 'Low'
    elif low < median_price <= high:
        return 'Medium'
    else:
        return 'High'

df['PriceGroup'] = df['ZipMedianPrice'].apply(assign_price_group)

df.head()

group_medians = df.groupby('PriceGroup')['PriceUnformatted'].median()
df['GroupMedianPrice'] = df['PriceGroup'].map(group_medians)

fill_cols = ['Bedroom#', 'Bathroom#', 'Area(sqft)']

# Impute missing values using hierarchical median strategy
for col in fill_cols:
    df[col] = df.groupby(['Zipcode', 'PriceGroup'])[col].transform(lambda x: x.fillna(x.median()))
    df[col] = df.groupby('Zipcode')[col].transform(lambda x: x.fillna(x.median()))
    df[col].fillna(df[col].median(), inplace=True)

df.isnull().any()

df[fill_cols + ['Zipcode', 'PriceGroup']].groupby(['Zipcode', 'PriceGroup']).apply(lambda x: x.isnull().sum())

df['Zipcode'].isnull().sum()

df = df[df['Zipcode'].notnull()]

df['Zipcode'].isnull().sum()

(df['PriceUnformatted'] == df['ZillowPriceEstimate']).all()

df['PriceUnformatted'].corr(df['ZillowPriceEstimate'])

df.drop(['Price$', 'ZillowPriceEstimate', 'ZipMedianPrice', 'Lat&Long'], axis=1, inplace=True)

df.head()

plt.figure(figsize=(10, 6))
sns.boxplot(x='PriceGroup', y='PriceUnformatted', data=df)
plt.title('Home Prices by ZIP Code Price Group')
plt.xlabel('Zipcode Price Group')
plt.ylabel('Home Price')
plt.tight_layout()
plt.show()

# -----------------------------
# 5. Statistical Testing
# -----------------------------

for group in ['Low', 'Medium', 'High']:
    stat, p = shapiro(df[df['PriceGroup'] == group]['PriceUnformatted'])
    print(f"{group} Group: W = {stat:.4f}, p = {p:.4f}")

low = df[df['PriceGroup'] == 'Low']['PriceUnformatted']
med = df[df['PriceGroup'] == 'Medium']['PriceUnformatted']
high = df[df['PriceGroup'] == 'High']['PriceUnformatted']

stat, p = levene(low, med, high)
print(f"Levene’s Test: Stat = {stat:.4f}, p = {p:.4f}")

# Welch’s ANOVA is used due to unequal variances across price groups
model = ols('PriceUnformatted ~ PriceGroup', data=df).fit()
welch_anova = sm.stats.anova_lm(model, typ=2, robust='hc3')
print(welch_anova)

# -----------------------------
# 6. Results Interpretation
# -----------------------------

tukey = pairwise_tukeyhsd(
    endog=df['PriceUnformatted'],
    groups=df['PriceGroup'],
    alpha=0.05)
print(tukey.summary())

# Results support ZIP code segmentation for improved pricing strategies

# Export cleaned dataset for reporting and downstream analysis
df.to_csv('cleaned_houston_housing_data.csv', index=False)
