#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import packages and libraries
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


# In[2]:


#set pandas option to display all columns
pd.set_option('display.max_rows', None)


# In[3]:


#import data

file_path = r"C:\Users\elish\Downloads\archive\capstonedf\houston_housing market 2024_light.json"

df = pd.read_json(file_path, encoding="utf-8")


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.info()


# In[8]:


#drop unnecessary columns
drop = ['zpid', 'id', 'rawHomeStatusCd', 'marketingStatusSimplifiedCd', 'imgSrc', 'hasImage', 'detailUrl', 'statusType', 'statusText', 'brokerName',
        'countryCurrency', 'isUndisclosedAddress', 'isZillowOwned', 'variableData', 'hdpData', 'isSaved', 'isUserClaimingOwner',
        'isUserConfirmedClaim', 'pgapt', 'sgapt', 'shouldShowZestimateAsPrice', 'has3DModel', 'hasVideo', 'isHomeRec', 'hasAdditionalAttributions',
        'isFeaturedListing', 'isShowcaseListing', 'list', 'relaxed', 'carouselPhotos', 'hasOpenHouse', 'openHouseStartDate', 'openHouseEndDate',
        'openHouseDescription', 'lotAreaString','providerListingId', 'builderName', 'streetViewURL', 'streetViewMetadataURL', 'isPropertyResultCDP',
        'flexFieldText', 'flexFieldType', 'info3String','info6String', 'info2String', 'availabilityDate']

df = df.drop(columns=drop)


# In[9]:


#check shape
df.shape


# In[10]:


df.head()


# In[11]:


#rename columns for better understanding
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


# In[12]:


df.info()


# In[13]:


#check nulls # will fill nulls once grouped for better accuracy, fits the research question
df.isnull().any()


# In[14]:


df.describe()


# In[15]:


#visualizing the distribution of the numeric columns #$- right skewed, Bed-normal dist., Bath-normal dist., Area- right skewed, Zillow- right skewed #outliers present
df[['PriceUnformatted', 'Bedroom#', 'Bathroom#', 'Area(sqft)', 'ZillowPriceEstimate']].hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()


# In[16]:


#visualizing relationships #positive relationship(area increases, prive increases) #not a tight cluster-high varience #outliers present
sns.scatterplot(data=df, x='Area(sqft)', y='PriceUnformatted')
plt.title('Price vs Living Area')
plt.show()


# In[17]:


#visualizing relationships
sns.heatmap(df[['PriceUnformatted', 'Bedroom#', 'Bathroom#', 'Area(sqft)', 'ZillowPriceEstimate']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[18]:


#calculating the median based on Zip
zip_medians = df.groupby('Zipcode')['PriceUnformatted'].median()


# In[19]:


#creating quartiles, since there are three quartiles in housing
low = zip_medians.quantile(0.25)
medium = zip_medians.quantile(0.50)
high = zip_medians.quantile(0.75)


# In[20]:


# putting zip medians back to the main DataFrame
df['ZipMedianPrice'] = df['Zipcode'].map(zip_medians)


# In[21]:


#assigning groups
def assign_price_group(median_price):
    if median_price <= low:
        return 'Low'
    elif low < median_price <= high:
        return 'Medium'
    else:
        return 'High'


# In[22]:


#assigning groups
df['PriceGroup'] = df['ZipMedianPrice'].apply(assign_price_group)


# In[23]:


df.head()


# In[24]:


#filling nulls in each group created with the medians of the groups
group_medians = df.groupby('PriceGroup')['PriceUnformatted'].median()
df['GroupMedianPrice'] = df['PriceGroup'].map(group_medians)


# In[25]:


# handling nulls #not zillowestimate yet, may be dropped before test
fill_cols = ['Bedroom#', 'Bathroom#', 'Area(sqft)']

for col in fill_cols:
    df[col] = df.groupby(['Zipcode', 'PriceGroup'])[col].transform(lambda x: x.fillna(x.median()))


# In[26]:


#double check #there are still nulls...
df.isnull().any()


# In[27]:


#handling nulls again #(Trying to Fill Null Values With Sub-grouped Mean Value Using Pandas Fillna() and Groupby().Transform() Is Doing Nothing With the Null Values, n.d.)
fill_cols = ['Bedroom#', 'Bathroom#', 'Area(sqft)']

for col in fill_cols:
    df[col] = df.groupby(['Zipcode', 'PriceGroup'])[col].transform(lambda x: x.fillna(x.median()))
    df[col] = df.groupby('Zipcode')[col].transform(lambda x: x.fillna(x.median()))
    df[col].fillna(df[col].median(), inplace=True)


# In[28]:


#double check #nulls have been filled
df.isnull().any()


# In[29]:


#there are columns with no zipcodes, which is main feature needed. Will have to drop.
df[fill_cols + ['Zipcode', 'PriceGroup']].groupby(['Zipcode', 'PriceGroup']).apply(lambda x: x.isnull().sum())


# In[30]:


#dropping rows with no zipcode
df['Zipcode'].isnull().sum()


# In[31]:


#dropping rows with no zipcode #necessary step due to the research question
df = df[df['Zipcode'].notnull()]


# In[32]:


#check if any nulls still exist
df['Zipcode'].isnull().sum()


# In[33]:


#checking if columns are exactly the same
(df['PriceUnformatted'] == df['ZillowPriceEstimate']).all()


# In[34]:


#checking how similar they are #0.998 is way to high, could cause mulitcollinearity, a column needs to be removed to avoid
df['PriceUnformatted'].corr(df['ZillowPriceEstimate'])


# In[35]:


#dropping Prce$, same as PriceUnformatted but a string due to the '$' and ',' #dropping all unneeded columns before stat test
df.drop(['Price$', 'ZillowPriceEstimate', 'ZipMedianPrice', 'Lat&Long'], axis=1, inplace=True)


# In[36]:


df.head()


# In[37]:


#viewing group distributions
plt.figure(figsize=(10, 6))
sns.boxplot(x='PriceGroup', y='PriceUnformatted', data=df)
plt.title('Home Prices by ZIP Code Price Group')
plt.xlabel('Zipcode Price Group')
plt.ylabel('Home Price')
plt.tight_layout()
plt.show()


# In[38]:


#normality test using Shapiro #(One-way ANOVA With Python, n.d.)
for group in ['Low', 'Medium', 'High']:
    stat, p = shapiro(df[df['PriceGroup'] == group]['PriceUnformatted'])
    print(f"{group} Group: W = {stat:.4f}, p = {p:.4f}")


# In[39]:


#testing ANOVA assumptions using levene's Test #(Reddy, 2023)
low = df[df['PriceGroup'] == 'Low']['PriceUnformatted']
med = df[df['PriceGroup'] == 'Medium']['PriceUnformatted']
high = df[df['PriceGroup'] == 'High']['PriceUnformatted']

stat, p = levene(low, med, high)
print(f"Levene’s Test: Stat = {stat:.4f}, p = {p:.4f}")


# In[40]:


#running Welch ANOVA since levene's test failed #(One-way ANOVA With Python, n.d.)
model = ols('PriceUnformatted ~ PriceGroup', data=df).fit()
welch_anova = sm.stats.anova_lm(model, typ=2, robust='hc3')
print(welch_anova)


# In[41]:


#running Tukey's HSD to determine difference and its significance #(Reddy, 2023)
tukey = pairwise_tukeyhsd(
    endog=df['PriceUnformatted'],
    groups=df['PriceGroup'],
    alpha=0.05)
print(tukey.summary())


# In[42]:


#cleaned df saved to CSV for submission
df.to_csv('cleaned_houston_housing_data.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




