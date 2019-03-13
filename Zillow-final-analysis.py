
# coding: utf-8

# In[3]:


# Import the important libraries
import pandas as pd


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


# current directory
get_ipython().system('pwd')


# In[6]:


# list of the files in the directory
get_ipython().system('ls')


# ### Load properties 2016 and 2017 together

# In[6]:


df_properties_16_17 = pd.concat([pd.read_csv("properties_2016.csv"), pd.read_csv("properties_2017.csv")])
pd.set_option('display.max_columns', None)


# In[7]:


df_properties_16_17.head()


# In[10]:


df_properties_16_17.shape


# In[11]:


df_properties_16_17.dtypes


# In[12]:


df_properties_16_17.isna().sum()


# In[13]:


df_properties_16_17.isnull().sum()


# ### Let's count the missing value and also reset the index

# In[7]:


df_missing_values = df_properties_16_17.isnull().sum(axis=0).reset_index()
df_missing_values.columns=['column_name', 'missing_count']
df_missing_values=df_missing_values.ix[df_missing_values['missing_count']>0]
df_missing_values=df_missing_values.sort_values(by='missing_count')


# In[9]:


df_missing_values.head(10)


# ### visualize the missing value count

# In[8]:


#  Let's plot the missing values
ind = np.arange(df_missing_values.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize = (12,18))
rects = ax.barh(ind, df_missing_values.missing_count.values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(df_missing_values.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of Missing values")
ax.set_title("Missing values in each column")
plt.show()


# In[9]:


df_properties_16_17.columns


# ### make a new file of relevant columns with least missing values

# In[10]:


df_selected_columns = df_properties_16_17[['rawcensustractandblock','lotsizesquarefeet','latitude', 'longitude', 'regionidcounty', 'fips',
                                         'propertylandusetypeid', 'assessmentyear', 'bedroomcnt', 'bathroomcnt', 'roomcnt',
                                         'regionidzip', 'taxamount', 'landtaxvaluedollarcnt',
                                          'regionidcity', 'yearbuilt', 'structuretaxvaluedollarcnt', 'calculatedfinishedsquarefeet',
                                          'taxvaluedollarcnt']]


# In[11]:


df_selected_columns.head(10)


# In[123]:


df_selected_columns.isna().sum()


# In[83]:


df_selected_columns.dtypes


# In[124]:


df_selected_columns.shape


# In[125]:


df_selected_columns.lotsizesquarefeet.describe()


# In[12]:


# nba["College"].fillna("No College", inplace = True
df_selected_columns['lotsizesquarefeet'].fillna(0, inplace=True)                      


# In[13]:


df_selected_columns['lotsizesquarefeet'].fillna(df_selected_columns['lotsizesquarefeet'].mean(), inplace=True)


# In[14]:


df_selected_columns.lotsizesquarefeet.isna().sum()


# ### Drop all NaNs

# In[15]:


df_no_na = df_selected_columns.dropna()


# In[16]:


df_no_na.isna().sum()


# In[131]:


df_no_na.shape


# ### Create an apprasied value column combining land appraised value and built structure value

# In[17]:


df_no_na['appraised_value_y'] = df_no_na['structuretaxvaluedollarcnt'] + df_no_na['landtaxvaluedollarcnt'] 


# In[19]:


df_no_na.head(10)


# In[28]:


df_no_na.shape


# ### drop these columns

# In[18]:


df_no_na = df_no_na.drop(columns=['rawcensustractandblock','structuretaxvaluedollarcnt','taxvaluedollarcnt','landtaxvaluedollarcnt' ])


# ### Rename all the columns

# In[19]:


df_no_na.columns


# In[20]:


df_no_na = df_no_na.rename(columns = {'lotsizesquarefeet':'TOTAL_LOT_AREA_SQFT','latitude': 'LATITUDE', 'longitude' : 'LONGITUDE', 'regionidcounty':'COUNTYID', 'fips':'FED_CODE',
       'propertylandusetypeid' :'LAND_USE_TYPE', 'assessmentyear' :'YEAR_ASSESSMENT', 'bedroomcnt':'BEDS', 'bathroomcnt': 'BATHS',
       'roomcnt' : 'TOTAL_ROOMS', 'regionidzip' :'ZIP', 'taxamount' :'ASSESSED_PROPERTY_TAXES',
       'regionidcity': 'CITY_ID', 'yearbuilt' :'YEAR_BUILT', 'calculatedfinishedsquarefeet' :'SQAURE_FEET_HOUSE',
       'appraised_value_y' :'APPRAISED_VALUE'})


# In[21]:


df_no_na.columns


# In[22]:


df_no_na.dtypes


# ### Change the data type of some columns

# In[23]:


df_no_na['COUNTYID'] = df_no_na['COUNTYID'].astype(np.int)
df_no_na['FED_CODE'] = df_no_na['FED_CODE'].astype(np.int)
df_no_na['LAND_USE_TYPE'] = df_no_na['LAND_USE_TYPE'].astype(np.int)
df_no_na['BEDS'] = df_no_na['BEDS'].astype(np.int)
df_no_na['TOTAL_ROOMS'] = df_no_na['TOTAL_ROOMS'].astype(np.int)
df_no_na['ZIP'] = df_no_na['ZIP'].astype(np.int)
df_no_na['YEAR_BUILT'] = df_no_na['YEAR_BUILT'].astype(np.int)
df_no_na['CITY_ID'] = df_no_na['CITY_ID'].astype(np.int)
df_no_na['YEAR_ASSESSMENT'] = df_no_na['YEAR_ASSESSMENT'].astype(np.int)


# In[ ]:


# df_no_na_int = df_no_na.astype(np.int)       


# In[24]:


df_no_na.head()


# In[79]:


df_no_na.shape


# In[25]:


df_no_na.corr()


# In[142]:


df_no_na.columns


# ### pair plots (taking very long time, skip this for now)

# In[ ]:


import seaborn as sns 
cols = ['APPRAISED_VALUE','TOTAL_LOT_AREA_SQFT',
        'BEDS', 'BATHS', 'TOTAL_ROOMS',
       'ASSESSED_PROPERTY_TAXES', 'YEAR_BUILT', 'SQAURE_FEET_HOUSE'
       ]

sns.pairplot(df_no_na[cols], size=2.5)
plt.tight_layout()
plt.show()


# In[26]:


df_no_na.COUNTYID.value_counts()


# ### Define X matrix and y vector

# In[28]:


# let's define our X and y
X = df_no_na[['TOTAL_LOT_AREA_SQFT', 'LATITUDE', 'LONGITUDE', 'COUNTYID', 'FED_CODE',
       'LAND_USE_TYPE', 'YEAR_ASSESSMENT', 'BEDS', 'BATHS', 'TOTAL_ROOMS',
       'ZIP', 'ASSESSED_PROPERTY_TAXES', 'CITY_ID', 'YEAR_BUILT',
       'SQAURE_FEET_HOUSE',
        ]]


# In[29]:


X.head()


# ### y vector

# In[30]:


y = df_no_na['APPRAISED_VALUE']


# In[31]:


y.head()


# In[ ]:


X.shape, y.shape


# ### Split the data into train and test sets

# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)


# In[33]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


y_test


# ### Simple Linear Regression as a baseline model

# In[34]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)


# In[35]:


y_test_predict = model.predict(X_test)


# In[36]:


y_test_predict.shape


# In[37]:


MSE = mean_squared_error(y_test, y_test_predict)

RMSE = (np.sqrt(mean_squared_error(y_test, y_test_predict)))

print('MSE is {}'.format(MSE))
print('RMSE is {}'.format(RMSE))

R2 = r2_score(y_test, y_test_predict)
print("Test set score: {:.2f}".format(model.score(X_test, y_test)))
print('R^2 is {}'.format(R2))
print("Coefficient: \n", model.coef_)
print("\n Intercept: ", model.intercept_)


# ## Random Forests

# In[39]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

X = df_no_na[['TOTAL_LOT_AREA_SQFT', 'LATITUDE', 'LONGITUDE', 'COUNTYID', 'FED_CODE',
       'LAND_USE_TYPE', 'YEAR_ASSESSMENT', 'BEDS', 'BATHS', 'TOTAL_ROOMS',
       'ZIP', 'ASSESSED_PROPERTY_TAXES', 'CITY_ID', 'YEAR_BUILT',
       'SQAURE_FEET_HOUSE',
        ]]
y = df_no_na['APPRAISED_VALUE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, shuffle=True)


m = RandomForestRegressor(max_depth=7, random_state=42, n_estimators=100)
m.fit(X_train, y_train)
y_pred = m.predict(X_test)


# In[50]:


MSE = mean_squared_error(y_test, y_pred)

RMSE = (np.sqrt(mean_squared_error(y_test, y_pred)))

print("Test set score: {:.2f}".format(m.score(X_test, y_test)))
print('MSE is {}'.format(MSE))
print('RMSE is {}'.format(RMSE))

R2 = r2_score(y_test, y_pred)

print('R^2 is {}'.format(R2))


# ### Our linear regression seems to be doing better than Random forests

# ## Let's remove the outliers

# In[43]:


from scipy import stats


# In[44]:


df_no_outliers = df_no_na[(np.abs(stats.zscore(df_no_na))<3).all(axis=1)]


# In[45]:


df_no_outliers.head()


# In[46]:


df_no_outliers.shape


# In[48]:


df_no_outliers.corr()


# ### Random Forests with outliers removed

# In[51]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

X = df_no_outliers[['TOTAL_LOT_AREA_SQFT', 'LATITUDE', 'LONGITUDE', 'COUNTYID', 'FED_CODE',
       'LAND_USE_TYPE', 'YEAR_ASSESSMENT', 'BEDS', 'BATHS', 'TOTAL_ROOMS',
       'ZIP', 'ASSESSED_PROPERTY_TAXES', 'CITY_ID', 'YEAR_BUILT',
       'SQAURE_FEET_HOUSE',
        ]]
y = df_no_outliers['APPRAISED_VALUE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, shuffle=True)


m = RandomForestRegressor(max_depth=7, random_state=42, n_estimators=100)
m.fit(X_train, y_train)
y_pred = m.predict(X_test)


# In[52]:


MSE = mean_squared_error(y_test, y_pred)

RMSE = (np.sqrt(mean_squared_error(y_test, y_pred)))

print("Test set score: {:.2f}".format(m.score(X_test, y_test)))
print('MSE is {}'.format(MSE))
print('RMSE is {}'.format(RMSE))

R2 = r2_score(y_test, y_pred)

print('R^2 is {}'.format(R2))


# ### Simple Linear Regression model with outlier removed
# 
# I also removed a few features such as longitude, latitude, fed_code, city_id etc. the result is a little worse than before in terms of RMSE. But Test score is still on with 94% 

# In[61]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = df_no_outliers[['TOTAL_LOT_AREA_SQFT',
       'LAND_USE_TYPE', 'YEAR_ASSESSMENT', 'BEDS', 'BATHS', 'TOTAL_ROOMS',
       'ZIP', 'ASSESSED_PROPERTY_TAXES','YEAR_BUILT',
       'SQAURE_FEET_HOUSE',
        ]]
y = df_no_outliers['APPRAISED_VALUE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, shuffle=True)

model = LinearRegression()
model.fit(X_train, y_train)
y_test_predict = model.predict(X_test)


# In[62]:


MSE = mean_squared_error(y_test, y_test_predict)

RMSE = (np.sqrt(mean_squared_error(y_test, y_test_predict)))

print('MSE is {}'.format(MSE))
print('RMSE is {}'.format(RMSE))

R2 = r2_score(y_test, y_test_predict)
print("Test set score: {:.2f}".format(model.score(X_test, y_test)))
print('R^2 is {}'.format(R2))
print("Coefficient: \n", model.coef_)
print("\n Intercept: ", model.intercept_)


# ### Random Forests with Features removed

# In[63]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

X = df_no_outliers[['TOTAL_LOT_AREA_SQFT',
       'LAND_USE_TYPE', 'YEAR_ASSESSMENT', 'BEDS', 'BATHS', 'TOTAL_ROOMS',
       'ZIP', 'ASSESSED_PROPERTY_TAXES', 'YEAR_BUILT',
       'SQAURE_FEET_HOUSE',
        ]]
y = df_no_outliers['APPRAISED_VALUE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, shuffle=True)


m = RandomForestRegressor(max_depth=7, random_state=42, n_estimators=100)
m.fit(X_train, y_train)
y_pred = m.predict(X_test)


# In[72]:


v = [ 7736.0, 261, 2015, 4, 2.0, 0, 96505, 3402.94, 1963, 1288.0]


# In[73]:


len(v)


# In[64]:


MSE = mean_squared_error(y_test, y_pred)

RMSE = (np.sqrt(mean_squared_error(y_test, y_pred)))

print("Test set score: {:.2f}".format(m.score(X_test, y_test)))
print('MSE is {}'.format(MSE))
print('RMSE is {}'.format(RMSE))

R2 = r2_score(y_test, y_pred)

print('R^2 is {}'.format(R2))


# #### In conclusion, the random forests with features removed seems to be more reliable and decent

# In[80]:


df_no_outliers.dtypes


# ### The final Model

# In[78]:


RF_regression_model =m.fit(X,y)


# In[79]:


RF_regression_model


# ### Pickle the file 

# In[65]:


import pickle


# In[66]:


pickle.dump(m, open('zillow_model.p','wb'))


# ### define a method for a .py file that will go with the backend

# In[ ]:


def estimate_value(house_features):
    m = pickle.load(open('zillow_model.p', 'rb'))
    appraised_value = m.predict(house_features)
    return appraised_value

