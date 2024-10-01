#!/usr/bin/env python
# coding: utf-8

# ## Dragon Real Estate - Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("C://Users/saura/Downloads/BostonHousing.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing.isna().sum()


# In[6]:


housing.describe()


# In[7]:


housing['chas'].value_counts()


# In[8]:
%matplotlib inline


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


housing.hist(bins=50, figsize=(20,15))


# # Train-Test Splitting

# In[11]:


import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


# In[12]:


train_set, test_set = split_train_test(housing, 0.2)


# In[13]:


print(f"Row's in Train Set: {len(train_set)}\nRows in Test Set: {len(test_set)}\n")


# In[14]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)
print(f"Row's in Train Set: {len(train_set)}\nRows in Test Set: {len(test_set)}\n")


# In[15]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits =1, test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing,housing['chas']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[16]:


strat_test_set.info()


# In[17]:


strat_test_set['chas'].value_counts()


# In[18]:


strat_train_set['chas'].value_counts()


# In[19]:


95/7        # StratifiedShuffleSplit  gives same values


# In[20]:


376/28      # StratifiedShuffleSplit  gives same values


# In[21]:


strat_train_set


# In[22]:


housing = strat_train_set.copy()


# # Looking for correlations

# In[23]:


corr_matrix = housing.corr()


# In[24]:


corr_matrix['medv'].sort_values(ascending = False)


# In[25]:


from pandas.plotting import scatter_matrix
attributes = ['medv','rm','zn','lstat']
scatter_matrix(housing[attributes],figsize=(12,8))


# In[26]:


housing.plot(kind='scatter',x='rm',y='medv',alpha=0.8)  
# aplha is used if anywhere in the graph the density is high it gets dark 


# # Attribute Combination

# In[27]:


housing['taxrm'] = housing['tax']/housing['rm']


# In[28]:


housing.head()


# In[29]:


corr_matrix = housing.corr()
corr_matrix['medv'].sort_values(ascending = False)


# In[30]:


housing.plot(kind='scatter',x='taxrm',y='medv',alpha=0.8)  


# In[31]:


housing = strat_train_set.drop('medv',axis=1)
housing_labels = strat_train_set['medv'].copy()


# # Missing Attributes

# # To take care of missing attributes , you have three options:
#     -1.Get rid of the missing data points
#     -2.Get rid of the whole attribute
#     -3.Set the value to some bvalue(0, mean or median)

# In[32]:


a = housing.dropna(subset=['rm'])   # Option 1
a.shape


# In[33]:


housing.drop('rm',axis = 1)  # Option 2
# Note that there is no rm column and also note that the original housing dataframe will remain unchanged


# In[34]:


median = housing['rm'].median()  # Compute Median for option 3


# In[35]:


housing['rm'].fillna(median)   # Option 3
# Note that the original housing dataframe will remain unchanged


# In[36]:


housing.shape


# In[37]:


housing.describe()


# In[38]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')
imputer.fit(housing)


# In[39]:


imputer.statistics_


# In[40]:


X = imputer.transform(housing)


# In[41]:


housing_tr = pd.DataFrame(X,columns=housing.columns)


# In[42]:


housing_tr.describe()


# # Scikit-learn Design

# Primarily, Three Types of objects
# 1. Estimators - It estimates some parameter based on a dataset. Eg. imputer
# It has a fit method and transform method.
# Fit method - Fits the dataset and calculates internal parameters
# 2. Transformers - Transform method takes input and returns output based on the
# learnings from fit(). It also has a convenience function called fit_transform()
# which fits and then transforms.
# 3. Predictors - linearRegression model is an example of predictor. fit() and
# predict() are two common function. It aslo gives score() function which will evaluate the predictions.

# # Feature Scaling

# Primarily, two types of feature scaling methods:
# 1. Min-Max scaling (Normalization)
#    (value-min)/(max-min)
#    Sklearn provides a class called MinMaxScaler for this
# 
# 2. Standardization
#    (value-mean)/std
#    Sklearn provides a class called StandardScaler for this
# 

# # Creating a Pipeline

# In[43]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    
    #  ..... add as many as you want in your pipeline
    
    ('std_scaler', StandardScaler()),
])


# In[44]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[45]:


housing_num_tr


# In[46]:


housing_num_tr.shape


# # Selecting a desired model for Dragon Real Estates

# In[47]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
#model = LinearRegression()
#model = DecisionTreeRegressor()
model.fit(housing_num_tr,housing_labels)


# In[48]:


some_data = housing.iloc[:5]


# In[49]:


some_labels = housing_labels.iloc[:5]


# In[50]:


prepared_data = my_pipeline.transform(some_data)


# In[51]:


model.predict(prepared_data)


# In[52]:


list(some_labels)


# # Evaluating the model

# In[53]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels,housing_predictions)
rmse = np.sqrt(mse)


# In[54]:


mse


# In[55]:


rmse


# # Using better evaluation technique - Cross Validation

# In[56]:


# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels,scoring='neg_mean_squared_error',cv=10)
rmse_scores = np.sqrt(-scores)


# In[57]:


rmse_scores


# In[58]:


def print_scores(scores):
    print('Scores:',scores)
    print('Mean: ',scores.mean())
    print('Standard deviation: ',scores.std())


# In[59]:


print_scores(rmse_scores)


# In[ ]:




