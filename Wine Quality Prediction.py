#!/usr/bin/env python
# coding: utf-8

# # Import all Lab's

# In[21]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')


# # Loading Dataset

# In[14]:


df = pd.read_csv("C://Users/saura/Downloads/wine.csv")


# In[15]:


df


# Let’s explore the type of data present in each of the columns present in the dataset.

# In[7]:


df.info()


# Now we’ll explore the descriptive statistical measures of the dataset.

# In[8]:


df.describe().T


# # Exploratory Data Analysis (EDA):-
#     - EDA is an approach to analysing the data using visual techniques. It is used to discover trends, and patterns, or to check assumptions with the help of statistical summaries and graphical representations.  Now let’s check the number of null values in the dataset columns wise.

# In[16]:


df.isnull().sum()


# In[17]:


for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())
df.isnull().sum().sum()


# # Plotting Map 

# In[18]:


df.hist(bins = 20, figsize=(10,10))
plt.show()


# Now let’s draw the count plot to visualise the number data for each quality of wine.

# In[19]:


plt.bar(df['quality'],df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()


# There are times the data provided to us contains redundant features they do not help with increasing the model’s performance that is why we remove them before using them to train our model.

# In[23]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder  # Assuming 'white' is a category

# Assuming 'df' is your DataFrame
# Identify and address the string value ('white') based on its meaning in your data
# (Code for handling 'white' as a categorical feature using label encoding)

# Create a LabelEncoder object
le = LabelEncoder()

# Assuming 'white' is in the first column
df.iloc[:, 0] = le.fit_transform(df.iloc[:, 0])

# Calculate correlation matrix (consider using df.corr(method='spearman') for non-normal data)
corr_matrix = df.corr()

# Create the heatmap (filtering correlations above 0.7)
plt.figure(figsize=(12, 12))
sns.heatmap(corr_matrix > 0.7, annot=True, cbar=False)
plt.show()


# From the above heat map we can conclude that the ‘total sulphur dioxide’ and ‘free sulphur dioxide‘ are highly correlated features so, we will remove them.

# In[24]:


df = df.drop('total sulfur dioxide',axis=1)


# # Model Development

# In[25]:


df['best quality'] = [1 if x > 5 else 0 for x in df.quality]


# We have a column with object data type as well let’s replace it with the 0 and 1 as there are only two categories.

# In[26]:


df.replace({'white':1,'red':0}, inplace = True)


# After segregating features and the target variable from the dataset we will split it into 80:20 ratio for model selection.

# In[27]:


features = df.drop(['quality','best quality'], axis = 1)
target = df['best quality']

xtrain,xtest,ytrain,ytest = train_test_split(features,target,test_size = 0.2,random_state = 40)
xtrain.shape, xtest.shape


# Normalising the data before training help us to achieve stable and fast training of the model.

# In[28]:


norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)


# As the data has been prepared completely let’s train some state of the art machine learning model on it.

# In[29]:


models = [LogisticRegression(),XGBClassifier(),SVC(kernel = 'rbf')]

for i in range(3):
    models[i].fit(xtrain,ytrain)
    
    print(f'{models[i]} : ')
    print('Training Accuracy : ',metrics.roc_auc_score(ytrain,models[i].predict(xtrain)))
    print('Validation Accuracy : ',metrics.roc_auc_score(
    ytest,models[i].predict(xtest)))
    print()


# # Model Evaluation :-
#     -From the above accuracies we can say that Logistic Regression and SVC() classifier performing better on the validation data with less difference between the validation and training data. Let’s plot the confusion matrix as well for the validation data using the Logistic Regression model.

# In[30]:


metrics.plot_confusion_matrix(models[1],xtest,ytest)
plt.show()


# Let’s also print the classification report for the best performing model.

# In[32]:


print(metrics.classification_report(ytest,
                                   models[1].predict(xtest)))


# In[ ]:




