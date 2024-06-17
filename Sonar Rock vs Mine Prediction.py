#!/usr/bin/env python
# coding: utf-8

# # Import All Lab's

# In[1]:


## Import All Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# # Data Collection as Data Processing

# In[4]:


## Loading the dataset to a pandas Dataframe
sonar_data = pd.read_csv("C://Users/saura/Downloads/copy of sonar data.csv",header = None)

## Here we have to mension that header = None because there is no name given to columns


# In[5]:


sonar_data


# In[6]:


sonar_data.head()


# In[7]:


## Number of rows and columns
sonar_data.shape


# In[8]:


sonar_data.info()


# In[9]:


sonar_data.describe() # Gives statisticals methods


# In[11]:


sonar_data[60].value_counts() # R -> Rock  , M -> Mine


# In[12]:


sonar_data.groupby(60).mean()


# In[13]:


# Separating data and labels
X = sonar_data.drop(columns = 60,axis=1)
Y = sonar_data[60]


# In[17]:


X


# In[18]:


Y


# # Training and Testing Data

# In[20]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.1,stratify = Y,random_state = 1)


# In[21]:


print(X.shape,X_train.shape,X_test.shape)


# In[23]:


print(X_train)
print(Y_train)


# # Model Training --> Logistic Regression Model

# In[22]:


model = LogisticRegression()


# In[24]:


# Training the Logistic Regression Model with training data
model.fit(X_train,Y_train)


# # Model Evaluation

# In[25]:


## accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)


# In[27]:


print("Accuracy on training data = ",training_data_accuracy)


# In[34]:


## accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print("Accuracy on test data = ", test_data_accuracy)


# # Making a Predictive System to predict Rock or Mine

# In[36]:


input_data = (0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032
)

# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]=='R'):
    print("The object is Rock")
else:
    print("The object is Mine")


# In[38]:


input_data = (0.0346,0.0509,0.0079,0.0243,0.0432,0.0735,0.0938,0.1134,0.1228,0.1508,0.1809,0.2390,0.2947,0.2866,0.4010,0.5325,0.5486,0.5823,0.6041,0.6749,0.7084,0.7890,0.9284,0.9781,0.9738,1.0000,0.9702,0.9956,0.8235,0.6020,0.5342,0.4867,0.3526,0.1566,0.0946,0.1613,0.2824,0.3390,0.3019,0.2945,0.2978,0.2676,0.2055,0.2069,0.1625,0.1216,0.1013,0.0744,0.0386,0.0050,0.0146,0.0040,0.0122,0.0107,0.0112,0.0102,0.0052,0.0024,0.0079,0.0031)
# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]=='R'):
    print("The object is Rock")
else:
    print("The object is Mine")


# In[ ]:




