#!/usr/bin/env python
# coding: utf-8

# # Import All Lab's

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from keras import layers

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("C://Users/saura/Downloads/auto-mpg.csv")


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# # EDA

# In[8]:


df['horsepower'].unique()


# In[9]:


df = df[df['horsepower'] != '?']


# In[10]:


df.shape


# In[11]:


df['horsepower'] = df['horsepower'].astype(int)
df.isnull().sum()


# In[12]:


df.nunique()


# In[13]:


plt.subplots(figsize=(15, 5))
for i, col in enumerate(['cylinders', 'origin']):
    
    plt.subplot(1, 2, i + 1)
  # Calculate average mpg for each category (assuming 'mpg' is the mpg column)
    avg_mpg = df.groupby(col)['mpg'].mean()
    avg_mpg.plot.bar()
    plt.xticks(rotation=0)
    plt.title(f"Average MPG by {col}")  # Add informative title

plt.tight_layout()
plt.show()


# In[14]:


import pandas as pd  # Assuming pandas is already imported

# Create a pivot table with mpg as values and 'cylinders' and 'origin' as columns
mpg_pivot_table = df.pivot_table(values='mpg', index='cylinders', columns='origin')

# Plot the pivot table as a heatmap (or other chart types)
mpg_pivot_table.plot(kind='bar')  # You can use 'heatmap' or other plot kinds
plt.show()


# In[17]:


plt.figure(figsize=(8, 8))
sns.heatmap(df[[col for col in ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin'] if col != 'car name']] .corr() > 0.9,
            annot=True,
            cbar=False)
plt.show()


# In[18]:


df.drop('displacement',
       axis = 1,
       inplace = True)


# # Data Input Pipeline

# In[19]:


from sklearn.model_selection import train_test_split
features = df.drop(['mpg','car name'],axis = 1)
target = df['mpg'].values

X_train,X_val,\
Y_train,Y_val = train_test_split(features,target,
                                               test_size = 0.2,
                                               random_state = 22)
X_train.shape,X_val.shape


# In[20]:


AUTO = tf.data.experimental.AUTOTUNE

train_ds = (
tf.data.Dataset
.from_tensor_slices((X_train,Y_train))
.batch(32)
.prefetch(AUTO)
)

val_ds = (
tf.data.Dataset
.from_tensor_slices((X_val,Y_val))
.batch(32)
.prefetch(AUTO)
)


# # Model Architecture

# In[21]:


model = keras.Sequential([
    layers.Dense(256,activation='relu',input_shape=[6]),
    layers.BatchNormalization(),
    layers.Dense(256,activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1,activation='relu')
])


# In[22]:


model.compile(
loss='mae',
optimizer='adam',
metrics=['mape']
)


# Let’s print the summary of the model’s architecture:

# In[23]:


model.summary()


# # Model Training

# In[24]:


history = model.fit(train_ds,
                   epochs=50,
                   validation_data = val_ds)


# In[25]:


history_df = pd.DataFrame(history.history)
history_df.head()


# In[26]:


history_df.loc[:,['loss','val_loss']].plot()
history_df.loc[:,['mape','val_mape']].plot()
plt.show()


#  
# 
# The training error has gone down smoothly but the case with the validation is somewhat different.

# In[ ]:




