#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/saura/Downloads/electronics.csv")
df


# In[2]:


df.head()


# In[3]:


df.tail()


# In[4]:


df.info()


# In[5]:


from datetime import datetime
pd.to_datetime(df['timestamp'])


# In[6]:


df['brand'] = df['brand'].astype(str)


# In[7]:


df['category'] = df['category'].astype(str)


# In[8]:


df['rating'] = df['rating'].astype(float)


# In[9]:


df['user_id'] = df['user_id'].astype(str)


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


df.nunique()


# In[13]:


rating.dropna(inplace = True)
rating.drop_duplicates(inplace = True)


# In[14]:


df.duplicated().sum()


# In[15]:


df.isna().sum()


# # Finding the asnwer with data we have

# In[16]:


sns.countplot(x = 'rating',data=df)


# In[17]:


df['year'].max()


# In[18]:


df['year'] = pd.DatetimeIndex(df['timestamp']).year
df.groupby('year')['rating'].count().plot(kind='bar')


# In[19]:


df_2015 = df[df['year'] == 2015]
df_2015.groupby('brand')['rating'].count().sort_values(ascending = True).head(10).plot(kind='bar')


# In[20]:


df[df['year'] == 2016].groupby('brand')['rating'].count().sort_values(ascending = True).head(10).plot(kind='bar')


# In[21]:


df[df['year'] == 2018].groupby('brand')['rating'].count().sort_values(ascending = True).head(10).plot(kind='bar')


# In[22]:


df[df['year'] == 2015].groupby('year')['rating'].count().plot(kind = 'bar')


# In[23]:


df['month'] = pd.DatetimeIndex(df['timestamp']).month
df.groupby('month')['rating'].count().plot(kind='bar')


# In[24]:


df.groupby('brand')['rating'].count().sort_values(ascending = True).head(10).plot(kind='bar')


# In[25]:


df.groupby('category')['rating'].count().sort_values(ascending = True).head(10).plot(kind='bar')


# In[26]:


df.groupby('brand')['rating'].count().sort_values(ascending = True).head(10).plot(kind = 'bar')


# In[27]:


df.groupby('category')['rating'].count().sort_values(ascending = True).head(10).plot(kind='pie')


# In[28]:


df.groupby('brand')['rating'].count().sort_values(ascending = True).head(10).plot(kind='pie')


# In[30]:


df.groupby('brand')['rating'].count().sort_values(ascending = True).head(10).plot(kind='pie',autopct = '%1.2f%%')


# In[ ]:




