#!/usr/bin/env python
# coding: utf-8

# Normally, a lot of businesses are remained as failures due to lack of profit, lack of proper improvement measures. Mostly, restaurant owners face a lot of difficulties to improve their productivity. This project really helps those who want to increase their productivity, which in turn increases their business profits. This is the main objective of this project.
# 
# What the project does is that the restaurant owner gets to know about drawbacks of his restaurant such as most disliked food items of his restaurant by customer’s text review which is processed with ML classification algorithm(Naive Bayes) and its results gets stored in the database using SQLite. 

# In[1]:


# 1. General Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. NLP Libraries
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# 3. Bag of words Model Count
from sklearn.feature_extraction.text import CountVectorizer

# 4. Train Test Split
from sklearn.model_selection import train_test_split

# 5. Classification Model - Naive Bayes
from sklearn.naive_bayes import GaussianNB

# 6. Classification Model - Score
from sklearn.metrics import confusion_matrix , accuracy_score


# In[2]:


df = pd.read_csv('C://Users/saura/Downloads/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


# In[3]:


df.head(2)


# In[4]:


len(df)


# In[5]:


df['Review'][0]


# In[16]:


review = re.sub('[^a-zA-Z]', ' ', df['Review'][0])
review = review.lower()
review


# In[17]:


review = review.split()
review


# In[18]:


stopwords.words('english')[:10]


# In[19]:


porter_stemmer = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')


# In[20]:


review = [porter_stemmer.stem(word) for word in review if not word in set(all_stopwords)]
review


# In[21]:


review = ' '.join(review)
review


# # Building Corpus

# In[23]:


corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ',df['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)


# In[24]:


print(corpus)


# In[25]:


cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, -1].values


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 0)


# In[27]:


classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[28]:


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[29]:


cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test,y_pred)


# In[30]:


print(X_test)


# In[31]:


sample_text = 'I love this restaurant so much'


# In[33]:


new_review = sample_text
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)


# In[34]:


sample_text = 'I hate this restaurant so much'


# In[35]:


new_review = sample_text
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)


# In[ ]:




