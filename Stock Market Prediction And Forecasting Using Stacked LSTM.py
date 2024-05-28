#!/usr/bin/env python
# coding: utf-8

# # Things To Do
# 1. We will collect the Stock Data-- APPL
# 2. Preprocess the Data - Train and Test
# 3. create An Stacked LSTM Model
# 4. Predict the test data and plot the output
# 5. Predict the future 30 days and plot the output

# # Import lab's And Dataset

# In[7]:


import pandas as pd
import yfinance as yf

# Replace 'AAPL' with the desired ticker symbol
target_ticker = 'AAPL'
start_date = '2019-05-29'
end_date = '2024-05-27'

try:
  # Download data using yfinance
  df = yf.download(target_ticker, start=start_date, end=end_date)
  # Proceed with your data analysis using 'df'
except (yfinance.DownloadError, ValueError) as e:
    print(f"Error retrieving data: {e}")


# In[8]:


df.to_csv("APPL.csv")


# In[9]:


import pandas as pd


# In[10]:


df


# In[11]:


df = pd.read_csv("APPL.csv")


# In[12]:


df.head()


# In[13]:


df1 = df.reset_index()['Close']


# In[14]:


df1.shape


# In[15]:


df1


# In[16]:


import matplotlib.pyplot as plt
plt.plot(df1)


# # LSTM are sensitive to the scale of the data . so we apply MinMax Scaler

# In[17]:


import numpy as np


# In[18]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[19]:


df1.shape


# In[20]:


print(df1)


# # Train Test Splitting

# In[21]:


# Splitting dataset into train and test split
training_size = int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data = df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[22]:


training_size


# In[23]:


test_size


# In[24]:


import numpy
# Convert an array of values into a dataset matrix
def create_dataset(dataset,time_step=1):
    dataX,dataY = [],[]
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step),0]    ## i = 0, 0,1,2,3----99 , 100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX),numpy.array(dataY)


# # Reshape

# In[25]:


# Reshape into X = t,t+1,t+2,t+3 and Y = t+4
time_step = 100
X_train,y_train = create_dataset(train_data,time_step)
X_test,ytest = create_dataset(test_data,time_step)


# In[26]:


print(X_train)


# In[27]:


X_train.shape,y_train.shape


# In[28]:


X_test.shape,ytest.shape


# # Create An Stacked LSTM Model

# In[29]:


# reshape input to be [samples,time steps,features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)


# # Stacked LSTM

# In[30]:


# Create LSTM Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[31]:


model = Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences = True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[32]:


model.summary()


# In[33]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# # Prediction and Performance Metrics

# In[43]:


# Lets Do the prediction and check performance matrics
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# In[44]:


# Transform to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)


# In[45]:


# Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[46]:


# Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# # Plotting

# In[39]:


# Plotting
# Shift train predictions for plotting
look_back = 100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back,:] = train_predict

# Shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:,:] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1,:] = test_predict

# Plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[48]:


len(test_data)


# In[49]:


x_input = test_data[341:].reshape(1,-1)
x_input.shape


# In[50]:


temp_input = list(x_input)
temp_input = temp_input[0].tolist()


# In[52]:


print(temp_input)


# In[53]:


# Demonstrate prediction for next 10 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        
        # print(temp_input)
        x_input = np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1,n_steps,1))
        
        # print(x_input)
        yhat = model.predict(x_input,verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        
        #print(temp_ )
        lst_output.extend(yhat.tolist())
        i = i+1
    else:
        
        x_input = x_input.reshape((1,n_steps,1))
        yhat = model.predict(x_input,verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i = i+1
print(lst_output)


# In[56]:


day_new = np.arange(1,101)
day_pred = np.arange(101,131)


# In[57]:


import matplotlib.pyplot as plt


# In[58]:


len(df1)


# In[59]:


df3 = df1.tolist()
df3.extend(lst_output)


# In[60]:


plt.plot(day_new,scaler.inverse_transform(df1[1158:]))  # We have take here 1158 because 100 data is been taken previously
plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[63]:


df3 = df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1000:])


# In[64]:


df3 = df1.tolist()
df3.extend(lst_output)
plt.plot(df3[100:])


# In[66]:


df3 = df1.tolist()
df3.extend(lst_output)
plt.plot(df3[500:])


# In[ ]:




