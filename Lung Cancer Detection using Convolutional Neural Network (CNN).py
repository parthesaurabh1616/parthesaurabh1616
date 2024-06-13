#!/usr/bin/env python
# coding: utf-8

# Computer Vision is one of the applications of deep neural networks that enables us to automate tasks that earlier required years of expertise and one such use in predicting the presence of cancerous cells.
# 
# In this article, we will learn how to build a classifier using a simple Convolution Neural Network which can classify normal lung tissues from cancerous. This project has been developed using collab and the dataset has been taken from Kaggle whose link has been provided as well.

# The process which will be followed to build this classifier:

# ![image.png](attachment:image.png)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

from sklearn.model_selection import train_test_split
from sklearn import metrics

import cv2
import gc
import os

import tensorflow as tf
from tensorflow import keras
from keras import layers

import warnings
warnings.filterwarnings('ignore')


# In[2]:


import zipfile

data_path = 'C://Users/saura/Downloads/lung_colon_image_set.zip'

try:
    with zipfile.ZipFile(data_path, 'r') as zip_ref:
        zip_ref.extractall()
    print('The data set has been extracted.')
except FileNotFoundError:
    print(f"Error: File '{data_path}' not found.")


# In[3]:


import zipfile

data_path = 'C://Users/saura/Downloads/lung_colon_image_set.zip'

with ZipFile(data_path,'r') as zip:
    zip.extractall()
    print('The data set has been extracted')


# # Data Visualization
# In this section, we will try to understand visualize some images which have been provided to us to build the classifier for each class.

# In[4]:


path = 'lung_image_sets'
classes = os.listdir(path)
classes


# In[5]:


path = '/lung_colon_image_set/lung_image_sets'


# In[6]:


path = 'lung_image_sets'

for cat in classes:
    image_dir = f'{path}/{cat}'
    images = os.listdir(image_dir)
    
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    fig.suptitle(f'Images for {cat} category . . . .',fontsize = 20)
    
    for i in range(3):
        k = np.random.randint(0,len(images))
        img = np.array(Image.open(f'{path}/{cat}/{images[k]}'))
        ax[i].imshow(img)
        ax[i].axis('off')
    plt.show()


# The above output may vary if you will run this in your notebook because the code has been implemented in such a way that it will show different images every time you rerun the code.

# # Data Preparation for Training
# In this section, we will convert the given images into NumPy arrays of their pixels after resizing them because training a Deep Neural Network on large-size images is highly inefficient in terms of computational cost and time.
# 
# For this purpose, we will use the OpenCV library and Numpy library of python to serve the purpose. Also, after all the images are converted into the desired format we will split them into training and validation data so, that we can evaluate the performance of our model.

# In[7]:


IMG_SIZE = 256
SPLIT = 0.2
EPOCHS = 10
BATCH_SIZE = 64


# Some of the hyperparameters which we can tweak from here for the whole notebook.

# In[8]:


X = []
Y = []

for i, cat in enumerate(classes):
    images = glob(f'{path}/{cat}/*.jpeg')
    
    for image in images:
        img = cv2.imread(image)
        
        X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        Y.append(i)
        
X = np.asarray(X)
one_hot_encoded_Y = pd.get_dummies(Y).values


# One hot encoding will help us to train a model which can predict soft probabilities of an image being from each class with the highest probability for the class to which it really belongs.

# In[9]:


X_train,X_val,Y_train,Y_val = train_test_split(X,one_hot_encoded_Y,
                                              test_size = SPLIT,
                                              random_state = 2022)
print(X_train.shape, X_val.shape)


# In this step, we will achieve the shuffling of the data automatically because the train_test_split function split the data randomly in the given ratio.

# # Model Development
# From this step onward we will use the TensorFlow library to build our CNN model. Keras framework of the tensor flow library contains all the functionalities that one may need to define the architecture of a Convolutional Neural Network and train it on the data.

# # Model Architecture
# We will implement a Sequential model which will contain the following parts:
# 
# - Three Convolutional Layers followed by MaxPooling Layers.
# - The Flatten layer to flatten the output of the convolutional layer.
# - Then we will have two fully connected layers followed by the output of the flattened layer.
# - We have included some BatchNormalization layers to enable stable and fast training and a Dropout layer before the final layer to avoid any possibility of overfitting.
# - The final layer is the output layer which outputs soft probabilities for the three classes. 

# In[10]:


model = keras.models.Sequential([
    layers.Conv2D(filters = 32,
                  kernel_size = (5,5),
                  activation = 'relu',
                  input_shape = (IMG_SIZE,
                                IMG_SIZE,
                                3),
                  padding = 'same'),           
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(filters = 64, 
                  kernel_size = (3,3),
                  activation = 'relu',
                  padding = 'same'),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(filters = 128,
                  kernel_size = (3,3),
                  activation = 'relu',
                  padding = 'same'),
    layers.MaxPooling2D(2,2),
    
    layers.Flatten(),
    layers.Dense(256, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dense(128, activation = 'relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(3, activation = 'softmax')
])


# Let’s print the summary of the model’s architecture:

# In[11]:


model.summary()


# From above we can see the change in the shape of the input image after passing through different layers. The CNN model we have developed contains about 33.5 Million parameters. This huge number of parameters and complexity of the model is what helps to achieve a high-performance model which is being used in real-life applications.

# In[12]:


keras.utils.plot_model(
    model,
    show_shapes = True,
    show_dtype = True,
    show_layer_activations = True
)


# In[13]:


model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)


# In[14]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs = {}):
        if logs.get('val_accuracy') > 0.90:
            print( '\n Validation accuracy has reached upto \ 90% so,stopping further training.')
            self.model.stop_training = True

es = EarlyStopping(patience = 3,
                   monitor = 'val_accuracy',
                   restore_best_weights = True)

lr = ReduceLROnPlateau(monitor = 'val_loss',
                       patience = 2,
                       factor = 0.5,
                       verbose = 1)


# Now we will train our model:

# In[16]:


history = model.fit(X_train,Y_train,
                    validation_data = (X_val,Y_val),
                    batch_size = BATCH_SIZE,
                    epochs = EPOCHS,
                    verbose = 1,
                    callbacks = [es,lr,myCallback()])


# Let’s visualize the training and validation accuracy with each epoch.

# In[17]:


history_df = pd.DataFrame(history.history)
history_df.loc[:,['loss','val_loss']].plot()
history_df.loc[:,['accuracy','val_accuracy']].plot()
plt.show()


#  
# 
# From the above graphs, we can certainly say that the model has not overfitted the training data as the difference between the training and validation accuracy is very low.

# # Model Evaluation
# Now as we have our model ready let’s evaluate its performance on the validation data using different metrics. For this purpose, we will first predict the class for the validation data using this model and then compare the output with the true labels.

# In[18]:


Y_pred = model.predict(X_val)
Y_val = np.argmax(Y_val, axis = 1)
Y_pred = np.argmax(Y_pred, axis = 1)


# Let’s draw the confusion metrics and classification report using the predicted labels and the true labels.

# In[19]:


metrics.confusion_matrix(Y_val,Y_pred)


# In[20]:


print(metrics.classification_report(Y_val,Y_pred,target_names = classes))


# # Conclusion:
# Indeed the performance of our simple CNN model is very good as the f1-score for each class is above 0.90 which means our model’s prediction is correct 90% of the time. This is what we have achieved with a simple CNN model what if we use the Transfer Learning Technique to leverage the pre-trained parameters which have been trained on millions of datasets and for weeks using multiple GPUs? It is highly likely to achieve even better performance on this dataset.

# In[ ]:




