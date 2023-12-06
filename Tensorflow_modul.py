#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os


# In[2]:


os.environ['TF_CPP_MIN_LDG_LEVEL'] = '2'


# In[3]:


import tensorflow as tf


# In[4]:


from tensorflow import keras


# In[5]:


from tensorflow.keras import layers


# In[6]:


from tensorflow.keras.datasets import cifar10


# In[7]:


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)


# In[8]:


(x_train,y_train),(x_test,y_test) = cifar10.load_data()


# In[9]:


x_train = x_train.astype("float32")/255


# In[10]:


x_test = x_test.astype("float32")/255


# In[12]:


model = keras.Sequential([
    keras.Input(shape=(32,32,3)),
    layers.Conv2D(32,3,padding = 'valid', activation = 'relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(64,3, activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128,3, activation = 'relu'),
    layers.Flatten(),
    layers.Dense(64,activation = 'relu'),
    layers.Dense(10),
])


# In[14]:


model.compile(
           loss= keras.losses.SpareCategoricalCrossentropy(from_logits=True),
    
optimizer= keras.optimizers.Adam(lr=3e-4),
metrics = ['accuracy'],)


# In[16]:


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=['accuracy']
)


# 

# In[22]:


model.fit(x_train,y_train,batch_size = 64, epochs = 11, verbose = 2)


# In[23]:


model.evaluate(x_test,y_test, batch_size = 64, verbose = 2)


# In[ ]:




