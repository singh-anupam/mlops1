#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Convolution2D


# In[2]:


from keras.layers import MaxPooling2D


# In[3]:


from keras.layers import Flatten


# In[4]:


from keras.layers import Dense


# In[5]:


from keras.models import Sequential


# In[6]:


model = Sequential()


# In[7]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))


# In[8]:





# In[9]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[10]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))


# In[11]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[12]:


model.add(Convolution2D(filters=16,
                       kernel_size=(3,3),
                       activation='relu'))


# In[13]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[ ]:





# In[14]:


model.add(Convolution2D(filters=8,
                       kernel_size=(3,3),
                       activation='relu'))


# In[15]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[16]:





# In[17]:


model.add(Flatten())


# In[18]:





# In[19]:


model.add(Dense(units=128, activation='relu'))


# In[20]:


model.add(Dense(units=64,activation='relu'))


# In[21]:





# In[22]:


model.add(Dense(units=1, activation='sigmoid'))


# In[23]:


model.summary()


# In[24]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[25]:


from keras_preprocessing.image import ImageDataGenerator


# In[ ]:





# In[27]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'train_1/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'test_1/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
history = model.fit(
        training_set,
        steps_per_epoch=2541,
        epochs=3,
        validation_data=test_set,
        validation_steps=800)


# In[28]:


print ("Accuracy of the trained model is : {} %".format ( 100 * history.history['val_accuracy'][-1])) 


# In[ ]:





# In[ ]:





# In[ ]:




