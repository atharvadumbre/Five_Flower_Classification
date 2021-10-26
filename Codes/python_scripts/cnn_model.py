#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


# In[2]:


from pathlib import Path
import glob
import os
import os.path


# In[3]:


path = Path('flowers')

images = list(path.glob(r'*/*.jpg'))

labels = list(map(lambda x:os.path.split(os.path.split(x)[0])[1],images))


# In[4]:


images_series = pd.Series(images,name='image_path').astype(str)
labels_series = pd.Series(labels,name='labels')


# In[5]:


data = pd.concat([images_series,labels_series],axis=1)


# In[6]:


data = data.sample(frac=1).reset_index(drop=True)


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train , X_test = train_test_split(data,test_size=0.2,shuffle=True,stratify=data.labels,random_state=17)
print(X_train.shape,X_test.shape)


# In[9]:


import seaborn as sns


# In[11]:


sns.countplot(data=X_train,x=X_train['labels'])


# In[12]:


sns.countplot(data=X_test,x=X_test['labels'])


# In[10]:


train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale = 1.0/255,
        shear_range = 0.3,
        zoom_range = 0.3,
        horizontal_flip = True,
        vertical_flip = True,
        rotation_range = 45,
        brightness_range = [0.5,1],
        width_shift_range=0.1,
        height_shift_range=0.1
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale = 1.0/255,
        validation_split = 0.4
)


# In[11]:


train_data = train_datagen.flow_from_dataframe(
        dataframe = X_train,
        x_col = 'image_path',
        y_col = 'labels',
        batch_size = 64,
        target_size = (150,150),
        class_mode = 'categorical',
        color_mode = 'rgb',
        subset='training'
)

test_data = test_datagen.flow_from_dataframe(
    dataframe = X_test,
    x_col = 'image_path',
    y_col = 'labels',
    batch_size = 64,
    target_size = (150,150),
    class_mode = 'categorical',
    color_mode = 'rgb',
    shuffle=False
)

val_data = test_datagen.flow_from_dataframe(
    dataframe = X_test,
    x_col = 'image_path',
    y_col = 'labels',
    batch_size = 64,
    target_size = (150,150),
    class_mode = 'categorical',
    color_mode = 'rgb',
    subset = 'validation'
)


# In[12]:


opti = keras.optimizers.Adam(learning_rate=0.001)
def my_learning_rate(epoch, lrate):
    if epoch%7 == 0 and epoch!=0:
        lrate = lrate * 0.2
    return lrate
 
lrs = keras.callbacks.LearningRateScheduler(my_learning_rate)


# In[13]:


model = keras.Sequential([
    keras.layers.Conv2D(16,(3,3),activation='relu',padding='same',input_shape=(150,150,3)),
    keras.layers.Conv2D(16,(3,3),activation='relu',padding='same'),
    keras.layers.MaxPooling2D(2,2),
    
    keras.layers.Conv2D(32,(3,3),activation='relu',padding='same'),
    keras.layers.Conv2D(32,(3,3),activation='relu',padding='same'),
    keras.layers.MaxPooling2D(2,2),
    
    keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'),
    keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'),
    keras.layers.MaxPooling2D(2,2),
    
    keras.layers.Conv2D(128,(3,3),activation='relu',padding='same'),
    keras.layers.Conv2D(128,(3,3),activation='relu',padding='same'),
    keras.layers.MaxPooling2D(2,2),
    
    
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation = 'relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(320,activation = 'relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(5,activation = 'softmax')
])

model.compile(optimizer = opti,loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()


# In[14]:


cnn_model = model.fit(train_data,epochs=50,steps_per_epoch=54,validation_data=val_data,validation_steps=6,callbacks=[lrs])


# In[16]:


model.save('scratch_cnn')


# In[ ]:





# In[18]:


epochs = range(1, 51)
plt.figure(figsize=(10,6))
plt.plot(epochs, cnn_model.history['loss'], label='Train Loss')
plt.plot(epochs, cnn_model.history['val_loss'], label='Val Loss')
plt.plot(epochs, cnn_model.history['accuracy'], label='Train Acc')
plt.plot(epochs, cnn_model.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epochs')
plt.title('CNN-5-Layer-Model')
plt.grid()
plt.legend()

plt.savefig('cnn_5_layer.jpg',dpi=500)
plt.show()


# In[ ]:




