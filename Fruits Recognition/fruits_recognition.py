
# coding: utf-8

# In[1]:

import sys
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.models import Sequential, Model
img_size=120
vgg16_model=VGG16(weights='imagenet',include_top=False,input_shape=(img_size,img_size,3))


# In[2]:


model=Sequential()
for layer in vgg16_model.layers:
    model.add(layer)


# In[3]:


for layer in model.layers:
    layer.trainable=False


# In[4]:


from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8,activation='softmax'))


# In[5]:


model.load_weights('my_weights.h5',by_name=True)


# In[6]:


from tqdm import tqdm
import os
fruits_label=['Apple Braeburn','Avocado','Banana','Dates','Guava','Orange','Pomogranate','Strawberry']
img_path=sys.argv[1]


# In[135]:


from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.preprocessing import image
import numpy as np
img_size=120
img=load_img(img_path,target_size=(img_size,img_size))
img1=image.img_to_array(img)
img2=np.expand_dims(img1,axis=0)


# In[136]:


img3=preprocess_input(img2)
res=model.predict(img3)


# In[137]:


strng=''
strng=fruits_label[np.argmax(res)]
import matplotlib.pyplot as plt
plt.imshow(img)
plt.title(strng)
plt.show()


