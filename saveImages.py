#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import time
from PIL import Image


# In[2]:


class NewEntry():
    def __init__(self,train_dir,val_dir,name,images):
        self.train_dir,self.val_dir = train_dir,val_dir
        self.target_train_dir = self.train_dir+"/"+name
        self.target_val_dir = self.val_dir+"/"+name
        self.save_images(images)
    def save_images(self,images):
        
        
        #cam.release()
        
        if not os.path.isdir(self.target_train_dir) :
            os.mkdir(self.target_train_dir)
        if not os.path.isdir(self.target_val_dir) :
            os.mkdir(self.target_val_dir)
            
        train_images = images[0:30]
        val_images = images[30:]
        
        for i in range(30):
            img_name = self.target_train_dir +"/"+"{}.jpg".format(i)
            img = Image.fromarray(train_images[i])
            img.save(img_name)
        for i in range(20):
            img_name = self.target_val_dir +"/"+"{}.jpg".format(i)
            img = Image.fromarray(val_images[i])
            img.save(img_name)


# In[ ]:




