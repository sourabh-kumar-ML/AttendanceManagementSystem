#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np
from PIL import Image
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from datetime import datetime
from keras.models import Sequential,load_model
from keras.layers import Flatten,BatchNormalization,Dense,Dropout,Input
from sklearn.preprocessing import OneHotEncoder


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


class RecogoniseFace():
    def __init__(self,threshold = 0.60):
        self.model = load_model("face_Classification_model.h5")
        self.encoder = pickle.load(open("encoder.data", 'rb'))
        self.model_facenet = load_model("facenet_keras.h5")
        self.threshold = threshold
        
    def recogonise(self,image_raw,mark=False):
        self.get_faces(image_raw)
        self.prediction()
        if mark:
            self.manage_book()
    # function to extract faces 
    def get_faces(self,image,target_shape=(160,160)):
        self.face_list = []
        self.face_cordinates = []
        self.image = Image.fromarray(image)
        # Covert it to RGB
        self.image = self.image.convert("RGB")
        # convert image to array
        self.image = np.asarray(self.image)
        self.face_detector = MTCNN()
        # detect faces
        self.results = self.face_detector.detect_faces(self.image)
        # Go through every detecte face and save it
        for i in range(len(self.results)):
            x1, y1, width, height = self.results[i]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            self.face_cordinates.append([x1,x2,y1,y2])
            face = self.image[y1:y2, x1:x2]
            # resize pixels to the model size
            img = Image.fromarray(face)
            img = img.resize(target_shape)
            face_array = np.asarray(img)
            self.face_list.append(face_array)
    
    
    # get the face embedding for one face
    def get_embedding(self,face_pixels):
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = np.expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = self.model_facenet.predict(samples)
        return yhat[0]

    def prediction(self):
        self.name = []
        for face in self.face_list:
            val = self.get_embedding(face)
            val = val.reshape([1,128,])
            pred = self.model.predict(val)[0]
            idx = np.argmax(pred)
            
            if pred[idx] > self.threshold:
                num = len(self.encoder.categories_[0])
                array = np.zeros((1,num))
                array[0][idx] = 1
                self.name.append(self.encoder.inverse_transform(array)[0][0])
            else :
                self.name.append("NewStudent")
                print("New Candidate Detected")
            
    def manage_book(self):
        file = pd.read_excel("attendance.xlsx")
        for ids in self.name:
            if ids not in file.Name.values:
                values = [np.NaN for _ in range (len(file.columns)-1)]
                values.insert(0,ids)
                file = file.append(pd.Series(values, index = file.columns), ignore_index=True)
        
            idx = file.loc[file.Name == ids].index
            date = datetime.today().strftime('%Y-%m-%d')
            time = datetime.now().strftime('%H:%M')
            
            if date not in file.columns:
                file[date] = ["A" for _ in range(len(file))]
                file[date][idx] = "P" + "   " + time
            else :
                file[date][idx] = "P" + "   " + time
        
        file.to_excel("attendance.xlsx",index=None)


# In[ ]:




