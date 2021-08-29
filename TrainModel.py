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
from keras.models import Sequential,load_model
from keras.layers import Flatten,BatchNormalization,Dense,Dropout,Input
from sklearn.preprocessing import OneHotEncoder


# In[2]:


import warnings
warnings.filterwarnings(action="ignore")


# In[6]:


class TrainModel():
    # function to extract faces 
    def get_faces(self,image_name,target_shape=(160,160)):
        face_list = []
        # open image
        image = Image.open(image_name)
        # Covert it to RGB
        image = image.convert("RGB")
        # convert image to array
        image = np.asarray(image)
        face_detector = MTCNN()
        # detect faces
        results = face_detector.detect_faces(image)
        # Go through every detecte face and save it
        for i in range(len(results)):
            x1, y1, width, height = results[i]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = image[y1:y2, x1:x2]
            # resize pixels to the model size
            img = Image.fromarray(face)
            img = img.resize(target_shape)
            face_array = np.asarray(img)
            face_list.append(face_array)
        return face_list
    
    def load_data(self,directory_name):
        X,y = [],[]
        for sub in os.listdir(directory_name):
            current_path = directory_name + "/" + sub + "/"
            for image_name in os.listdir(current_path):
                image_name = current_path + image_name

                faces = self.get_faces(image_name)
                for i in range(len(faces)):
                    X.append(faces[i])
                    y.append(sub)
        return np.asarray(X),np.asarray(y)
    
    # get the face embedding for one face
    def get_embedding(self,model, face_pixels):
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = np.expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        return yhat[0]
    
    def define_model(sele,num_labels):
        model = Sequential()
        model.add(Dense(256, activation='relu',input_dim=128))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(num_labels, activation='softmax'))
        
        return model

    def train(self,train_dir,val_dir):
        trainX, trainy = self.load_data(train_dir)
        testX, testy = self.load_data(val_dir)
        
        model_facenet = load_model("facenet_keras.h5")
        
        X_train = []
        for img in trainX:
            embedding = self.get_embedding(model_facenet, img)
            X_train.append(embedding)
        X_train = np.asarray(X_train)


        # convert each face in the test set to an embedding
        X_test = []
        for img in testX:
            embedding = self.get_embedding(model_facenet, img)
            X_test.append(embedding)
        X_test = np.asarray(X_test)
        
        oe_enc = OneHotEncoder()
        Y_train = oe_enc.fit_transform(trainy.reshape(-1,1)).toarray()
        Y_test = oe_enc.transform(testy.reshape(-1,1)).toarray()
        pickle.dump(oe_enc, open("encoder.data", "wb"))
        
        num_labels = len(os.listdir(train_dir))
        
        model = self.define_model(num_labels)
        model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
        model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=50)
        
        model.save("face_Classification_model.h5")


# In[ ]:




