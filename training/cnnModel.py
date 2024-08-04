# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 00:51:06 2021

@author: reham
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import Sequential
from tensorflow import keras
from sklearn.model_selection import KFold
############################# DATA LOADING #######################
print(os.listdir(r'C:\Users\USER\Downloads\ai project -20240803T181254Z-001\ai project\Project\Project\another_model\data'))
train_path = r'C:\Users\USER\Downloads\ai project -20240803T181254Z-001\ai project\Project\Project\another_model\data\train'
val_path = r'C:\Users\USER\Downloads\ai project -20240803T181254Z-001\ai project\Project\Project\another_model\data\test'
emotion_labels = sorted(os.listdir(train_path))
print(emotion_labels)
batch_size = 64
# image size 
target_size = (48,48)
# reduce the range of pixeles range to reduce total loss 
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen   = ImageDataGenerator(rescale=1./255)
#"categorical". Determines the type of label arrays that are returned
train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=target_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=True)

val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=target_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')
##################################MODEL ###############################################3
# first model will take the input data 
#MaxPooling2D(pool_size=(2, 2)) -> reduse number of pixels 
input_shape = (48,48,1) # img_rows, img_colums, color_channels
num_classes = 7
optimizer = keras.optimizers.RMSprop(learning_rate = 0.0001, decay = 1e-6)
cnn4 = Sequential()
# first lyer : termed as the Feature map which gives us information about the image such as the corners and edges
cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
cnn4.add(BatchNormalization())

cnn4.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(MaxPooling2D(pool_size=(2, 2)))
cnn4.add(Dropout(0.25))

cnn4.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(Dropout(0.25))

cnn4.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(MaxPooling2D(pool_size=(2, 2)))
cnn4.add(Dropout(0.25))
# mathematical functions operations usually take place. In this stage, the classification process begins to take place.
cnn4.add(Flatten())

cnn4.add(Dense(512, activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(Dropout(0.5))

cnn4.add(Dense(128, activation='relu'))
cnn4.add(BatchNormalization())
cnn4.add(Dropout(0.5))

cnn4.add(Dense(len(emotion_labels), activation='softmax'))
# 
cnn4.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["categorical_accuracy"])
#Softmax converts a vector of values to a probability distribution.
cnn4.add(layers.Activation('softmax'))

cnn4.summary()

cnn4.compile(loss = 'binary_crossentropy',optimizer = optimizer, metrics = ['accuracy',keras.metrics.Precision(), keras.metrics.Recall()])

num_epochs = 30

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VAL   = val_generator.n//val_generator.batch_size

history = cnn4.fit(train_generator, steps_per_epoch=STEP_SIZE_TRAIN, epochs=num_epochs, batch_size=batch_size, validation_data=val_generator, validation_steps=STEP_SIZE_VAL)
history
models.save_model(cnn4, 'CNN.h5')
