#!/usr/bin/env python
# coding: utf-8

#Importing necessary packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D,Dense,Flatten,GlobalAveragePooling2D,MaxPooling2D
from tensorflow.keras.models import Sequential,Model
import os
import cv2
import keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.optimizers import RMSprop

#checking tensorflow version
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

train_large = '/home/chs.rintu/Documents/chs-lab-ws02/nisha/project-2-oscc/data/original/'


#reading & displaying an image
a = np.random.choice(['wdoscc','mdoscc','pdoscc'])
path = '/home/chs.rintu/Documents/chs-lab-ws02/nisha/project-2-oscc/data/original/{}/'.format(a)
ip = np.random.choice(os.listdir(path))

image = cv2.imread(path+ip,0)
plt.imshow(image)
plt.title(a)
plt.show()


#(trainX, testX, trainY, testY) = train_test_split(data, train_large.target, test_size=0.25)


# ImageDataGenerator
# color images

datagen_train = ImageDataGenerator(rescale = 1.0/255.0,validation_split=0.2)
# Training Data
train_generator = datagen_train.flow_from_directory(
        train_large,
        target_size=(300, 300),
        batch_size=100,
        class_mode='categorical',
        subset = 'training')
#Validation Data
valid_generator = datagen_train.flow_from_directory(
        train_large,
        target_size=(300, 300),
        batch_size=100,
        class_mode='categorical',
        subset = 'validation',
        shuffle=False)



#creating and training the model

densenet = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(300,300,3)
)


for layer in densenet.layers:
  layer.trainable = False


x = layers.Flatten()(densenet.output)
x = layers.Dense(1024, activation = 'relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(3, activation = 'softmax')(x)
model = Model(densenet.input, x)

model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

model.summary()
#TF_CPP_MIN_LOG_LEVEL=2
history = model.fit(train_generator, validation_data = valid_generator, epochs=50)