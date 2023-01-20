#!/usr/bin/env python
# coding: utf-8

#Importing necessary packages
import pandas as pd
import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D,Dense,Flatten,GlobalAveragePooling2D,MaxPooling2D
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.models import load_model
import os
import cv2
import keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications import MobileNetV3Small
# from tensorflow.keras.applications import ConvNeXtBase
# from tensorflow.keras.applications import ConvNeXtLarge
# from tensorflow.keras.applications import ConvNeXtSmall
# from tensorflow.keras.applications import ConvNeXtTiny
# from tensorflow.keras.applications import ConvNeXtXLarge
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.applications import EfficientNetB6
from tensorflow.keras.applications import EfficientNetB7
# from tensorflow.keras.applications import efficientnet_v2
# from tensorflow.keras.applications import EfficientNetV2B0
# from tensorflow.keras.applications import EfficientNetV2B1
# from tensorflow.keras.applications import EfficientNetV2B2
# from tensorflow.keras.applications import EfficientNetV2B3
# from tensorflow.keras.applications import EfficientNetV2L
# from tensorflow.keras.applications import EfficientNetV2M
# from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import kl_divergence
from tensorflow.keras.metrics import mean_squared_error
from tensorflow.keras.metrics import poisson

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

#checking tensorflow version
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

train_large = '/home/chs.rintu/Documents/chs-lab-ws02/nisha/project-2-oscc/data/original'


#reading & displaying an image
a = np.random.choice(['wdoscc','mdoscc','pdoscc'])
path = '/home/chs.rintu/Documents/chs-lab-ws02/nisha/project-2-oscc/data/original/{}/'.format(a)


#(trainX, testX, trainY, testY) = train_test_split(data, train_large.target, test_size=0.25)

# models = ['DenseNet121','DenseNet169','DenseNet201','MobileNetV3Large','MobileNetV3Small','EfficientNetB0','EfficientNetB1','EfficientNetB2','EfficientNetB3','EfficientNetB4','EfficientNetB5','EfficientNetB6','EfficientNetB7','InceptionResNetV2''InceptionV3','MobileNetV2','NASNetLarge','NASNetMobile','ResNet101','ResNet101V2','ResNet152','ResNet152V2','ResNet50','ResNet50V2','VGG16','VGG19','Xception']
# ImageDataGenerator
# color images

model_type = 'DenseNet121'

datagen_train = ImageDataGenerator(rescale = 1.0/255.0,validation_split=0.2)
# Training Data
train_generator = datagen_train.flow_from_directory(
        train_large,
        target_size=(300, 300),
        batch_size=32,
        class_mode='categorical',
        subset = 'training')
#Validation Data
valid_generator = datagen_train.flow_from_directory(
        train_large,
        target_size=(300, 300),
        batch_size=32,
        class_mode='categorical',
        subset = 'validation',
        shuffle=False)


# Creating the model
if model_type == 'DenseNet121':
        densenet = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(300,300,3)
        )
        for layer in densenet.layers:
                layer.trainable = True
        x = layers.Flatten()(densenet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(densenet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'DenseNet169':
        densenet = DenseNet169(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in densenet.layers:
                layer.trainable = True
        x = layers.Flatten()(densenet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(densenet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'DenseNet201':
        densenet = DenseNet201(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in densenet.layers:
                layer.trainable = True
        x = layers.Flatten()(densenet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(densenet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

# if model_type == 'ConvNeXtBase':
#         convnext = ConvNeXtBase(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=(300,300,3)
#                 )
#         for layer in convnext.layers:
#                 layer.trainable = True
#         x = layers.Flatten()(convnext.output)
#         x = layers.Dense(1024, activation = 'relu')(x)
#         x = layers.Dropout(0.2)(x)
#         x = layers.Dense(3, activation = 'softmax')(x)
#         model = Model(convnext.input, x)
#         model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

# if model_type == 'ConvNeXtSmall':
#         convnext = ConvNeXtSmall(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=(300,300,3)
#                 )
#         for layer in convnext.layers:
#                 layer.trainable = True
#         x = layers.Flatten()(convnext.output)
#         x = layers.Dense(1024, activation = 'relu')(x)
#         x = layers.Dropout(0.2)(x)
#         x = layers.Dense(3, activation = 'softmax')(x)
#         model = Model(convnext.input, x)
#         model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

# if model_type == 'ConvNeXtLarge':
#         convnext = ConvNeXtLarge(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=(300,300,3)
#                 )
#         for layer in convnext.layers:
#                 layer.trainable = True
#         x = layers.Flatten()(convnext.output)
#         x = layers.Dense(1024, activation = 'relu')(x)
#         x = layers.Dropout(0.2)(x)
#         x = layers.Dense(3, activation = 'softmax')(x)
#         model = Model(convnext.input, x)
#         model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

# if model_type == 'ConvNeXtXLarge':
#         convnext = ConvNeXtXLarge(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=(300,300,3)
#                 )
#         for layer in convnext.layers:
#                 layer.trainable = True
#         x = layers.Flatten()(convnext.output)
#         x = layers.Dense(1024, activation = 'relu')(x)
#         x = layers.Dropout(0.2)(x)
#         x = layers.Dense(3, activation = 'softmax')(x)
#         model = Model(convnext.input, x)
#         model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

# if model_type == 'ConvNeXtTiny':
#         convnext = ConvNeXtTiny(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=(300,300,3)
#                 )
#         for layer in convnext.layers:
#                 layer.trainable = True
#         x = layers.Flatten()(convnext.output)
#         x = layers.Dense(1024, activation = 'relu')(x)
#         x = layers.Dropout(0.2)(x)
#         x = layers.Dense(3, activation = 'softmax')(x)
#         model = Model(convnext.input, x)
#         model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'EfficientNetB0':
        effnet = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in effnet.layers:
                layer.trainable = True
        x = layers.Flatten()(effnet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(effnet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'EfficientNetB1':
        effnet = EfficientNetB1(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in effnet.layers:
                layer.trainable = True
        x = layers.Flatten()(effnet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(effnet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'EfficientNetB2':
        effnet = EfficientNetB2(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in effnet.layers:
                layer.trainable = True
        x = layers.Flatten()(effnet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(effnet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'EfficientNetB3':
        effnet = EfficientNetB3(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in effnet.layers:
                layer.trainable = True
        x = layers.Flatten()(effnet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(effnet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'EfficientNetB4':
        effnet = EfficientNetB4(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in effnet.layers:
                layer.trainable = True
        x = layers.Flatten()(effnet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(effnet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'EfficientNetB5':
        effnet = EfficientNetB5(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in effnet.layers:
                layer.trainable = True
        x = layers.Flatten()(effnet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(effnet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'EfficientNetB6':
        effnet = EfficientNetB6(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in effnet.layers:
                layer.trainable = True
        x = layers.Flatten()(effnet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(effnet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'EfficientNetB7':
        effnet = EfficientNetB7(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in effnet.layers:
                layer.trainable = True
        x = layers.Flatten()(effnet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(effnet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

# if model_type == 'EfficientNetV2B0':
#         effnet = EfficientNetV2B0(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=(300,300,3)
#                 )
#         for layer in effnet.layers:
#                 layer.trainable = True
#         x = layers.Flatten()(effnet.output)
#         x = layers.Dense(1024, activation = 'relu')(x)
#         x = layers.Dropout(0.2)(x)
#         x = layers.Dense(3, activation = 'softmax')(x)
#         model = Model(effnet.input, x)
#         model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

# if model_type == 'EfficientNetV2B1':
#         effnet = EfficientNetV2B1(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=(300,300,3)
#                 )
#         for layer in effnet.layers:
#                 layer.trainable = True
#         x = layers.Flatten()(effnet.output)
#         x = layers.Dense(1024, activation = 'relu')(x)
#         x = layers.Dropout(0.2)(x)
#         x = layers.Dense(3, activation = 'softmax')(x)
#         model = Model(effnet.input, x)
#         model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

# if model_type == 'EfficientNetV2B2':
#         effnet = EfficientNetV2B2(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=(300,300,3)
#                 )
#         for layer in effnet.layers:
#                 layer.trainable = True
#         x = layers.Flatten()(effnet.output)
#         x = layers.Dense(1024, activation = 'relu')(x)
#         x = layers.Dropout(0.2)(x)
#         x = layers.Dense(3, activation = 'softmax')(x)
#         model = Model(effnet.input, x)
#         model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

# if model_type == 'EfficientNetV2B3':
#         effnet = EfficientNetV2B3(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=(300,300,3)
#                 )
#         for layer in effnet.layers:
#                 layer.trainable = True
#         x = layers.Flatten()(effnet.output)
#         x = layers.Dense(1024, activation = 'relu')(x)
#         x = layers.Dropout(0.2)(x)
#         x = layers.Dense(3, activation = 'softmax')(x)
#         model = Model(effnet.input, x)
#         model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

# if model_type == 'EfficientNetV2L':
#         effnet = EfficientNetV2L(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=(300,300,3)
#                 )
#         for layer in effnet.layers:
#                 layer.trainable = True
#         x = layers.Flatten()(effnet.output)
#         x = layers.Dense(1024, activation = 'relu')(x)
#         x = layers.Dropout(0.2)(x)
#         x = layers.Dense(3, activation = 'softmax')(x)
#         model = Model(effnet.input, x)
#         model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

# if model_type == 'EfficientNetV2M':
#         effnet = EfficientNetV2M(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=(300,300,3)
#                 )
#         for layer in effnet.layers:
#                 layer.trainable = True
#         x = layers.Flatten()(effnet.output)
#         x = layers.Dense(1024, activation = 'relu')(x)
#         x = layers.Dropout(0.2)(x)
#         x = layers.Dense(3, activation = 'softmax')(x)
#         model = Model(effnet.input, x)
#         model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

# if model_type == 'EfficientNetV2S':
#         effnet = EfficientNetV2S(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=(300,300,3)
#                 )
#         for layer in effnet.layers:
#                 layer.trainable = True
#         x = layers.Flatten()(effnet.output)
#         x = layers.Dense(1024, activation = 'relu')(x)
#         x = layers.Dropout(0.2)(x)
#         x = layers.Dense(3, activation = 'softmax')(x)
#         model = Model(effnet.input, x)
#         model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'InceptionResNetV2':
        inception = InceptionResNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in inception.layers:
                layer.trainable = True
        x = layers.Flatten()(inception.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(inception.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'InceptionV3':
        inception = InceptionV3(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in inception.layers:
                layer.trainable = True
        x = layers.Flatten()(inception.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(inception.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'MobileNetV2':
        mobilenet = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in mobilenet.layers:
                layer.trainable = True
        x = layers.Flatten()(mobilenet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(mobilenet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'MobileNetV3Large':
        mobilenet = MobileNetV3Large(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in mobilenet.layers:
                layer.trainable = True
        x = layers.Flatten()(mobilenet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(mobilenet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'MobileNetV3Small':
        mobilenet = MobileNetV3Small(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in mobilenet.layers:
                layer.trainable = True
        x = layers.Flatten()(mobilenet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(mobilenet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'NASNetLarge':
        nasnet = NASNetLarge(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in nasnet.layers:
                layer.trainable = True
        x = layers.Flatten()(nasnet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(nasnet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])        

if model_type == 'NASNetMobile':
        nasnet = NASNetMobile(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in nasnet.layers:
                layer.trainable = True
        x = layers.Flatten()(nasnet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(nasnet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'ResNet50V2':
        resnet = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in resnet.layers:
                layer.trainable = True
        x = layers.Flatten()(resnet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(resnet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'ResNet101V2':
        resnet = ResNet101V2(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in resnet.layers:
                layer.trainable = True
        x = layers.Flatten()(resnet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(resnet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'ResNet152V2':
        resnet = ResNet152V2(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in resnet.layers:
                layer.trainable = True
        x = layers.Flatten()(resnet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(resnet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'ResNet50':
        resnet = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in resnet.layers:
                layer.trainable = True
        x = layers.Flatten()(resnet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(resnet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'ResNet101':
        resnet = ResNet101(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in resnet.layers:
                layer.trainable = True
        x = layers.Flatten()(resnet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(resnet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'ResNet152':
        resnet = ResNet152(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in resnet.layers:
                layer.trainable = True
        x = layers.Flatten()(resnet.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(resnet.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'VGG16':
        vgg16 = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in vgg16.layers:
                layer.trainable = True
        x = layers.Flatten()(vgg16.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(vgg16.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'VGG19':
        vgg19 = VGG19(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in vgg19.layers:
                layer.trainable = True
        x = layers.Flatten()(vgg19.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(vgg19.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if model_type == 'Xception':
        xception = Xception(
                weights='imagenet',
                include_top=False,
                input_shape=(300,300,3)
                )
        for layer in xception.layers:
                layer.trainable = True
        x = layers.Flatten()(xception.output)
        x = layers.Dense(1024, activation = 'relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(3, activation = 'softmax')(x)
        model = Model(vgg19.input, x)
        model.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

if not os.path.exists(f'/home/chs.rintu/Documents/chs-lab-ws02/nisha/project-2-oscc/Histology-image-analysis/models/{model_type}'):
        os.makedirs(f'/home/chs.rintu/Documents/chs-lab-ws02/nisha/project-2-oscc/Histology-image-analysis/models/{model_type}')
# Model Summary


#TF_CPP_MIN_LOG_LEVEL=2
# Training the model

print("------------------------------------------")
print(f'Training the model {model_type}')
print("------------------------------------------")
history = model.fit(train_generator, validation_data = valid_generator, epochs=50)

print("------------------------------------------")
print(f'Training Complete')
print("------------------------------------------")
# Creating a directory to save the model paths 

# Saving the model
model.save(f'/home/chs.rintu/Documents/chs-lab-ws02/nisha/project-2-oscc/Histology-image-analysis/models/{model_type}/dense121_01.h5')
print("------------------------------------------")
print(f'Model saved')
print("------------------------------------------")


#plotting the accuracy and loss
print("------------------------------------------")
print(f'Plotting and supplimentary data')
print("------------------------------------------")
plt.figure(figsize=(10, 10))
plt.lineplot(history.history['acc'], label='Training Accuracy')
plt.lineplot(history.history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.savefig(f'/home/chs.rintu/Documents/chs-lab-ws02/nisha/project-2-oscc/Histology-image-analysis/models/{model_type}/Accuracy.jpg')

np.save('/home/chs.rintu/Documents/chs-lab-ws02/nisha/project-2-oscc/Histology-image-analysis/models/{model_type}/history1.npy',history.history)

loaded_model = load_model(f'/home/chs.rintu/Documents/chs-lab-ws02/nisha/project-2-oscc/Histology-image-analysis/models/{model_type}/dense121_01.h5')
outcomes = loaded_model.predict(valid_generator)
y_pred = np.argmax(outcomes, axis=1)
# confusion matrix
confusion = confusion_matrix(valid_generator.classes, y_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(f'/home/chs.rintu/Documents/chs-lab-ws02/nisha/project-2-oscc/Histology-image-analysis/models/{model_type}/Confusion_matrix.jpg')

conf_df = pd.DataFrame(confusion, index = ['wdoscc','mdoscc','pdoscc'], columns = ['wdoscc','mdoscc','pdoscc'])
conf_df.to_csv(f'/home/chs.rintu/Documents/chs-lab-ws02/nisha/project-2-oscc/Histology-image-analysis/models/{model_type}/Confusion_matrix.csv')

# classification report
target_names = ['wdoscc','mdoscc','pdoscc']
report = classification_report(valid_generator.classes, y_pred, target_names=target_names, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv(f'/home/chs.rintu/Documents/chs-lab-ws02/nisha/project-2-oscc/Histology-image-analysis/models/{model_type}/Classification_report.csv')

print("------------------------------------------")
print(f'Supplimentary Data Saved')
print("------------------------------------------")