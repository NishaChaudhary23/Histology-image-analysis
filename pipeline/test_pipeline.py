import pandas as pd
import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import os
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import RMSprop

model_2a = load_model('/home/chs.rintu/Documents/office/researchxoscc/project_2/output/M2a/M2a_finetune.h5')
model_2b = load_model('/home/chs.rintu/Documents/office/researchxoscc/project_2/output/M2b/M2b_finetune.h5')

# model summary
print("-----------------Model 2a-----------------")
model_2a.summary()
print("------------------------------------------")
print("-----------------Model 2b-----------------")
model_2b.summary()
print("------------------------------------------")

base = '/home/chs.rintu/Documents/office/researchxoscc/project_2/dataSet'
out_path = '/home/chs.rintu/Documents/office/researchxoscc/project_2/output'
datapath = f'{base}/train_all'


df_test = pd.read_csv(f'{base}/pipeline/all/test.csv')
label_2a = ['wpdoscc','mdoscc']
label_2b = ['pdoscc','wdoscc']
datagen_test = ImageDataGenerator(rescale = 1.0/255.0)
test_generator = datagen_test.flow_from_dataframe(
        dataframe=df_test,
        folder=datapath,
        target_size=(300, 300),
        class_mode='categorical',
        shuffle=False,
        validate_filenames=False)

predictions = model_2a.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
print(predictions)
print(y_pred)