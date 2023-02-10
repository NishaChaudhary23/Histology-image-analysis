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

model_2a = load_model('/home/chs.rintu/Documents/office/researchxoscc/project_2/output/M2a/M2a.h5')
model_2b = load_model('/home/chs.rintu/Documents/office/researchxoscc/project_2/output/M2b/M2b.h5')

# model summary
print("-----------------Model 2a-----------------")
# model_2a.summary()
print("------------------------------------------")
print("-----------------Model 2b-----------------")
# model_2b.summary()
print("------------------------------------------")

base = '/home/chs.rintu/Documents/office/researchxoscc/project_2/dataSet'
out_path = '/home/chs.rintu/Documents/office/researchxoscc/project_2/output'
datapath = f'{base}/train_all'


df_test = pd.read_csv(f'{base}/pipeline/all/master_test.csv')
df_test = df_test[df_test['class'] != 'pdoscc']
print(df_test.head(5))

y_pred = []

label_2a = ['wmdoscc','pdoscc']
label_2b = ['wdoscc','mdoscc']
datagen_test = ImageDataGenerator(rescale = 1.0/255.0)
test_generator = datagen_test.flow_from_dataframe(
        dataframe=df_test,
        folder=datapath,
        target_size=(300, 300),
        class_mode='categorical',
        shuffle=False,
        validate_filenames=False)

# model_2a
print("Model 2a")
y_pred_2a = model_2a.predict(test_generator)
y_pred_2a = np.argmax(y_pred_2a, axis=1)
y_pred_2a = [label_2a[i] for i in y_pred_2a]


# model_2b
print("Model 2b")
y_pred_2b = model_2b.predict(test_generator)
y_pred_2b = np.argmax(y_pred_2b, axis=1)
y_pred_2b = [label_2b[i] for i in y_pred_2b]

# combined 3 column datatframe for model_2a, model_2b and final prediction
df = pd.DataFrame({'model_2a':y_pred_2a, 'model_2b':y_pred_2b,'ground_truth':df_test['class'].values.tolist()})
# df['correct_prediction'] = np.where(df['model_2a'] == df['ground_truth'] or df['model_2b'] == df['ground_truth'], 1, 0)
df.to_csv(f'{out_path}/test_pipeline_output.csv', index=False)