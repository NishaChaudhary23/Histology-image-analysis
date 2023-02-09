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
# model_2a.summary()
print("------------------------------------------")
print("-----------------Model 2b-----------------")
# model_2b.summary()
print("------------------------------------------")

base = '/home/chs.rintu/Documents/office/researchxoscc/project_2/dataSet'
out_path = '/home/chs.rintu/Documents/office/researchxoscc/project_2/output'
datapath = f'{base}/train_all'


df_test = pd.read_csv(f'{base}/pipeline/all/master_test.csv')
# dropping columns of image and class
df_test = df_test.drop(['image','label'], axis=1)
# renaming filename and label to image and class
# df_test = df_test.rename(columns={'':'filename'})
print(df_test.head(5))



label_2a = ['wmdoscc','pdoscc']
label_2b = ['wdoscc','mdoscc']
# datagen_test = ImageDataGenerator(rescale = 1.0/255.0)
# test_generator = datagen_test.flow_from_dataframe(
#         dataframe=df_test,
#         folder=datapath,
#         target_size=(300, 300),
#         class_mode='categorical',
#         shuffle=False,
#         validate_filenames=False)
for ID in df_test['filename']:
    img = tf.keras.preprocessing.image.load_img(
        f'{ID}', target_size=(300, 300)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model_2a.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    if label_2a[np.argmax(score)] == "wmdoscc":
        prediction = model_2b.predict(img_array)
        score = tf.nn.softmax(prediction[0])
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(label_2b[np.argmax(score)], 100 * np.max(score))
        )
    else:
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(label_2a[np.argmax(score)], 100 * np.max(score))
        )
# predictions = model_2a.predict()
# y_pred = np.argmax(predictions, axis=1)
# print(predictions)
# print(y_pred)