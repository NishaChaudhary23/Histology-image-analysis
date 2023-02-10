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
print(df_test.head(5))

y_pred = []

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
for ID in df_test['filename'].values.tolist():
    true_label = df_test[df_test['filename'] == ID]['class'].values[0]
    img = tf.keras.preprocessing.image.load_img(
        f'{ID}', target_size=(300, 300)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model_2a.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(score)
    print(predictions)
    pred_label = label_2a[np.argmax(score)]
    if pred_label == "wmdoscc":
        prediction = model_2b.predict(img_array)
        score = tf.nn.softmax(prediction[0])
        pred_label = label_2b[np.argmax(score)]
        print(
            "This image,{} most likely belongs to {} with a {:.2f} percent confidence. the original label is {}"
            .format(ID.split("/")[-1],pred_label, 100 * np.max(score), true_label)
        )
    else:
        print(
            "This image, {}, most likely belongs to {} with a {:.2f} percent confidence. the original label is {}"
            .format(ID.split("/")[-1],pred_label, 100 * np.max(score), true_label)
        )
    y_pred.append(pred_label)
y_true = df_test['class'].values.tolist()
# classification report
print(classification_report(y_true, y_pred))
# saving classification report to csv
report = classification_report(y_true, y_pred, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv(f'{out_path}/combined/combined_classification_report.csv')
# confusion matrix
cm = confusion_matrix(y_true, y_pred)
# saving confusion matrix to csv
df_cm = pd.DataFrame(cm, index = [i for i in ['pdoscc','wdoscc','mdoscc']],
                columns = [i for i in ['pdoscc','wdoscc','mdoscc']])
df_cm.to_csv(f'{out_path}/combined/combined_confusion_matrix.csv')
print(cm)
    # plot confusion matrix
# predictions = model_2a.predict()
# y_pred = np.argmax(predictions, axis=1)
# print(predictions)
# print(y_pred)