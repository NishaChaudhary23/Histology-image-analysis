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

model_2a1 = load_model('/home/chs.rintu/Documents/office/researchxoscc/project_2/output/M2a/fold_1/M2a.h5')
model_2a2 = load_model('/home/chs.rintu/Documents/office/researchxoscc/project_2/output/M2a/fold_2/M2a.h5')
model_2a3 = load_model('/home/chs.rintu/Documents/office/researchxoscc/project_2/output/M2a/fold_3/M2a.h5')
model_2a4 = load_model('/home/chs.rintu/Documents/office/researchxoscc/project_2/output/M2a/fold_4/M2a.h5')
model_2a5 = load_model('/home/chs.rintu/Documents/office/researchxoscc/project_2/output/M2a/fold_5/M2a.h5')
model_2b1 = load_model('/home/chs.rintu/Documents/office/researchxoscc/project_2/output/M2b/fold_1/M2b.h5')
model_2b2 = load_model('/home/chs.rintu/Documents/office/researchxoscc/project_2/output/M2b/fold_2/M2b.h5')
model_2b3 = load_model('/home/chs.rintu/Documents/office/researchxoscc/project_2/output/M2b/fold_3/M2b.h5')
model_2b4 = load_model('/home/chs.rintu/Documents/office/researchxoscc/project_2/output/M2b/fold_4/M2b.h5')
model_2b5 = load_model('/home/chs.rintu/Documents/office/researchxoscc/project_2/output/M2b/fold_5/M2b.h5')

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

label_2a = ['pdoscc','wmdoscc']
label_2b = ['mdoscc','wdoscc']
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
y_pred_2a1 = model_2a1.predict(test_generator)
confidence_2a = np.argmax(y_pred_2a1, axis=1)
y_pred_2a1 = np.argmax(y_pred_2a1, axis=1)
y_pred_2a1 = [label_2a[i] for i in y_pred_2a1]
# model 2a2
y_pred_2a2 = model_2a2.predict(test_generator)
confidence_2a = np.argmax(y_pred_2a2, axis=1)
y_pred_2a2 = np.argmax(y_pred_2a2, axis=1)
y_pred_2a2 = [label_2a[i] for i in y_pred_2a2]
# model 2a3
y_pred_2a3 = model_2a3.predict(test_generator)
confidence_2a = np.argmax(y_pred_2a3, axis=1)
y_pred_2a3 = np.argmax(y_pred_2a3, axis=1)
y_pred_2a3 = [label_2a[i] for i in y_pred_2a3]
# model 2a4
y_pred_2a4 = model_2a4.predict(test_generator)
confidence_2a = np.argmax(y_pred_2a4, axis=1)
y_pred_2a4 = np.argmax(y_pred_2a4, axis=1)
y_pred_2a4 = [label_2a[i] for i in y_pred_2a4]
# model 2a5
y_pred_2a5 = model_2a5.predict(test_generator)
confidence_2a = np.argmax(y_pred_2a5, axis=1)
y_pred_2a5 = np.argmax(y_pred_2a5, axis=1)
y_pred_2a5 = [label_2a[i] for i in y_pred_2a5]
y_pred_2a = [y_pred_2a1,y_pred_2a2,y_pred_2a3,y_pred_2a4,y_pred_2a5]

# for every individual model, get the majority prediction
y_pred_2a = np.array(y_pred_2a)
y_pred_2a = np.transpose(y_pred_2a)
y_pred_2a_majority = []
for i in y_pred_2a:
    y_pred_2a_majority.append(np.argmax(np.bincount(i)))
y_pred_2a = [label_2a[i] for i in y_pred_2a_majority]
print(y_pred_2a)




# model_2b1
print("Model 2b")
y_pred_2b1 = model_2b1.predict(test_generator)
confidence_2b1 = np.argmax(y_pred_2b1, axis=1)
y_pred_2b1 = np.argmax(y_pred_2b1, axis=1)
y_pred_2b1 = [label_2b[i] for i in y_pred_2b1]
# model 2b2
y_pred_2b2 = model_2b2.predict(test_generator)
confidence_2b2 = np.argmax(y_pred_2b2, axis=1)
y_pred_2b2 = np.argmax(y_pred_2b2, axis=1)
y_pred_2b2 = [label_2b[i] for i in y_pred_2b2]
# model 2b3
y_pred_2b3 = model_2b3.predict(test_generator)
confidence_2b3 = np.argmax(y_pred_2b3, axis=1)
y_pred_2b3 = np.argmax(y_pred_2b3, axis=1)
y_pred_2b3 = [label_2b[i] for i in y_pred_2b3]
# model 2b4
y_pred_2b4 = model_2b4.predict(test_generator)
confidence_2b4 = np.argmax(y_pred_2b4, axis=1)
y_pred_2b4 = np.argmax(y_pred_2b4, axis=1)
y_pred_2b4 = [label_2b[i] for i in y_pred_2b4]
# model 2b5
y_pred_2b5 = model_2b5.predict(test_generator)
confidence_2b5 = np.argmax(y_pred_2b5, axis=1)
y_pred_2b5 = np.argmax(y_pred_2b5, axis=1)
y_pred_2b5 = [label_2b[i] for i in y_pred_2b5]
y_pred_2b = [y_pred_2b1,y_pred_2b2,y_pred_2b3,y_pred_2b4,y_pred_2b5]
y_pred_2b = np.array(y_pred_2b)
print(y_pred_2b)




# # combined 3 column datatframe for model_2a, model_2b and final prediction
# df = pd.DataFrame({'model_2a':y_pred_2a, 'model_2b':y_pred_2b,'ground_truth':df_test['class'].values.tolist()})
# # final prediction 
# df['final_prediction'] = df.apply(lambda x: x['model_2a'] if x['model_2a'] == "pdoscc" else x['model_2b'], axis=1)
# df.to_csv(f'{out_path}/test_pipeline_output.csv', index=False)

# # confusion matrix
# print("Confusion Matrix")
# print(confusion_matrix(df_test['class'].values.tolist(), df['final_prediction'].values.tolist()))
# cm = confusion_matrix(df_test['class'].values.tolist(), df['final_prediction'].values.tolist())
# df_cm = pd.DataFrame(cm, index = [i for i in ["pdoscc","wdoscc","mdoscc"]], columns = [i for i in ["pdoscc","wdoscc","mdoscc"]])
# df_cm.to_csv(f'{out_path}/test_pipeline_output_cm.csv', index=True)
# plt.figure(figsize = (10,7))
# sns.heatmap(df_cm, annot=True, fmt='g')
# plt.savefig(f'{out_path}/test_pipeline_output_cm.png')


# print("Classification Report")
# print(classification_report(df_test['class'].values.tolist(), df['final_prediction'].values.tolist()))
# report = classification_report(df_test['class'].values.tolist(), df['final_prediction'].values.tolist(), output_dict=True)
# df = pd.DataFrame(report).transpose()
# df.to_csv(f'{out_path}/test_pipeline_output_report.csv', index=True)
