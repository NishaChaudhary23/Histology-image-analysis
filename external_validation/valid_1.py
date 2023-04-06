#Importing necessary packages
import pandas as pd
import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from tensorflow.keras.metrics import AUC, PrecisionAtRecall

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

#checking tensorflow version
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
paths = ['/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/normal', '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/osmf', '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/oscc']
datapath = '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/all'
plotpath = '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/plots/project_1'

if not os.path.exists(datapath):
    os.makedirs(datapath)

master_dataframe = pd.DataFrame()
conf_key = ['normal', 'osmf', 'oscc']

for i in paths:
    items = os.listdir(i)
    # shuffling the list
    np.random.shuffle(items)
    # taking the first 1000 items
    items = items[:1000]
    # adding the path to the items
    items = [os.path.join(datapath, item) for item in items]
    # creating a dataframe
    df = pd.DataFrame(items, columns=['filename'])
    # adding the label
    df['class'] = i.split('/')[-1]
    # appending the dataframe to the master dataframe
    master_dataframe = master_dataframe.append(df)

print(master_dataframe)
# shuffling the master dataframe
master_dataframe = master_dataframe.sample(frac=1).reset_index(drop=True)
print(master_dataframe)


datagen_test = ImageDataGenerator(rescale = 1.0/255.0)
# Test Data
test_generator = datagen_test.flow_from_dataframe(
        dataframe=master_dataframe,
        folder=datapath,
        target_size=(300, 300),
        class_mode='categorical',
        shuffle=True,
        validate_filenames=False)


# loading the model
model = load_model('/home/chs.rintu/Documents/office/researchxoscc/project_1/InceptionV3-20230404T121058Z-001/InceptionV3/InceptionV3.h5')
model.summary()

eval = model.predict(test_generator)
# finding the class with the highest probability
scores = eval
eval = np.argmax(eval, axis=1)
# converting the class to the original label
eval = [conf_key[i] for i in eval]
gt = np.array(test_generator.classes)
gt = [conf_key[i] for i in gt]
conf = confusion_matrix(gt, eval)
# conf = pd.DataFrame(conf, columns=conf_key, index=conf_key)

# conf = conf.values[:,1:]
conf = conf.astype(np.int32)
conf_percentages = conf / conf.sum(axis=1)[:, np.newaxis]
conf_percentages = conf_percentages * 100
conf_percentages = np.round(conf_percentages, 2).flatten()
labels = [f"{v1}\n{v2}%" for v1, v2 in
        zip(conf.flatten(),conf_percentages)]
labels = np.asarray(labels).reshape(3,3)
plt.figure(figsize=(3.5,3))
sns.heatmap(conf_percentages.reshape((3,3)), annot=labels, xticklabels=conf_key, cmap=sns.color_palette("ch:s=-.2,r=.6", as_cmap=True), yticklabels=conf_key, fmt='', cbar=True, annot_kws={"font":'Sans',"size": 9.5,"fontstyle":'italic' })
plt.xlabel('Predicted',fontname="Sans", fontsize=9, labelpad=10,fontweight='bold')
plt.ylabel('Ground Truth',fontname="Sans", fontsize=9, labelpad=10,fontweight='bold')
plt.title(f'External Validation Confusion Matrix',fontname="Sans", fontsize=11,fontweight='bold')
plt.tight_layout()
plt.savefig(f'{plotpath}project_1_exVal_cm.png', dpi = 300)


auc = AUC()(test_generator.classes, np.zeros(len(test_generator.classes)))
print("AUC:", auc.numpy())


fpr, tpr, thresholds = roc_curve(test_generator.classes, np.zeros(len(test_generator.classes)))

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()