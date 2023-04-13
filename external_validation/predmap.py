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
# from support import get_all_roc_coordinates, plot_roc_curve, calculate_tpr_fpr
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns



os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

#checking tensorflow version
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
paths = ['/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/normal', '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/osmf', '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/oscc']
datapath = '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/all'
plotpath = '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/plots/project_1/'

if not os.path.exists(datapath):
    os.makedirs(datapath)

master_dataframe = pd.DataFrame()

for i in paths:
    items = os.listdir(i)
    # shuffling the list
    np.random.shuffle(items)
    # taking the first 1000 items
    items = items[:1000]
    print(len(items))
    # adding the path to the items
    items = [os.path.join(datapath, item) for item in items]
    # creating a dataframe
    df = pd.DataFrame(items, columns=['filename'])
    # adding the label
    df['class'] = i.split('/')[-1]
    # appending the dataframe to the master dataframe
    master_dataframe = master_dataframe.append(df)

# print(master_dataframe)
# shuffling the master dataframe
# master_dataframe = master_dataframe.sample(frac=1).reset_index(drop=True)
# print(master_dataframe)


datagen_test = ImageDataGenerator(rescale = 1.0/255.0)
# Test Data
test_generator = datagen_test.flow_from_dataframe(
        dataframe=master_dataframe,
        folder=datapath,
        target_size=(300, 300),
        class_mode='categorical',
        shuffle=False,
        validate_filenames=False)


conf_key = [*test_generator.class_indices.keys()]
print(conf_key)

# loading the model
model = load_model('/home/chs.rintu/Documents/office/researchxoscc/project_1/InceptionV3/model_log/model-03-0.97.h5')
# model.summary()

for layer in model.layers:
    if 'conv' not in layer.name:
        print(layer.name)