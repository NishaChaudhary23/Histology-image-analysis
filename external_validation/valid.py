#Importing necessary packages
import pandas as pd
import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import os
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import RMSprop

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

#checking tensorflow version
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
paths = ['/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/normal', '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/osmf', '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/oscc']
master_dataframe = pd.DataFrame()
for i in paths:
    items = os.listdir(i)
    # shuffling the list
    np.random.shuffle(items)
    # taking the first 1000 items
    items = items[:1000]
    # adding the path to the items
    items = [os.path.join(i, item) for item in items]
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

