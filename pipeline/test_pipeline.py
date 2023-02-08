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