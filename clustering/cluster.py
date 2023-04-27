import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from progress.bar import Bar
from numpy.linalg import norm
from sklearn.manifold import TSNE


paths = ['/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/normal', '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/osmf', '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/oscc']
datapath = '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/all'
plotpath = '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/plots/project_1/'
outpath = '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/'

if not os.path.exists(datapath):
    os.makedirs(datapath)

master_dataframe = pd.read_csv(os.path.join(outpath, 'master_dataframe.csv'))
print(master_dataframe.head())

# getting all images and labels saving them in a list
images = []
labels = []
for i in range(len(master_dataframe)):
    images.append(cv2.imread(master_dataframe['filename'][i]))
    labels.append(master_dataframe['class'][i])

print(len(images))
print(len(labels))