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

# converting images and labels to numpy array
images = np.array(images)
labels = np.array(labels)

# beginning FTSNE
# reshaping images to 1D array
images = images.reshape(len(images), -1)
print(images.shape)

# creating a dataframe with images and labels
df = pd.DataFrame(images)
df['labels'] = labels
print(df.head())

model = TSNE(n_components=2, random_state=0, perplexity=50, n_iter=5000)
tsne_data = model.fit_transform(df.iloc[:, :-1])

# creating a new dataframe which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, df['labels'])).T
tsne_df = pd.DataFrame(data=tsne_data, columns=('Dim_1', 'Dim_2', 'labels'))
tsne_df.to_csv(os.path.join(outpath, 'tsne_data.csv'), index=False)