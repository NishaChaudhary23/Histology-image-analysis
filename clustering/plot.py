import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


paths = ['/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/normal', '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/osmf', '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/oscc']
datapath = '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/all'
plotpath = '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/plots/project_1/'
outpath = '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/'

if not os.path.exists(datapath):
    os.makedirs(datapath)

master_dataframe = pd.read_csv(os.path.join(outpath, 'tsne_data.csv'))
print(master_dataframe.head())

plot = sns.scatterplot(x='Dim_1', y='Dim_2', hue='labels', data=master_dataframe)
plot.set_title('TSNE Plot')
plot.figure.savefig(os.path.join(outpath, 'tsne_plot.png'))
