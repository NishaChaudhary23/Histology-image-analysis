import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from progress.bar import Bar


paths = ['/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/normal', '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/osmf', '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/oscc']
datapath = '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/all'
plotpath = '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/plots/project_1/'
outpath = '/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/'

if not os.path.exists(datapath):
    os.makedirs(datapath)

# master_dataframe = pd.DataFrame()

# for i in paths:
#     items = os.listdir(i)
#     # shuffling the list
#     np.random.shuffle(items)
#     # taking the first 1000 items
#     items = items[:1000]
#     print(len(items))
#     # adding the path to the items
#     items = [os.path.join(datapath, item) for item in items]
#     # creating a dataframe
#     df = pd.DataFrame(items, columns=['filename'])
#     # adding the label
#     df['class'] = i.split('/')[-1]
#     # appending the dataframe to the master dataframe
#     master_dataframe = master_dataframe.append(df)

# # print(master_dataframe)
# # shuffling the master dataframe
# # master_dataframe = master_dataframe.sample(frac=1).reset_index(drop=True)
# print(master_dataframe.head())

# # saving the master dataframe
# master_dataframe.to_csv(os.path.join(outpath, 'master_dataframe.csv'), index=False)

master_dataframe = pd.read_csv(os.path.join(outpath, 'master_dataframe.csv'))
print(master_dataframe.head())

with Bar('Comparing', max=len(master_dataframe)*len(master_dataframe),suffix='%(percent).1f%% - %(eta)ds') as bar:
    # iterating throuhg every row of the dataframe
        for index_1, row_1 in master_dataframe.iterrows():
            # opening the base image for comparison
            image_base = cv2.imread(row_1['filename'])
            # iterating through every row of the dataframe
            for index_2, row_2 in master_dataframe.iterrows():
                if index_1 != index_2:
                    image_test = cv2.imread(row_2['filename'])
                    # print(row_1['filename'], row_2['filename'])
                bar.next()