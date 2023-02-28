import os
import numpy as np
import pandas as pd
import warnings
import multiprocessing as mp
from multiprocessing import Pool
warnings.filterwarnings('ignore')



def five_fold_datagen(i):
    path = "/home/chs.rintu/Documents/office/researchxoscc/project_1/dataSet/train/"
    o_path = "/home/chs.rintu/Documents/office/researchxoscc/project_1/dataSet/pipeline"
    appendpath = '/home/chs.rintu/Documents/office/researchxoscc/project_1/dataSet/train_all'
    folders = ['normal', 'oscc', 'osmf']
    # folders = ['normaloscc', 'osccoscc', 'pdoscc']

    normal_files = os.listdir(os.path.join(path, folders[0]))
    oscc_files = os.listdir(os.path.join(path, folders[1]))
    osmf_files = os.listdir(os.path.join(path, folders[2]))

    print(f'normal: {len(normal_files)}')
    print(f'oscc: {len(oscc_files)}')
    print(f'osmf: {len(osmf_files)}')




    outpath = os.path.join(o_path, "data_fold_" + str(i))
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    combined_train_df = pd.DataFrame(columns=['filename', 'class'])
    combined_internal_val = pd.DataFrame(columns=['filename', 'class'])
    normal_train_df = pd.DataFrame(columns=['filename', 'class'])
    oscc_train_df = pd.DataFrame(columns=['filename', 'class'])
    osmf_train_df = pd.DataFrame(columns=['filename', 'class'])
    normal_internal_val = pd.DataFrame(columns=['filename', 'class'])
    oscc_internal_val = pd.DataFrame(columns=['filename', 'class'])
    osmf_internal_val = pd.DataFrame(columns=['filename', 'class'])

    normal_internal_val_list = normal_files[i * len(normal_files) // 5: (i + 1) * len(normal_files) // 5]
    oscc_internal_val_list = oscc_files[i * len(oscc_files) // 5: (i + 1) * len(oscc_files) // 5]
    osmf_internal_val_list = osmf_files[i * len(osmf_files) // 5: (i + 1) * len(osmf_files) // 5]

    for file in normal_files:
        appendpath = os.path.join(path, folders[0])
        if file in normal_internal_val_list:
            normal_internal_val = normal_internal_val.append({'filename': os.path.join(appendpath, file), 'class': "normal"},ignore_index=True)
        else:
            normal_train_df = normal_train_df.append({'filename': os.path.join(appendpath, file), 'class': "normal"},ignore_index=True)
    for file in oscc_files:
        appendpath = os.path.join(path, folders[1])
        if file in oscc_internal_val_list:
            oscc_internal_val = oscc_internal_val.append({'filename': os.path.join(appendpath, file), 'class': "oscc"},ignore_index=True)
        else:
            oscc_train_df = oscc_train_df.append({'filename': os.path.join(appendpath, file), 'class': "oscc"},ignore_index=True)
    for file in osmf_files:
        appendpath = os.path.join(path, folders[2])
        if file in osmf_internal_val_list:
            osmf_internal_val = osmf_internal_val.append({'filename': os.path.join(appendpath, file), 'class': "osmf"},ignore_index=True)
        else:
            osmf_train_df = osmf_train_df.append({'filename': os.path.join(appendpath, file), 'class': "osmf"},ignore_index=True)
    
    combined_train_df = combined_train_df.append(normal_train_df, ignore_index=True)
    combined_train_df = combined_train_df.append(oscc_train_df, ignore_index=True)
    combined_train_df = combined_train_df.append(osmf_train_df, ignore_index=True)

    combined_internal_val = combined_internal_val.append(normal_internal_val, ignore_index=True)
    combined_internal_val = combined_internal_val.append(oscc_internal_val, ignore_index=True)
    combined_internal_val = combined_internal_val.append(osmf_internal_val, ignore_index=True)

    combined_train_df_model_2a = combined_train_df.copy()
    combined_internal_val_model_2a = combined_internal_val.copy()

    combined_train_df_model_2b = combined_train_df.copy()
    combined_internal_val_model_2b = combined_internal_val.copy()

    # filtering model 2b to include only normal and oscc
    combined_train_df_model_2b = combined_train_df_model_2b[combined_train_df_model_2b['class'] != 'osmf']
    combined_internal_val_model_2b = combined_internal_val_model_2b[combined_internal_val_model_2b['class'] != 'osmf']

    # relabeling model class if normal and oscc to woscc
    combined_train_df_model_2a['class'] = combined_train_df_model_2a['class'].replace('normal', 'woscc')
    combined_train_df_model_2a['class'] = combined_train_df_model_2a['class'].replace('oscc', 'woscc')
    
    combined_internal_val_model_2a['class'] = combined_internal_val_model_2a['class'].replace('normal', 'woscc')
    combined_internal_val_model_2a['class'] = combined_internal_val_model_2a['class'].replace('oscc', 'woscc')

    # dropping the internal validation files from the training set
    internal_val_list_2a = combined_internal_val_model_2a['filename'].tolist()
    todrop = combined_train_df['filename'].isin(internal_val_list_2a)
    combined_train_df_model_2a = combined_train_df_model_2a[~todrop]
    internal_val_list_2b = combined_internal_val_model_2b['filename'].tolist()
    todrop = combined_train_df['filename'].isin(internal_val_list_2b)
    combined_train_df_model_2b = combined_train_df_model_2b[~todrop]

    # saving the dataframes to csvs
    combined_train_df.to_csv(os.path.join(outpath, "master_train.csv"), index=False)
    combined_internal_val.to_csv(os.path.join(outpath, "master_internal_val.csv"), index=False)
    combined_train_df_model_2a.to_csv(os.path.join(outpath, "master_train_model_2a.csv"), index=False)
    combined_internal_val_model_2a.to_csv(os.path.join(outpath, "master_internal_val_model_2a.csv"), index=False)
    combined_train_df_model_2b.to_csv(os.path.join(outpath, "master_train_model_2b.csv"), index=False)
    combined_internal_val_model_2b.to_csv(os.path.join(outpath, "master_internal_val_model_2b.csv"), index=False)


if __name__ == "__main__":
    pool = Pool(mp.cpu_count())
    pool.map(five_fold_datagen, [0,1,2,3,4])