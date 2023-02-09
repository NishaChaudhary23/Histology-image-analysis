import os
import numpy as np
import pandas as pd
import warnings
import multiprocessing as mp
from multiprocessing import Pool
warnings.filterwarnings('ignore')

def image_generator(case, batch_size=32):
    path = "/home/chs.rintu/Documents/office/researchxoscc/project_2/dataSet/train/"
    o_path = "/home/chs.rintu/Documents/office/researchxoscc/project_2/dataSet/pipeline"
    appendpath = '/home/chs.rintu/Documents/office/researchxoscc/project_2/dataSet/train_all'
    folders = ['wdoscc', 'mdoscc', 'pdoscc']
    train_df = pd.DataFrame(columns=['filename', 'class'])
    test_df = pd.DataFrame(columns=['filename', 'class'])
    if not os.path.isdir(os.path.join(o_path,"wm_p")):
        os.makedirs(os.path.join(o_path,"wm_p"), exist_ok=True)
    outpath1 = os.path.join(o_path,"wm_p") 
    if not os.path.isdir(os.path.join(o_path,"mp_w")):
        os.makedirs(os.path.join(o_path,"mp_w"), exist_ok=True)
    outpath4 = os.path.join(o_path,"w_m") 
    if not os.path.isdir(os.path.join(o_path,"m_p")):
        os.makedirs(os.path.join(o_path,"m_p"), exist_ok=True)
    outpath7 = os.path.join(o_path,"all")
    if case ==1:
        print("Case 1")
        foldpath_1 = os.path.join(path, folders[0])
        files_1 = os.listdir(foldpath_1)
        foldpath_2 = os.path.join(path, folders[1])
        files_2 = os.listdir(foldpath_2)
        foldpath_3 = os.path.join(path, folders[2])
        files_3 = os.listdir(foldpath_3)
        for file in files_1[:len(files_1)//4]:
            train_df = train_df.append({'filename': os.path.join(appendpath,file), 'class': "wmdoscc"}, ignore_index=True)
        for file in files_2[:len(files_2)//4]:
            train_df = train_df.append({'filename': os.path.join(appendpath,file), 'class': "wmdoscc"}, ignore_index=True)
        for file in files_3[:len(files_3)//2]:
            train_df = train_df.append({'filename': os.path.join(appendpath,file), 'class': "pdoscc"}, ignore_index=True)
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        train_df.to_csv(os.path.join(outpath1, "train.csv"), index=False)
        print(train_df)
    if case ==2:
        print("Case 2")
        foldpath_1 = os.path.join(path, folders[0])
        files_1 = os.listdir(foldpath_1)
        foldpath_2 = os.path.join(path, folders[1])
        files_2 = os.listdir(foldpath_2)
        foldpath_3 = os.path.join(path, folders[2])
        files_3 = os.listdir(foldpath_3)
        for file in files_1[len(files_1)//4:len(files_1)//2]:
            test_df = test_df.append({'filename': os.path.join(appendpath,file), 'class': "wmdoscc"}, ignore_index=True)
        for file in files_2[len(files_2)//4:len(files_2)//2]:
            test_df = test_df.append({'filename': os.path.join(appendpath,file), 'class': "wmdoscc"}, ignore_index=True)
        for file in files_3[len(files_3)//2:len(files_3)-len(files_3)//4]:
            test_df = test_df.append({'filename': os.path.join(appendpath,file), 'class': "pdoscc"}, ignore_index=True)
        test_df = test_df.sample(frac=1).reset_index(drop=True)
        test_df.to_csv(os.path.join(outpath1, "test.csv"), index=False)
        print(test_df)
    if case ==7:
        print("Case 7")
        foldpath_1 = os.path.join(path, folders[0])
        files_1 = os.listdir(foldpath_1)
        foldpath_2 = os.path.join(path, folders[1])
        files_2 = os.listdir(foldpath_2)
        for file in files_1[:len(files_1)//2]:
            train_df = train_df.append({'filename': os.path.join(appendpath,file), 'class': "wdoscc"}, ignore_index=True)
        for file in files_2[:len(files_2)//2]:
            train_df = train_df.append({'filename': os.path.join(appendpath,file), 'class': "mdoscc"}, ignore_index=True)
        train_df.to_csv(os.path.join(outpath4, "train.csv"), index=False)
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        print(train_df)
    if case ==13:
        print("Case 13")
        foldpath_1 = os.path.join(path, folders[0])
        files_1 = os.listdir(foldpath_1)
        foldpath_2 = os.path.join(path, folders[1])
        files_2 = os.listdir(foldpath_2)
        foldpath_3 = os.path.join(path, folders[2])
        files_3 = os.listdir(foldpath_3)
        for file in files_1[len(files_1)//2:]:
            train_df = train_df.append({'filename': os.path.join(appendpath,file), 'class': "wdoscc"}, ignore_index=True)
        for file in files_2[len(files_2)//2:]:
            train_df = train_df.append({'filename': os.path.join(appendpath,file), 'class': "mdoscc"}, ignore_index=True)
        for file in files_3[len(files_3)//2:]:
            train_df = train_df.append({'filename': os.path.join(appendpath,file), 'class': "pdoscc"}, ignore_index=True)
        # shuffle the dataframe
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        train_df.to_csv(os.path.join(outpath7, "master_test.csv"), index=False)
        print(train_df)


if __name__ == "__main__":
    pool = Pool(mp.cpu_count())

    pool.map(image_generator, [1,2,7,13])

image_generator(13)