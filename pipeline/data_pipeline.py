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
    if not os.path.isdir(os.path.join(o_path,"w_m")):
        os.makedirs(os.path.join(o_path,"w_m"), exist_ok=True)
    outpath4 = os.path.join(o_path,"w_m") 
    if not os.path.isdir(os.path.join(o_path,"w_m")):
        os.makedirs(os.path.join(o_path,"combined"), exist_ok=True)
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
    if case ==8:
        print("Case 8")
        foldpath_1 = os.path.join(path, folders[0])
        files_1 = os.listdir(foldpath_1)
        foldpath_2 = os.path.join(path, folders[1])
        files_2 = os.listdir(foldpath_2)
        for file in files_1[len(files_1)//2:len(files_1)-len(files_1)//4]:
            train_df = train_df.append({'filename': os.path.join(appendpath,file), 'class': "wdoscc"}, ignore_index=True)
        for file in files_2[len(files_2)//2:len(files_2)-len(files_2)//4]:
            train_df = train_df.append({'filename': os.path.join(appendpath,file), 'class': "mdoscc"}, ignore_index=True)
        train_df.to_csv(os.path.join(outpath4, "test.csv"), index=False)
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


def five_fold_datagen(i):
    path = "/home/chs.rintu/Documents/office/researchxoscc/project_2/dataSet/train/"
    o_path = "/home/chs.rintu/Documents/office/researchxoscc/project_2/dataSet/pipeline"
    appendpath = '/home/chs.rintu/Documents/office/researchxoscc/project_2/dataSet/train_all'
    folders = ['wdoscc', 'mdoscc', 'pdoscc']

    wd_files = os.listdir(os.path.join(path, folders[0]))
    md_files = os.listdir(os.path.join(path, folders[1]))
    pd_files = os.listdir(os.path.join(path, folders[2]))

    print(f'wdoscc: {len(wd_files)}')
    print(f'mdoscc: {len(md_files)}')
    print(f'pdoscc: {len(pd_files)}')




    outpath = os.path.join(o_path, "data_fold_" + str(i))
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    combined_train_df = pd.DataFrame(columns=['filename', 'class'])
    combined_internal_val = pd.DataFrame(columns=['filename', 'class'])
    wd_train_df = pd.DataFrame(columns=['filename', 'class'])
    md_train_df = pd.DataFrame(columns=['filename', 'class'])
    pd_train_df = pd.DataFrame(columns=['filename', 'class'])
    wd_internal_val = pd.DataFrame(columns=['filename', 'class'])
    md_internal_val = pd.DataFrame(columns=['filename', 'class'])
    pd_internal_val = pd.DataFrame(columns=['filename', 'class'])

    wd_internal_val_list = wd_files[i * len(wd_files) // 5: (i + 1) * len(wd_files) // 5]
    md_internal_val_list = md_files[i * len(md_files) // 5: (i + 1) * len(md_files) // 5]
    pd_internal_val_list = pd_files[i * len(pd_files) // 5: (i + 1) * len(pd_files) // 5]

    for file in wd_files:
        appendpath = os.path.join(path, folders[0])
        if file in wd_internal_val_list:
            wd_internal_val = wd_internal_val.append({'filename': os.path.join(appendpath, file), 'class': "wdoscc"},ignore_index=True)
        else:
            wd_train_df = wd_train_df.append({'filename': os.path.join(appendpath, file), 'class': "wdoscc"},ignore_index=True)
    for file in md_files:
        appendpath = os.path.join(path, folders[1])
        if file in md_internal_val_list:
            md_internal_val = md_internal_val.append({'filename': os.path.join(appendpath, file), 'class': "mdoscc"},ignore_index=True)
        else:
            md_train_df = md_train_df.append({'filename': os.path.join(appendpath, file), 'class': "mdoscc"},ignore_index=True)
    for file in pd_files:
        appendpath = os.path.join(path, folders[2])
        if file in pd_internal_val_list:
            pd_internal_val = pd_internal_val.append({'filename': os.path.join(appendpath, file), 'class': "pdoscc"},ignore_index=True)
        else:
            pd_train_df = pd_train_df.append({'filename': os.path.join(appendpath, file), 'class': "pdoscc"},ignore_index=True)
    
    combined_train_df = combined_train_df.append(wd_train_df, ignore_index=True)
    combined_train_df = combined_train_df.append(md_train_df, ignore_index=True)
    combined_train_df = combined_train_df.append(pd_train_df, ignore_index=True)

    combined_internal_val = combined_internal_val.append(wd_internal_val, ignore_index=True)
    combined_internal_val = combined_internal_val.append(md_internal_val, ignore_index=True)
    combined_internal_val = combined_internal_val.append(pd_internal_val, ignore_index=True)

    combined_train_df_model_2a = combined_train_df.copy()
    combined_internal_val_model_2a = combined_internal_val.copy()

    combined_train_df_model_2b = combined_train_df.copy()
    combined_internal_val_model_2b = combined_internal_val.copy()

    # filtering model 2b to include only wdoscc and mdoscc
    combined_train_df_model_2b = combined_train_df_model_2b[combined_train_df_model_2b['class'] != 'pdoscc']
    combined_internal_val_model_2b = combined_internal_val_model_2b[combined_internal_val_model_2b['class'] != 'pdoscc']

    # relabeling model class if wdoscc and mdoscc to wmdoscc
    combined_train_df_model_2a['class'] = combined_train_df_model_2a['class'].replace('wdoscc', 'wmdoscc')
    combined_train_df_model_2a['class'] = combined_train_df_model_2a['class'].replace('mdoscc', 'wmdoscc')

    combined_internal_val_model_2a['class'] = combined_internal_val_model_2a['class'].replace('wdoscc', 'wmdoscc')
    combined_internal_val_model_2a['class'] = combined_internal_val_model_2a['class'].replace('mdoscc', 'wmdoscc')

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
    # pool.map(image_generator, [1,2,7,13,8])
