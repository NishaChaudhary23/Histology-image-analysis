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
    folders = ['wdoscc', 'mdoscc', 'pdoscc']
    train_df = pd.DataFrame(columns=['image', 'label'])
    test_df = pd.DataFrame(columns=['image', 'label'])
    if not os.path.isdir(os.path.join(o_path,"wm_p")):
        os.makedirs(os.path.join(o_path,"wm_p"))
    outpath1 = os.path.join(o_path,"wm_p") 
    if not os.path.isdir(os.path.join(o_path,"mp_w")):
        os.makedirs(os.path.join(o_path,"mp_w"))
    outpath2 = os.path.join(o_path,"mp_w") 
    if not os.path.isdir(os.path.join(o_path,"pw_m")):
        os.makedirs(os.path.join(o_path,"pw_m"))
    outpath3 = os.path.join(o_path,"pw_m") 
    if not os.path.isdir(os.path.join(o_path,"w_m")):
        os.makedirs(os.path.join(o_path,"w_m"))
    outpath4 = os.path.join(o_path,"w_m") 
    if not os.path.isdir(os.path.join(o_path,"m_p")):
        os.makedirs(os.path.join(o_path,"m_p"))
    outpath5 = os.path.join(o_path,"m_p") 
    if not os.path.isdir(os.path.join(o_path,"p_w")):
        os.makedirs(os.path.join(o_path,"p_w"))
    outpath6 = os.path.join(o_path,"p_w") 
    if case ==1:
        print("Case 1")
        foldpath_1 = os.path.join(path, folders[0])
        files_1 = os.listdir(foldpath_1)
        foldpath_2 = os.path.join(path, folders[1])
        files_2 = os.listdir(foldpath_2)
        foldpath_3 = os.path.join(path, folders[2])
        files_3 = os.listdir(foldpath_3)
        for file in files_1[:len(files_1)//4]:
            train_df = train_df.append({'image': os.path.join(path, file), 'label': 0}, ignore_index=True)
        for file in files_2[:len(files_2)//4]:
            train_df = train_df.append({'image': os.path.join(path, file), 'label': 0}, ignore_index=True)
        for file in files_3[:len(files_3)//2]:
            train_df = train_df.append({'image': os.path.join(path, file), 'label': 1}, ignore_index=True)
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
            test_df = test_df.append({'image': os.path.join(path, file), 'label': 0}, ignore_index=True)
        for file in files_2[len(files_2)//4:len(files_2)//2]:
            test_df = test_df.append({'image': os.path.join(path, file), 'label': 0}, ignore_index=True)
        for file in files_3[len(files_3)//2:len(files_3)-len(files_3)//4]:
            test_df = test_df.append({'image': os.path.join(path, file), 'label': 1}, ignore_index=True)
        test_df.to_csv(os.path.join(outpath1, "test.csv"), index=False)
        print(test_df)
    if case ==3:
        print("Case 3")
        foldpath_1 = os.path.join(path, folders[1])
        files_1 = os.listdir(foldpath_1)
        foldpath_2 = os.path.join(path, folders[2])
        files_2 = os.listdir(foldpath_2)
        foldpath_3 = os.path.join(path, folders[0])
        files_3 = os.listdir(foldpath_3)
        for file in files_1[:len(files_1)//4]:
            train_df = train_df.append({'image': os.path.join(path, file), 'label': 0}, ignore_index=True)
        for file in files_2[:len(files_2)//4]:
            train_df = train_df.append({'image': os.path.join(path, file), 'label': 0}, ignore_index=True)
        for file in files_3[:len(files_3)//2]:
            train_df = train_df.append({'image': os.path.join(path, file), 'label': 1}, ignore_index=True)
        train_df.to_csv(os.path.join(outpath2, "train.csv"), index=False)
        print(train_df)
    if case ==4:
        print("Case 4")
        foldpath_1 = os.path.join(path, folders[1])
        files_1 = os.listdir(foldpath_1)
        foldpath_2 = os.path.join(path, folders[2])
        files_2 = os.listdir(foldpath_2)
        foldpath_3 = os.path.join(path, folders[0])
        files_3 = os.listdir(foldpath_3)
        for file in files_1[len(files_1)//4:len(files_1)//2]:
            test_df = test_df.append({'image': os.path.join(path, file), 'label': 0}, ignore_index=True)
        for file in files_2[len(files_2)//4:len(files_2)//2]:
            test_df = test_df.append({'image': os.path.join(path, file), 'label': 0}, ignore_index=True)
        for file in files_3[len(files_3)//2:len(files_3)-len(files_3)//4]:
            test_df = test_df.append({'image': os.path.join(path, file), 'label': 1}, ignore_index=True)
        test_df.to_csv(os.path.join(outpath2, "test.csv"), index=False)
        print(test_df)
    if case ==5:
        print("Case 5")
        foldpath_1 = os.path.join(path, folders[2])
        files_1 = os.listdir(foldpath_1)
        foldpath_2 = os.path.join(path, folders[0])
        files_2 = os.listdir(foldpath_2)
        foldpath_3 = os.path.join(path, folders[1])
        files_3 = os.listdir(foldpath_3)
        for file in files_1[:len(files_1)//4]:
            train_df = train_df.append({'image': os.path.join(path, file), 'label': 0}, ignore_index=True)
        for file in files_2[:len(files_2)//4]:
            train_df = train_df.append({'image': os.path.join(path, file), 'label': 0}, ignore_index=True)
        for file in files_3[:len(files_3)//2]:
            train_df = train_df.append({'image': os.path.join(path, file), 'label': 1}, ignore_index=True)
        train_df.to_csv(os.path.join(outpath3, "train.csv"), index=False)
        print(train_df)
    if case ==6:
        print("Case 6")
        foldpath_1 = os.path.join(path, folders[2])
        files_1 = os.listdir(foldpath_1)
        foldpath_2 = os.path.join(path, folders[0])
        files_2 = os.listdir(foldpath_2)
        foldpath_3 = os.path.join(path, folders[1])
        files_3 = os.listdir(foldpath_3)
        for file in files_1[len(files_1)//4:len(files_1)//2]:
            test_df = test_df.append({'image': os.path.join(path, file), 'label': 0}, ignore_index=True)
        for file in files_2[len(files_2)//4:len(files_2)//2]:
            test_df = test_df.append({'image': os.path.join(path, file), 'label': 0}, ignore_index=True)
        for file in files_3[len(files_3)//2:len(files_2)-len(files_2)//4]:
            test_df = test_df.append({'image': os.path.join(path, file), 'label': 1}, ignore_index=True)
        test_df.to_csv(os.path.join(outpath3, "test.csv"), index=False)
        print(test_df)
    if case ==7:
        print("Case 7")
        foldpath_1 = os.path.join(path, folders[0])
        files_1 = os.listdir(foldpath_1)
        foldpath_2 = os.path.join(path, folders[1])
        files_2 = os.listdir(foldpath_2)
        for file in files_1[:len(files_1)//2]:
            train_df = train_df.append({'image': os.path.join(path, file), 'label': 0}, ignore_index=True)
        for file in files_2[:len(files_2)//2]:
            train_df = train_df.append({'image': os.path.join(path, file), 'label': 1}, ignore_index=True)
        train_df.to_csv(os.path.join(outpath4, "train.csv"), index=False)
        print(train_df)
    if case ==8:
        print("Case 8")
        foldpath_1 = os.path.join(path, folders[0])
        files_1 = os.listdir(foldpath_1)
        foldpath_2 = os.path.join(path, folders[1])
        files_2 = os.listdir(foldpath_2)
        for file in files_1[len(files_1)//2:len(files_1)-len(files_1)//4]:
            test_df = test_df.append({'image': os.path.join(path, file), 'label': 0}, ignore_index=True)
        for file in files_2[len(files_2)//2:len(files_2)-len(files_2)//4]:
            test_df = test_df.append({'image': os.path.join(path, file), 'label': 1}, ignore_index=True)
        test_df.to_csv(os.path.join(outpath4, "test.csv"), index=False)
        print(test_df)
    if case ==9:
        print("Case 9")
        foldpath_1 = os.path.join(path, folders[1])
        files_1 = os.listdir(foldpath_1)
        foldpath_2 = os.path.join(path, folders[2])
        files_2 = os.listdir(foldpath_2)
        for file in files_1[:len(files_1)//2]:
            train_df = train_df.append({'image': os.path.join(path, file), 'label': 0}, ignore_index=True)
        for file in files_2[:len(files_2)//2]:
            train_df = train_df.append({'image': os.path.join(path, file), 'label': 1}, ignore_index=True)
        train_df.to_csv(os.path.join(outpath5, "train.csv"), index=False)
        print(train_df)
    if case ==10:
        print("Case 10")
        foldpath_1 = os.path.join(path, folders[1])
        files_1 = os.listdir(foldpath_1)
        foldpath_2 = os.path.join(path, folders[2])
        files_2 = os.listdir(foldpath_2)
        for file in files_1[len(files_1)//2:len(files_1)-len(files_1)//4]:
            test_df = test_df.append({'image': os.path.join(path, file), 'label': 0}, ignore_index=True)
        for file in files_2[len(files_2)//2:len(files_2)-len(files_2)//4]:
            test_df = test_df.append({'image': os.path.join(path, file), 'label': 1}, ignore_index=True)
        test_df.to_csv(os.path.join(outpath5, "test.csv"), index=False)
        print(test_df)
    if case ==11:
        print("Case 11")
        foldpath_1 = os.path.join(path, folders[2])
        files_1 = os.listdir(foldpath_1)
        foldpath_2 = os.path.join(path, folders[0])
        files_2 = os.listdir(foldpath_2)
        for file in files_1[:len(files_1)//4]:
            train_df = train_df.append({'image': os.path.join(path, file), 'label': 0}, ignore_index=True)
        for file in files_2[:len(files_2)//4]:
            train_df = train_df.append({'image': os.path.join(path, file), 'label': 1}, ignore_index=True)
        train_df.to_csv(os.path.join(outpath6, "train.csv"), index=False)
        print(train_df)
    if case ==12:
        print("Case 12")
        foldpath_1 = os.path.join(path, folders[2])
        files_1 = os.listdir(foldpath_1)
        foldpath_2 = os.path.join(path, folders[0])
        files_2 = os.listdir(foldpath_2)
        for file in files_1[len(files_1)//2:len(files_1)-len(files_1)//4]:
            test_df = test_df.append({'image': os.path.join(path, file), 'label': 0}, ignore_index=True)
        for file in files_2[len(files_2)//2:len(files_2)-len(files_2)//4]:
            test_df = test_df.append({'image': os.path.join(path, file), 'label': 1}, ignore_index=True)
        test_df.to_csv(os.path.join(outpath6, "test.csv"), index=False)
        print(test_df)


if __name__ == "__main__":
    pool = Pool(mp.cpu_count())
    pool.map(image_generator, range(1, 13))
# image_generator("/home/chs.rintu/Documents/chs-lab-ws02/nisha/project-2-oscc/data/original", 1)