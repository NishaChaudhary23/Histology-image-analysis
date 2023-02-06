import os
import numpy as np

def image_generator(path, case, batch_size=32):
    folders = ['wdoscc', 'mdoscc', 'pdoscc']
    if case ==1:
        foldpath_1 = os.path.join(path, folders[0])
        files_1 = os.listdir(foldpath_1)
        foldpath_2 = os.path.join(path, folders[1])
        files_2 = os.listdir(foldpath_2)
        foldpath_3 = os.path.join(path, folders[2])
        files_3 = os.listdir(foldpath_3)
        files = files_1[:len(files_1)//25] + files_2[:len(files_2)//25] + files_3[:len(files_3)//50]
        print(files)

image_generator("/home/chs.rintu/Documents/chs-lab-ws02/nisha/project-2-oscc/data/original", 1)