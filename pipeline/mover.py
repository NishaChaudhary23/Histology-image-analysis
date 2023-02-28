import os
import shutil
from multiprocessing import Pool
import multiprocessing as mp

def mover(filename):
    for root, dirs, files in os.walk(f'/home/chs.rintu/Documents/office/researchxoscc/project_1/dataSet/train/{filename}'):
        for file in files:
            if file.endswith('.tif'):
                shutil.copy(os.path.join(root, file), '/home/chs.rintu/Documents/office/researchxoscc/project_1/dataSet/train_all')


if __name__ == '__main__':
    pool = Pool(mp.cpu_count())
    pool.map(mover, os.listdir('/home/chs.rintu/Documents/office/researchxoscc/project_1/dataSet/train'))