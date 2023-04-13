import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


for folder in os.listdir('/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/'):
    if folder == 'all':
        continue
    print(folder)
    print(len(os.listdir(os.path.join('/home/chs.rintu/Documents/office/researchxoscc/Ensemble/external_validation_data/images/external validation-P1/', folder))))
    