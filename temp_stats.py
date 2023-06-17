import os
import pandas as pd 
import numpy as np


path = '/Users/mraoaakash/Documents/research/research-nisha/ORCHID_data/ORCHID_model_data/ORCHID_train/OSCC/WDOSCC'
test_path = '/Users/mraoaakash/Documents/research/research-nisha/ORCHID_data/ORCHID_model_data/ORCHID_test/OSCC/WDOSCC'

df = pd.read_csv(os.path.join(path, 'patch_metadata.csv'))
train_dataframe=pd.DataFrame(columns=df.columns)
print(len(df))
list = os.listdir(path)
# list = list.remove('.DS_Store') if '.DS_Store' in list else list
# list = list.remove('patch_metadata.csv') if 'patch_metadata.csv' in list else list
print(list)
for file in list:
    if '.DS_Store' in file or 'patch_metadata.csv' in file:
        continue
    patient = file.split('.')[0]
    # dropping the patientid from df
    train_dataframe = train_dataframe.append(df[df['patient_id'] == patient])
    df = df[df['patient_id'] != patient]

    print(patient)

# printing df length
print(len(df))
print(len(train_dataframe))
print(len(train_dataframe)+len(df))

train_dataframe.to_csv(os.path.join(path, 'patch_train_metadata.csv'), index=False)
df.to_csv(os.path.join(test_path, 'patch_test_metadata.csv'), index=False)