import os 
import pandas as pd

path = "/Users/mraoaakash/Documents/research/research-nisha/ORCHID_data/ORCHID_model_data/ORCHID_test/OSCC"

sums=0
for i in ["WDOSCC", "MDOSCC", "PDOSCC"]:
    print(i)
    df = pd.read_csv(os.path.join(path, f'{i}/patch_test_metadata.csv'))
    # finding unique image_id
    image_ids = df['image_id'].unique()
    # print(len(image_ids))
    # sums += len(image_ids)
    print(len(df))
    sums += len(df)

print(sums)

print("--------------------------------------------------")
path = "/Users/mraoaakash/Documents/research/research-nisha/ORCHID_data/ORCHID_model_data/ORCHID_test"

sums=0
for i in ["Normal","OSMF"]:
    print(i)
    df = pd.read_csv(os.path.join(path, f'{i}/patch_test_metadata.csv'))
    # finding unique image_id
    image_ids = df['image_id'].unique()
    # print(len(image_ids))
    # sums += len(image_ids)
    print(len(df))
    sums += len(df)

print(sums)