import os
import shutil
import pandas as pd
import tarfile


path = "/Users/mraoaakash/Documents/research/research-nisha/ORCHID_data/ORCHID_model_data_transfer/ORCHID_train/OSCC/WDOSCC"
outpath = "/Users/mraoaakash/Documents/research/research-nisha/ORCHID_data/ORCHID_model_data_transfer/model_2/train/WDOSCC"

df = pd.read_csv(os.path.join(path,"patch_train_metadata.csv"))
# shuffling the dataframe
df = df.sample(frac=1).reset_index(drop=True)
# sampling the first 30000 rows
df = df.iloc[:30000]
# adding a label column
df["label"] = "WDOSCC"

df.to_csv(os.path.join(outpath,"WDOSCC_metadata.csv"), index=False)

# iterating through the dataframe
for index, row in df.iterrows():
    # getting the file name
    filename = row["patient_id"]+"/patches/"+row["patch_id"]+".png"
    # getting the file path
    filepath = os.path.join(path, filename)
    print(filepath)
    out = outpath+"/"+row["patch_id"]+".png"
    shutil.copyfile(filepath, out)