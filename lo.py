import os
# IMPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
#  for f in *.tar.gz; do tar -xvf "$f"; done


# path = "/Users/mraoaakash/Documents/research/research-nisha/ORCHID_data/ORCHID_100x_data/Normal"

# for file in os.listdir(path):
#     if ".DS_Store" in file:
#         continue
#     filepath = os.path.join(path, file)
#     tarpath = os.path.join(path, f'{file.split(".")[0]}.tar.gz')
#     print(filepath)
#     print(tarpath)
#     # running a terminal command to tar the file
#     os.system(f"tar -czvf {tarpath} {filepath}")

path = "/Users/mraoaakash/Documents/research/research-nisha/ORCHID_data/ORCHID_100x_data/OSCC/PDOSCC"

for file in os.listdir(path):
    if ".DS_Store" in file:
        continue
    filepath = os.path.join(path, file)
    tarpath = os.path.join(path, f'{file.split(".")[0]}.tar.gz')
    print(filepath)
    print(tarpath)
    # running a terminal command to tar the file
    os.system(f"tar -czvf {tarpath} {filepath}")

path = "/Users/mraoaakash/Documents/research/research-nisha/ORCHID_data/ORCHID_100x_data/OSCC/WDOSCC"

for file in os.listdir(path):
    if ".DS_Store" in file:
        continue
    filepath = os.path.join(path, file)
    tarpath = os.path.join(path, f'{file.split(".")[0]}.tar.gz')
    print(filepath)
    print(tarpath)
    # running a terminal command to tar the file
    os.system(f"tar -czvf {tarpath} {filepath}")

path = "/Users/mraoaakash/Documents/research/research-nisha/ORCHID_data/ORCHID_100x_data/Normal"

for file in os.listdir(path):
    if ".DS_Store" in file:
        continue
    filepath = os.path.join(path, file)
    tarpath = os.path.join(path, f'{file.split(".")[0]}.tar.gz')
    print(filepath)
    print(tarpath)
    # running a terminal command to tar the file
    os.system(f"tar -czvf {tarpath} {filepath}")

path = "/Users/mraoaakash/Documents/research/research-nisha/ORCHID_data/ORCHID_100x_data/OSMF"

for file in os.listdir(path):
    if ".DS_Store" in file:
        continue
    filepath = os.path.join(path, file)
    tarpath = os.path.join(path, f'{file.split(".")[0]}.tar.gz')
    print(filepath)
    print(tarpath)
    # running a terminal command to tar the file
    os.system(f"tar -czvf {tarpath} {filepath}")
