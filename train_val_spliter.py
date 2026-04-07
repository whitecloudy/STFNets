import numpy as np
from widar_dataset import find_npz_files
import os
import fnmatch

dir_path = "widar_data"
must_have_g=[f'-gesture{i}-' for i in range(4)]+['-gesture17-','-gesture18-']
must_not_have_g=["-user5-",]


def filter_files(file_paths):
    filtered_paths = []
    
    must_have = [must_have_g] if isinstance(must_have_g, str) else must_have_g
    must_not_have = [must_not_have_g] if isinstance(must_not_have_g, str) else must_not_have_g
    for path in file_paths:
        os.path.normpath(path)
        if must_have and not any(fnmatch.fnmatch(path, f'*{feature}*') for feature in must_have):
            continue
        if must_not_have and any(fnmatch.fnmatch(path, f'*{feature}*') for feature in must_not_have):
            continue
        filtered_paths.append(path)
    return filtered_paths

def split_files(file_paths):
    file_paths = sorted(file_paths)
    np.random.seed(42)
    np.random.shuffle(file_paths)
    split_index = int(0.8 * len(file_paths))
    return file_paths[:split_index], file_paths[split_index:]

file_paths = []
if isinstance(dir_path, str):
    # find .npz files in recursive way
    file_paths = find_npz_files(dir_path)
elif isinstance(dir_path, list):
    for path in dir_path:
        file_paths.extend(find_npz_files(path))
else:
    raise ValueError("dir_path should be a string or a list of strings.")

file_paths = filter_files(file_paths)
train_files, val_files = split_files(file_paths)

with open("train_files.txt", "w") as train_file:
    for path in train_files:
        train_file.write(f"{path}\n")

with open("val_files.txt", "w") as val_file:
    for path in val_files:
        val_file.write(f"{path}\n")
        