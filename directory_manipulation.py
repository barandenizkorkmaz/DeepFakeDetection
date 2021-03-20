import sys
import os
import pathlib
from distutils.dir_util import copy_tree

def check_subset_class(path:str):
    path_list = path.split(sep="/")
    subset = path_list[-1]
    if "real" in subset:
        return True
    return False

def get_subset_split(path:str):
    path_list = path.split(sep="/")
    split = path_list[-1]
    return split

def get_split_id(split_name:str) -> int:
    if split_name.lower()[0:2] == "tr":
        return 0
    elif split_name.lower()[0:2] == "va":
        return 1
    else:
        return 2 # test

def get_faceforensics_folder_name(path:str):
    path_list = path.split(sep="/")
    return path_list[-1]

ARGUMENTS = dict()

for i,arg in enumerate(sys.argv):
    if arg[0] == "-":
        ARGUMENTS[arg] = sys.argv[i+1]

SOURCE_DIRECTORY = ARGUMENTS["-src"]
TARGET_DIRECTORY = ARGUMENTS["-dest"]
subsets = ["Training","Validation","Test"]
split_dict = {
    0: "Training",
    1: "Validation",
    2: "Test"
}
classes = ["Real","Fake"]
faceforensics_folders = list()

tmp = str()

if os.path.exists(TARGET_DIRECTORY):
    tmp = input("Target Directory: {} already exists.\nDo you want to overwrite?\n[Y]es | [N]o\n".format(TARGET_DIRECTORY))
    if tmp.lower() == "n":
        raise SystemExit("The program has been terminated!")
    os.system("rm -r {}".format(TARGET_DIRECTORY))

os.system("mkdir {}".format(TARGET_DIRECTORY))

for subset in subsets:
    for label in classes:
        PATH = "{}/{}/{}".format(TARGET_DIRECTORY,subset,label)
        os.makedirs(PATH)

for root,dirs,files in os.walk(SOURCE_DIRECTORY):
    for subdirectory in dirs:
        subdir = os.path.join(root, subdirectory)
        faceforensics_folders.append(subdir)
    break

for faceforensics_folder in faceforensics_folders:
    ff_folder_name = get_faceforensics_folder_name(faceforensics_folder)
    split_dirs = list()
    for root, dirs, files in os.walk(faceforensics_folder):
        for subdirectory in dirs:
            subdir = os.path.join(root, subdirectory)
            split_dirs.append(subdir)
        break
    is_real = check_subset_class(faceforensics_folder)
    class_name = "Real" if is_real else "Fake"
    for split_dir in split_dirs:
        split_name = get_subset_split(split_dir)
        split_id = get_split_id(split_name)
        split_name = split_dict[split_id]
        target_directory_path = "{}/{}/{}".format(TARGET_DIRECTORY,split_name,class_name)
        for root, dirs, files in os.walk(split_dir):
            for file in files:
                src_file_path = "{}/{}".format(split_dir,file)
                target_file_path = "{}/{}_{}".format(target_directory_path,ff_folder_name,file)
                os.system("cp {} {}".format(src_file_path,target_file_path))
            break