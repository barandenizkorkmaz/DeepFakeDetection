import multiprocessing
import sys
import cv2
import os
import pathlib
import numpy as np
from deepface.commons import functions

NAME_DATA = "CMPE492_Deepfakedetection_Data"
NUM_FRAMES = 50

current_directory = os.getcwd()
data_dir = os.path.join(current_directory, NAME_DATA)
root_directory = ['original_sequences','manipulated_sequences']
directories_to_scan = []

def create_dirs():
    os.makedirs(data_dir)
    training_data_path = os.path.join(data_dir, 'Training')
    test_data_path = os.path.join(data_dir, 'Test')
    training_fake_data_path = os.path.join(training_data_path, 'Fake')
    training_real_data_path = os.path.join(training_data_path, 'Real')
    test_fake_data_path = os.path.join(test_data_path, 'Fake')
    test_real_data_path = os.path.join(test_data_path, 'Real')
    os.makedirs(training_data_path)
    os.makedirs(test_data_path)
    os.makedirs(training_fake_data_path)
    os.makedirs(training_real_data_path)
    os.makedirs(test_fake_data_path)
    os.makedirs(test_real_data_path)


def copy_images(source_file_path_template, name_template, indices, isFake, isTraining):
    for index in indices:
        source_file_full_path =  source_file_path_template + str(index) + ".png"
        target_path = os.path.join(data_dir, 'Training') if isTraining else os.path.join(data_dir, 'Test')

        detectors = ['opencv', 'ssd', 'dlib', 'mtcnn']
        aligned_face = functions.preprocess_face(img=source_file_full_path, detector_backend=detectors[3])[0]
        aligned_face = np.multiply(aligned_face, 255)

        if isFake:
            #shutil.copy(source_file_full_path, "{}/Fake".format(target_path))
            target_path = target_path + "/Fake"
        else:
            #shutil.copy(source_file_full_path, "{}/Real".format(target_path))
            target_path = target_path + "/Real"
        cv2.imwrite(os.path.join(target_path, '{}_{}.png'.format(name_template, str(index))), aligned_face)


def get_indices(num_of_frames, count=NUM_FRAMES):
    indices = []
    stride = int(num_of_frames/count)
    index = 1
    while len(indices) != count:
        indices.append(index)
        index = (index+stride) % num_of_frames
    return indices


def format_file_name(file_name):
    index = len(file_name)-1
    for element in file_name[::-1]:
        if element == '_':
            return index
        index = index - 1


def process_directory(dir):
    for root, dirs, files in os.walk(dir):
        for curr_dir in dirs:
            if curr_dir != 'images':
                continue
            images_path = os.path.join(pathlib.Path(root),curr_dir)
            isFake = True if "manipulated_sequences" in images_path else False
            for images_root, images_dir, images_files in os.walk(images_path):
                num_of_videos = len(images_dir)
                cut_off = int(round(num_of_videos*0.8))
                for video_index,sub_dir in enumerate(images_dir):
                    current_frames_dir = os.path.join(images_path, sub_dir)
                    isTraining = True if video_index < cut_off else False
                    process_current_frames_dir(current_frames_dir, isFake, isTraining)
                    #print(current_frames_dir)


def process_current_frames_dir(dir, isFake, isTraining):
    for root, dirs, files in os.walk(dir):
        files_length = len(files)
        if files_length < NUM_FRAMES :
            continue
        name_template = ""
        for file in files:
            print("File: ", file)
            index = format_file_name(file)
            name_template = file[:index+1]
            break
        source_file_path_template = dir + "/" + name_template
        indices = get_indices(files_length)
        copy_images(source_file_path_template, name_template, indices, isFake, isTraining)


for dir in root_directory:
    subdir = os.path.join(current_directory,dir)
    for root,dirs,files in os.walk(os.path.join(current_directory,dir)):
        for subdirectory in dirs:
            next_directory = os.path.join(subdir, subdirectory)
            directories_to_scan.append(next_directory)
        break

if not os.path.exists(data_dir):
    create_dirs()
else:
    answer = input("Already found: {}\nWould you like to overwrite?\n[Y]es | [N]o")

    if answer.lower() == "n":
        sys.exit(0)
    else:
        command = "rm -r {}".format(data_dir)
        os.system(command=command)
        create_dirs()

for dir in directories_to_scan:
    p = multiprocessing.Process(target=process_directory, args=(dir,))
    p.start()