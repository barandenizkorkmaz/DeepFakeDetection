import os
import pathlib
import multiprocessing
import random

TARGET_DIRECTORY_NAME = 'FaceForensicsDFD'

root_directory = ['manipulated_sequences'] # Manually configured!
directories_to_scan = list()

def select_videos_from_directory(dir):
    video_list = list()
    print("Processing: {}".format(dir))
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file[-4:] == ".mp4":
                video_path = pathlib.Path(root)
                video_list.append(os.path.join(root,file))
    random.shuffle(video_list)
    video_list = video_list[:int(len(video_list)/5)]
    for i,video in enumerate(video_list):
        video_path_list = video.split(sep='/')
        sequence_title = video_path_list[-5]
        method = video_path_list[-4]
        compressionRate = video_path_list[-3]
        target_path = os.path.join(os.getcwd(),'{}/{}/{}/{}/videos'.format(TARGET_DIRECTORY_NAME,sequence_title,method,compressionRate))
        if i==0:
            os.makedirs(target_path)
        os.system('cp {} {}'.format(video,target_path))

def copy_real_videos_from_directory():
    real_videos_directory = os.path.join(os.getcwd(), 'original_sequences')
    print("Processing: {}".format(real_videos_directory))
    video_list = list()
    for root, dirs, files in os.walk(real_videos_directory):
        for file in files:
            if file[-4:] == ".mp4":
                video_list.append(os.path.join(root, file))
    for i, video in enumerate(video_list):
        video_path_list = video.split(sep='/')
        sequence_title = video_path_list[-5]
        method = video_path_list[-4]
        compressionRate = video_path_list[-3]
        target_path = os.path.join(os.getcwd(),
                                   '{}/{}/{}/{}/videos'.format(TARGET_DIRECTORY_NAME, sequence_title, method,
                                                               compressionRate))
        if i == 0:
            os.makedirs(target_path)
        os.system('cp {} {}'.format(video, target_path))

for dir in root_directory:
    subdir = os.path.join(os.getcwd(),dir)
    for root,dirs,files in os.walk(os.path.join(os.getcwd(),dir)):
        for subdirectory in dirs:
            next_directory = os.path.join(subdir, subdirectory)
            directories_to_scan.append(next_directory)
        break

command = str()
if os.path.exists(os.path.join(os.getcwd(),TARGET_DIRECTORY_NAME)):
    command = input("{} already exists.\nDo you want to overwrite?\n[Y]es | [N]o\n").lower()
    if command == 'n' or command == 'no':
        raise SystemExit("You have terminated the program.")
    elif command == 'y' or command == 'yes':
        os.system('rm -r {}'.format(os.path.join(os.getcwd(),TARGET_DIRECTORY_NAME)))
    else:
        raise SystemExit("Please enter a valid input.")

for dir in directories_to_scan:
    p = multiprocessing.Process(target=select_videos_from_directory,args=(dir,))
    p.start()
copy_real_videos_from_directory()