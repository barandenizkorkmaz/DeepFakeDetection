import os
import sys

def getUserArguments(args):
    userArguments = dict()
    for i, arg in enumerate(args):
        if arg[0] == '-':
            userArguments[arg] = args[i + 1]
    return userArguments

"""
User Arguments
"""
userArguments = getUserArguments(sys.argv[1:])

COMPRESSION = {
    'lq' : 'c40',
    'hq' : 'c23',
    'raw': 'raw'
}

compression_rate = COMPRESSION[userArguments['-q'].lower()]

current_path = os.getcwd()
ALL_DATASETS = ['original', 'DeepFakeDetection_original', 'Deepfakes',
                'DeepFakeDetection', 'Face2Face', 'FaceShifter', 'FaceSwap',
                'NeuralTextures']

selected_datasets = ['original','Deepfakes','Face2Face','FaceShifter','FaceSwap','NeuralTextures']

"""
Dataset Options:
{original_youtube_videos,original_youtube_videos_info,original,DeepFakeDetection_original,Deepfakes,DeepFakeDetection,Face2Face,FaceShifter,FaceSwap,NeuralTextures,all}
"""

for dataset in selected_datasets:
    command = "python3 faceforensics_download_v4.py {} -d {} -c {} -t videos".format(current_path,dataset,compression_rate)
    print(command)
    os.system(command)

#EXAMPLES:
#os.system('python3 faceforensics_download_v4.py ~/PycharmProjects/FaceForensics++/ -d original -c raw -t videos -n 4')
#os.system('python3 faceforensics_download_v4.py ~/PycharmProjects/FaceForensics++/ -d Deepfakes -c raw -t videos -n 4')
#os.system('python3 faceforensics_download_v4.py ~/PycharmProjects/FaceForensics++/ -d FaceSwap -c raw -t videos -n 4')
#os.system('python3 faceforensics_download_v4.py ~/PycharmProjects/FaceForensics++/ -d Face2Face -c raw -t videos -n 4')