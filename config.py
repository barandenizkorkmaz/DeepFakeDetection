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

"""
Data Configurations
"""
NAME_DATA = "CMPE492_Deepfakedetection_Data"
PATH_TRAINING = os.path.join(userArguments['-d'],'Training')
PATH_VALIDATION = os.path.join(userArguments['-d'],'Validation')
PATH_TEST = os.path.join(userArguments['-d'],'Test')

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3

"""
Model Configurations
"""
# CNN MODEL
BASE_LEARNING_RATE = 0.0001
VALIDATION_STEPS = 20
DIM_FEATURES = 512
CNN_EPOCHS = int(userArguments['-e']) if '-e' in userArguments else int(20)
CNN_INITIAL_EPOCHS = int(CNN_EPOCHS/2)
CNN_BATCH_SIZE = int(userArguments['-b']) if '-b' in userArguments else int(32)

"""
Data Configurations - 2
"""
raw_data_config = {
    'Batch Size':CNN_BATCH_SIZE,
    'Height':224,
    'Width':224,
    'Channel':3,
    'Shuffle':True,
    'Seed':123,
    'Validation Split':0.2
}

training_data_config = {
    'Batch Size':CNN_BATCH_SIZE,
    'Height':224,
    'Width':224,
    'Channel':3,
    'Shuffle':True,
    'Seed':123,
    'Validation Split':None,
    'Subset': None
}

validation_data_config = {
    'Batch Size':CNN_BATCH_SIZE,
    'Height':224,
    'Width':224,
    'Channel':3,
    'Shuffle':False,
    'Seed':123,
    'Validation Split':None,
    'Subset': None
}

test_data_config = {
    'Batch Size':CNN_BATCH_SIZE,
    'Height':224,
    'Width':224,
    'Channel':3,
    'Shuffle':False,
    'Seed':None,
    'Validation Split':None,
    'Subset': None
}

IMG_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

"""
Evaluation Configurations
"""
SIGMOID_THRESHOLD = 0.5

"""
CONSTANTS
"""
PATH_BASE_MODEL = os.path.join(os.getcwd(), "DeepfakeDetection_Model_Base")
PATH_FINE_TUNING_MODEL = os.path.join(os.getcwd(), "DeepfakeDetection_Model_FT")
PATH_TRAINING_SCRIPT = os.path.join(os.getcwd(),"model/train_model.py")
PATH_TEST_SCRIPT = os.path.join(os.getcwd(),"model/test_model.py")

"""
PLOT CONSTANTS
"""
ACCURACY_PLOT_ID = 1
CONFUSION_MATRIX_PLOT_ID = 1