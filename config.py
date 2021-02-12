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
PATH_TEST = os.path.join(userArguments['-d'],'Test')

NUM_FRAMES_PER_VIDEO = 50
TRAINING_BATCH_SIZE = NUM_FRAMES_PER_VIDEO
TEST_BATCH_SIZE = NUM_FRAMES_PER_VIDEO

RAW_IMAGE_HEIGHT = 224
RAW_IMAGE_WIDTH = 224
RAW_IMAGE_CHANNELS = 3

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
TOTAL_EPOCHS = int(userArguments['-e']) if '-e' in userArguments else int(20)
INITIAL_EPOCHS = int(TOTAL_EPOCHS / 2)
FINE_TUNING_EPOCHS = TOTAL_EPOCHS - INITIAL_EPOCHS
CNN_BATCH_SIZE = int(userArguments['-b']) if '-b' in userArguments else 32

"""
Data Configurations - 2
"""
raw_data_config = {
    'Batch Size':CNN_BATCH_SIZE,
    'Height':RAW_IMAGE_HEIGHT,
    'Width':RAW_IMAGE_WIDTH,
    'Channel':RAW_IMAGE_CHANNELS,
    'Shuffle':True,
    'Seed':123,
    'Validation Split':0.2
}

training_data_config = {
    'Batch Size':CNN_BATCH_SIZE,
    'Height':IMAGE_HEIGHT,
    'Width':IMAGE_WIDTH,
    'Channel':IMAGE_CHANNELS,
    'Shuffle':True,
    'Seed':123,
    'Validation Split':0.2
}

test_data_config = {
    'Batch Size':CNN_BATCH_SIZE,
    'Height':IMAGE_HEIGHT,
    'Width':IMAGE_WIDTH,
    'Channel':IMAGE_CHANNELS,
    'Shuffle':False,
    'Seed':None,
    'Validation Split':None
}

IMG_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

"""
Evaluation Configurations
"""
SIGMOID_THRESHOLD = 0.5

"""
CONSTANTS
"""
PATH_BASE_MODEL = os.path.join(os.getcwd(), "DeepfakeDetection_BaseModel")
PATH_FINE_TUNED_MODEL = os.path.join(os.getcwd(), "DeepfakeDetection_FineTunedModel")
PATH_TRAINING_SCRIPT = os.path.join(os.getcwd(),"model/train_model.py")
PATH_TEST_SCRIPT = os.path.join(os.getcwd(),"model/test_model.py")

"""
PLOT CONSTANTS
"""
ACCURACY_PLOT_ID = 1
CONFUSION_MATRIX_PLOT_ID = 1