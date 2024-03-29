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
CNN_EPOCHS = int(userArguments['-e']) if '-e' in userArguments else 20
CNN_BATCH_SIZE = int(userArguments['-b']) if '-b' in userArguments else 32

#LSTM
NUMBER_OF_UNITS = 1024 # DIMENSIONALITY OF HIDDEN LAYERS
LSTM_EPOCHS = int(userArguments['-e']) if '-e' in userArguments else 50
LSTM_BATCH_SIZE = int(userArguments['-b']) if '-b' in userArguments else 16
LSTM_INPUT_SHAPE = (NUM_FRAMES_PER_VIDEO,DIM_FEATURES) # (timesteps, features)

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
    'Validation Split':0.2
}

test_data_config = {
    'Batch Size':CNN_BATCH_SIZE,
    'Height':224,
    'Width':224,
    'Channel':3,
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
PATH_MODEL = os.path.join(os.getcwd(), "DeepfakeDetection_Model")
PATH_TRAINING_SCRIPT = os.path.join(os.getcwd(),"model/train_model.py")
PATH_TEST_SCRIPT = os.path.join(os.getcwd(),"model/test_model.py")

"""
PLOT CONSTANTS
"""
ACCURACY_PLOT_ID = 1
CONFUSION_MATRIX_PLOT_ID = 1