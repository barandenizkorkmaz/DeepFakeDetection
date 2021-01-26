import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import config
from model import model_utils

args = ' '.join(sys.argv[1:])

model_found = model_utils.search_model(config.PATH_MODEL)
command_training = "python3 {} {}".format(config.PATH_TRAINING_SCRIPT,args)
command_test = "python3 {} {}".format(config.PATH_TEST_SCRIPT,args)

if (not model_found):
    print("Running: {}".format(command_training))
    os.system(command=command_training)

print("Running: {}".format(command_test))
os.system(command=command_test)