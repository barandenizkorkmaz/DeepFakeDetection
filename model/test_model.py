import pathlib
import tensorflow as tf
import numpy as np

import config
import model_utils
from utils import data_handler, evaluation_utils

# Import Training Data
print("Importing the Test Dataset...")
test_data_dir = pathlib.Path(config.PATH_TEST)
raw_test = data_handler.import_data(test_data_dir,config.test_data_config)

# Format the Dataset
print("Formatting the Test Dataset...")
test = raw_test.map(data_handler.format_example)
del raw_test

# Get Actual Class Labels
y_actual = list()
for image,label in test:
    current_labels = label.numpy()
    y_actual.extend(current_labels)

# Import Previously Trained CNN.
cnn = tf.keras.models.load_model(config.PATH_MODEL)

# Predict
print("Predicting for Test Dataset...")
class_probabilities = cnn.predict(test)
#class_probabilities = tf.nn.sigmoid(class_probabilities) # Not required since the output layer gives the sigmoid value.
y_predicted = tf.where(class_probabilities < config.SIGMOID_THRESHOLD, 0, 1)

# Output Formatting
class_probabilities = [item for sublist in class_probabilities.numpy() for item in sublist]
y_predicted = [item for sublist in y_predicted.numpy() for item in sublist]

# Accuracy
evaluation_utils.plot_confusion_matrix(y_actual, y_predicted)
evaluation_utils.evaluation(y_actual, y_predicted)

# Predictions
print("Predicted Labels: ",end='')
print(y_predicted)
print("Actual Labels: ",end='')
print(y_actual)