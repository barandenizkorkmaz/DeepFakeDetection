import pathlib
import tensorflow as tf
import config
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
y_actual = data_handler.get_labels(test)

# Import Previously Trained CNN.
cnn_base = tf.keras.models.load_model(config.PATH_BASE_MODEL)
cnn_fine_tuned = tf.keras.models.load_model(config.PATH_FINE_TUNING_MODEL)

# Predict
print("Predicting for Test Dataset...")
history_test = cnn_base.evaluate(test)
y_predicted = (cnn_base.predict(test) > config.SIGMOID_THRESHOLD).astype("int32")
print("Predictions:")
print(y_predicted)

# Accuracy
evaluation_utils.plot_confusion_matrix(y_actual, y_predicted)
evaluation_utils.evaluation(y_actual, y_predicted)

# Predict
print("Predicting for Test Dataset...")
history_test_fine = cnn_fine_tuned.evaluate(test)
y_predicted_fine_tuning = (cnn_fine_tuned.predict(test) > config.SIGMOID_THRESHOLD).astype("int32")
print("Predictions:")
print(y_predicted_fine_tuning)

# Accuracy
evaluation_utils.plot_confusion_matrix(y_actual, y_predicted_fine_tuning)
evaluation_utils.evaluation(y_actual, y_predicted_fine_tuning)