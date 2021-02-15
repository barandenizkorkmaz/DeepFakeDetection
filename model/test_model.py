import pathlib
import tensorflow as tf
import config
from utils import data_handler, evaluation_utils
import model_utils

# Import Training Data
print("Importing the Test Dataset...")
test_data_dir = pathlib.Path(config.PATH_TEST)
raw_test = data_handler.import_data(test_data_dir,config.test_data_config,subset=None)

# Format the Dataset
print("Formatting the Test Dataset...")
test = raw_test.map(data_handler.format_example)
del raw_test

# Get Actual Class Labels
y_actual = data_handler.get_labels(test)

# Import ArcFace
arc_face = model_utils.get_base_model()
print("Extracting features for Test Dataset by ArcFace...")
test_features = arc_face.predict(test)

# Import Previously Trained CNN.
cnn_base = tf.keras.models.load_model(config.PATH_BASE_MODEL)

# Predict
print("Predicting for Test Dataset...")
history_test = cnn_base.evaluate(x=test_features,y=y_actual)
y_predicted = (cnn_base.predict(test_features) > config.SIGMOID_THRESHOLD).astype("int32")

# Accuracy
evaluation_utils.plot_confusion_matrix(y_actual, y_predicted)
evaluation_utils.evaluation(y_actual, y_predicted)