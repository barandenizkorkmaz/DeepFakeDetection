import pathlib

import config
from utils import data_handler, evaluation_utils
import model_utils

# Import Training Data
training_data_dir = pathlib.Path(config.PATH_TRAINING)
validation_data_dir = pathlib.Path(config.PATH_VALIDATION)
print("Importing the Dataset...")
raw_train = data_handler.import_data(training_data_dir,config.training_data_config)
raw_validation = data_handler.import_data(validation_data_dir,config.validation_data_config)

# Format the Dataset
print("Formatting the Dataset...")
train = raw_train.map(data_handler.format_example)
validation = raw_validation.map(data_handler.format_example)
del raw_train, raw_validation

# Create CNN Model
cnn = model_utils.get_cnn_model()

# Initial Model Status
loss0,accuracy0 = cnn.evaluate(validation)
print("Initial Loss = {:.3f}, Initial Accuracy = {:.3f}".format(loss0,accuracy0))

# Train Model
history = cnn.fit(x=train, epochs=config.CNN_EPOCHS, validation_data=validation)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
evaluation_utils.accuracy_plot((acc, loss), (val_acc, val_loss))

# Save CNN Model
cnn.save(config.PATH_MODEL)