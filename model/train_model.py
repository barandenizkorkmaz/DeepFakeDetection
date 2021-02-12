import pathlib

import config
from utils import data_handler, evaluation_utils
import model_utils

# Import Training Data
data_dir = pathlib.Path(config.PATH_TRAINING)
print("Importing the Dataset...")
raw_train = data_handler.import_data(data_dir,config.raw_data_config,"training")
raw_validation = data_handler.import_data(data_dir,config.raw_data_config,"validation")

# Format the Dataset
print("Formatting the Dataset...")
train = raw_train.map(data_handler.format_example)
validation = raw_validation.map(data_handler.format_example)
del raw_train, raw_validation

# Create CNN Model
cnn_base = model_utils.get_cnn_model(fine_tuning=False)

# Initial Model Status
loss0,accuracy0 = cnn_base.evaluate(validation)
print("Initial Loss = {:.2f}, Initial Accuracy = {:.2f}".format(loss0,accuracy0))

# Train Model
print("Initial Epochs:",config.INITIAL_EPOCHS)
history = cnn_base.fit(x=train, epochs=config.INITIAL_EPOCHS, validation_data=validation)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
evaluation_utils.accuracy_plot((acc, loss), (val_acc, val_loss))

cnn_fine_tuning = model_utils.get_cnn_model(fine_tuning=True)

# Initial Model Status
loss0,accuracy0 = cnn_fine_tuning.evaluate(validation)
print("Initial Loss = {:.2f}, Initial Accuracy = {:.2f}".format(loss0,accuracy0))

history_fine = cnn_fine_tuning.fit(x=train,
                         epochs=config.TOTAL_EPOCHS,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation)

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']
loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']
evaluation_utils.accuracy_plot((acc, loss), (val_acc, val_loss))

# Save CNN Model
cnn_base.save(config.PATH_BASE_MODEL)
cnn_fine_tuning.save(config.PATH_FINE_TUNED_MODEL)