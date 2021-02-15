import pathlib

import config
from utils import data_handler, evaluation_utils
import model_utils

# Import Training Data
data_dir = pathlib.Path(config.PATH_TRAINING)
print("Importing the Dataset...")
raw_train = data_handler.import_data(data_dir,config.raw_data_config,"training")
raw_validation = data_handler.import_data(data_dir,config.raw_data_config,"validation")

# Import Training Data
print("Importing the Test Dataset...")
test_data_dir = pathlib.Path(config.PATH_TEST)
raw_test = data_handler.import_data(test_data_dir,config.test_data_config,subset=None)

# Get the number of videos and labels..
train_y = data_handler.get_labels(raw_train)
validation_y = data_handler.get_labels(raw_validation)
test_y = data_handler.get_labels(raw_test)

# Format the Dataset
print("Formatting the Dataset...")
train = raw_train.map(data_handler.format_example)
validation = raw_validation.map(data_handler.format_example)
del raw_train, raw_validation

# Format the Dataset
print("Formatting the Test Dataset...")
test = raw_test.map(data_handler.format_example)
del raw_test

# Create CNN Model
cnn_base = model_utils.get_cnn_model()

# Import ArcFace
arc_face = model_utils.get_base_model()
print("Extracting features for Training Dataset by ArcFace...")
training_features = arc_face.predict(train)
#print("Extracting features for Validation Dataset by ArcFace...")
#validation_features = arc_face.predict(validation)

# Initial Model Status
loss0,accuracy0 = cnn_base.evaluate(x=training_features,y=train_y, batch_size=config.CNN_BATCH_SIZE)
print("Initial Loss = {:.2f}, Initial Accuracy = {:.2f}".format(loss0,accuracy0))

print("Extracting features for Test Dataset by ArcFace...")
test_features = arc_face.predict(test)

# Train Model
print("Number of Epochs:",config.TOTAL_EPOCHS)
history = cnn_base.fit(x=training_features, y=train_y, batch_size= config.CNN_BATCH_SIZE, epochs=config.TOTAL_EPOCHS, validation_data=(test_features,test_y))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
evaluation_utils.accuracy_plot((acc, loss), (val_acc, val_loss))

# Save CNN Model
cnn_base.save(config.PATH_BASE_MODEL)