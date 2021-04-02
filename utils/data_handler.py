import tensorflow as tf
import numpy as np
import os
import pathlib
import csv

import config

# Imports the dataset.
def import_data(directory:str,config:dict):
  """
  Imports the data according to given configurations.
  Arguments:
    directory: Directory where the data is located.
        If `labels` is "inferred", it should contain
        subdirectories, each containing images for a class.
        Otherwise, the directory structure is ignored.
    config: [Dict] Previously determined data configurations.
    Example:
        config = {
          'Batch Size':64,
          'Height':160,
          'Width':160,
          'Channel':3,
          'Shuffle':True,
          'Seed':123,
          'Validation Split':0.2
        }
    subset: The title of subset created.
        "training": Training Data
        "validation": Validation Data
        None: Test Data
  """
  return tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    batch_size=config['Batch Size'],
    image_size=(config['Height'], config['Width']),
    shuffle=config['Shuffle'],
    seed=config['Seed'],
    validation_split=config['Validation Split'],
    subset=config['Subset']
  )

# Formats a dataset instance.
def format_example(image, label):
  """ Rescales the dataset into [-1,1]. The new configuration of data must
  be predetermined.
  """
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (config.IMG_SHAPE[0], config.IMG_SHAPE[1]))
  return image, label

def get_labels(data):
  y_actual = list()
  for image, label in data:
    current_labels = label.numpy()
    y_actual.extend(current_labels)
  return y_actual