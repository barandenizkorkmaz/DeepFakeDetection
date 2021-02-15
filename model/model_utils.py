import tensorflow as tf
from deepface import DeepFace
import config
import os

def get_base_model():
        return DeepFace.ArcFace.loadModel()

# Create the CNN-architecture.
def get_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(config.DIM_FEATURES,)),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])
    # compile model
    model.compile(loss=tf.losses.BinaryCrossentropy(),
                  optimizer=tf.optimizers.Adam(lr=config.BASE_LEARNING_RATE),
                  metrics=[tf.metrics.BinaryAccuracy(threshold=config.SIGMOID_THRESHOLD, name='accuracy')])
    print(model.summary())
    return model

def search_model(model_path):
    return os.path.isdir(model_path)