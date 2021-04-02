import tensorflow as tf
import config
import os

def get_base_model():
    return tf.keras.applications.Xception(input_shape=config.IMG_SHAPE,
                                          include_top=False,
                                          weights='imagenet')

# Create the CNN-architecture.
def get_cnn_model(fine_tuning:bool):
    base_model = get_base_model()
    print("Number of Layers in Base Model: ",len(base_model.layers))
    if fine_tuning == False:
        base_model.trainable = False
    else:
        for layer in base_model.layers[:-26]:
            layer.trainable = False
        for layer in base_model.layers[-26:]:
            layer.trainable = True
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    dense = tf.keras.layers.Dense(64, activation='relu')
    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        dense,
        prediction_layer
    ])
    # compile model
    model.compile(loss=tf.losses.BinaryCrossentropy(),
                  optimizer=tf.optimizers.Adam(lr=config.BASE_LEARNING_RATE),
                  metrics=[tf.metrics.BinaryAccuracy(threshold=config.SIGMOID_THRESHOLD, name='accuracy')])
    return model

def search_model(model_path):
    return os.path.isdir(model_path)