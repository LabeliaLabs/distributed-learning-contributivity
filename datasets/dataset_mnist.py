# -*- coding: utf-8 -*-
"""
MNIST dataset.
(inspired from: https://keras.io/examples/mnist_cnn/)
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras

# Init dataset-specific variables
img_rows = 28
img_cols = 28
input_shape = (img_rows, img_cols, 1)
num_classes = 10


# Data samples pre-processing method for inputs
def preprocess_dataset_inputs(x):

    x = x.reshape(x.shape[0], img_rows, img_cols, 1)
    x = x.astype("float32")
    x /= 255

    return x


# Data samples pre-processing method for labels
def preprocess_dataset_labels(y):

    y = keras.utils.to_categorical(y, num_classes)

    return y


# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Pre-process inputs
x_train = preprocess_dataset_inputs(x_train)
x_test = preprocess_dataset_inputs(x_test)


# Model structure and generation
def generate_new_model_for_dataset():
    """Return a CNN model from scratch based on given batch_size"""

    model = Sequential()
    model.add(Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=input_shape,
        ))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model
