# -*- coding: utf-8 -*-
"""
MNIST dataset.
(inspired from: https://keras.io/examples/mnist_cnn/)
"""
from time import sleep
from urllib.error import HTTPError, URLError

from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.utils import to_categorical
from loguru import logger
from sklearn.model_selection import train_test_split

from . import dataset
from .. import constants

# Init dataset-specific variables
img_rows = 28
img_cols = 28
input_shape = (img_rows, img_cols, 1)
num_classes = 10


def generate_new_dataset():
    # Load data
    attempts = 0
    while True:
        try:
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            break
        except (HTTPError, URLError) as e:
            if hasattr(e, 'code'):
                temp = e.code
            else:
                temp = e.errno
            logger.debug(
                f'URL fetch failure on '
                f'https://s3.amazonaws.com/img-datasets/mnist.npz : '
                f'{temp} -- {e.reason}')
            if attempts < constants.NUMBER_OF_DOWNLOAD_ATTEMPTS:
                sleep(2)
                attempts += 1
            else:
                raise

    # Pre-process inputs
    x_train = preprocess_dataset_inputs(x_train)
    x_test = preprocess_dataset_inputs(x_test)

    dataset_obj = dataset.Dataset(
        "mnist",
        x_train,
        x_test,
        y_train,
        y_test,
        input_shape,
        num_classes,
        preprocess_dataset_labels,
        generate_new_model_for_dataset,
        train_val_split_global,
        train_test_split_local,
        train_val_split_local
    )
    return dataset_obj


# Data samples pre-processing method for inputs
def preprocess_dataset_inputs(x):
    x = x.reshape(x.shape[0], img_rows, img_cols, 1)
    x = x.astype("float32")
    x /= 255

    return x


# Data samples pre-processing method for labels
def preprocess_dataset_labels(y):
    y = to_categorical(y, num_classes)

    return y


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
        loss=categorical_crossentropy,
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


# train, test, val splits

def train_test_split_local(x, y):
    return train_test_split(x, y, test_size=0.1, random_state=42)


def train_val_split_local(x, y):
    return train_test_split(x, y, test_size=0.1, random_state=42)


def train_val_split_global(x, y):
    return train_test_split(x, y, test_size=0.1, random_state=42)


def train_test_split_global():
    # The split is already done when importing the dataset
    return None
