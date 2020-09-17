# -*- coding: utf-8 -*-
"""
CIFAR10 dataset.
"""
from time import sleep
from urllib.error import HTTPError, URLError

from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from loguru import logger
from sklearn.model_selection import train_test_split

from . import dataset
from .. import constants

# Init dataset-specific variables
input_shape = (32, 32, 3)
num_classes = 10


# Data samples pre-processing method for inputs
def preprocess_dataset_inputs(x):
    x = x.astype("float32")
    x /= 255

    return x


# Data samples pre-processing method for labels
def preprocess_dataset_labels(y):
    y = to_categorical(y, num_classes)

    return y


def generate_new_dataset():
    attempts = 0
    while True:
        try:
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            break
        except (HTTPError, URLError) as e:
            if hasattr(e, 'code'):
                temp = e.code
            else:
                temp = e.errno
            logger.debug(
                f'URL fetch failure on '
                f'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz : '
                f'{temp} -- {e.reason}')
            if attempts < constants.NUMBER_OF_DOWNLOAD_ATTEMPTS:
                sleep(2)
                attempts += 1
            else:
                raise

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Pre-process inputs
    x_train = preprocess_dataset_inputs(x_train)
    x_test = preprocess_dataset_inputs(x_test)

    y_train = preprocess_dataset_labels(y_train)
    y_test = preprocess_dataset_labels(y_test)

    dataset_obj = dataset.Dataset(
        "cifar10",
        x_train,
        x_test,
        y_train,
        y_test,
        input_shape,
        num_classes,
        generate_new_model_for_dataset,
        train_val_split_global,
        train_test_split_local,
        train_val_split_local
    )
    return dataset_obj


# Load and flatten data


# Model structure and generation
def generate_new_model_for_dataset():
    """Return a CNN model from scratch based on given batch_size"""

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = RMSprop(learning_rate=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

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
