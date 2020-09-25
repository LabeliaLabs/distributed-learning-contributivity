# -*- coding: utf-8 -*-
from time import sleep
from urllib.error import HTTPError, URLError

import numpy as np
from keras.datasets import imdb
from keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.preprocessing import sequence
from loguru import logger
from sklearn.model_selection import train_test_split

from mplc import dataset, constants

num_words = 5000
input_shape = 500
num_classes = 2


def generate_new_dataset():
    attempts = 0
    while True:
        try:
            (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
            break
        except (HTTPError, URLError) as e:
            if hasattr(e, 'code'):
                temp = e.code
            else:
                temp = e.errno
            logger.debug(
                f'URL fetch failure : '
                f'{temp} -- {e.reason}')
            if attempts < constants.NUMBER_OF_DOWNLOAD_ATTEMPTS:
                sleep(2)
                attempts += 1
            else:
                raise
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((x_train, x_test), axis=0),
                                                        np.concatenate((y_train, y_test), axis=0),
                                                        test_size=0.2)

    # Pre-process inputs
    x_train = preprocess_dataset_inputs(x_train)
    x_test = preprocess_dataset_inputs(x_test)

    y_train = preprocess_dataset_labels(y_train)
    y_test = preprocess_dataset_labels(y_test)

    dataset_obj = dataset.Dataset(
        "imdb",
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


def preprocess_dataset_labels(y):
    # vanilla label
    return y


def preprocess_dataset_inputs(x):
    x = sequence.pad_sequences(x, maxlen=input_shape)
    return x


# Model structure and generation
def generate_new_model_for_dataset():
    """ Return a CNN model from scratch based on given batch_size"""

    model = Sequential()
    model.add(Embedding(num_words, 32, input_length=input_shape))
    model.add(Conv1D(filters=32,
                     kernel_size=3,
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


# train, test, val splits

def train_test_split_local(x, y):
    return x, np.array([]), y, np.array([])


def train_val_split_local(x, y):
    return x, np.array([]), y, np.array([])


def train_val_split_global(x, y):
    return train_test_split(x, y, test_size=0.1, random_state=42)


def train_test_split_global(data):
    return train_test_split(data, test_size=0.1, random_state=42)
