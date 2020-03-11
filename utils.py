# -*- coding: utf-8 -*-
"""
Some utils functions.
(inspired from: https://keras.io/examples/mnist_cnn/)
"""

from __future__ import print_function
import yaml
import os
from pathlib import Path
from loguru import logger


import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

import constants


def generate_new_cnn_model():
    """Return a CNN model from scratch based on given batch_size"""

    model = Sequential()
    model.add(
        Conv2D(
            32, kernel_size=(3, 3), activation="relu", input_shape=constants.INPUT_SHAPE
        )
    )
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(constants.NUM_CLASSES, activation="softmax"))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


def preprocess_input(x):
    """Return intput data so it works with default keras CNN"""

    x = x.reshape(x.shape[0], constants.IMG_ROWS, constants.IMG_COLS, 1)
    x = x.astype("float32")
    x /= 255

    return x


def load_cfg(yaml_filepath):
    """
    Load a YAML configuration file.

    Args:
        yaml_filepath : str

    Returns:
        cfg : dict
    """
    print("*** load_cfg ***")
    # Read YAML experiment definition file
    with open(yaml_filepath, "r") as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)

    print(cfg)

    return cfg


def init_result_folder(yaml_filepath, cfg):
    """
    Init the result folder.

    Args:
        yaml_filepath : str

    Returns:
        folder_name
    """

    logger.info("*** init_result_folder ***")

    experiment_path = Path.cwd() / 'experiments' / cfg["experiment_name"]
    experiment_path.mkdir(parents=True, exist_ok=True)

        
    cfg["experiment_path"] = experiment_path
    logger.info("experiment folder " + str(experiment_path)  + " created.")

    # TODO refactor
    os.system("cp " + yaml_filepath + ' "' + str(experiment_path) + '"')

    logger.info("result folder created")
    return cfg
