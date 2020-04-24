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
from shutil import copyfile
from itertools import product
import datetime

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
    logger.info("Loading experiment yaml file")
    # Read YAML experiment definition file
    with open(yaml_filepath, "r") as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)

    logger.info(cfg)

    return cfg


def get_scenario_params_list(config):
    """
    Create parameter list for each scenario from the config.
    
    Parameters
    ----------
    config : dict
        Dictionary of parameters for experiement

    Returns
    -------
    scenario_params_list : list
        list of parameters for each scenario.

    """

    scenario_params_list = []
    
    for list_scenario in config:
        params_name = list_scenario.keys()
        params_list = list(list_scenario.values())
        
        for el in product(*params_list):
            scenario = dict(zip(params_name, el))
            
            if scenario['partners_count'] != len(scenario['amounts_per_partner']):
                raise Exception("Length of amounts_per_node does not match number of partners.")
                
            if 'corrupted_datasets' in params_name:
                if scenario['partners_count'] != len(scenario['corrupted_datasets']):
                    raise Exception("Length of corrupted_datasets does not match number of partners.")

            scenario_params_list.append(scenario)
            
    print('\nNumber of scenario(s) configured:', len(scenario_params_list))
    return scenario_params_list


def init_result_folder(yaml_filepath, cfg):
    """
    Init the result folder.

    Args:
        yaml_filepath : str

    Returns:
        folder_name
    """

    logger.info("Init result folder")

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%Hh%M")

    full_experiment_name = cfg["experiment_name"] + "_" + now_str
    experiment_path = Path.cwd() / "experiments" / full_experiment_name

    # Check if experiment folder already exists
    while experiment_path.exists():
        logger.warning(f"Experiment folder, {experiment_path} already exists")
        new_experiment_name = Path(str(experiment_path) + "_bis")
        experiment_path = Path.cwd() / "experiments" / new_experiment_name
        logger.warning(f"Experiment folder has been renamed to: {experiment_path}")

    experiment_path.mkdir(parents=True, exist_ok=False)

    cfg["experiment_path"] = experiment_path
    logger.info("experiment folder " + str(experiment_path) + " created.")

    target_yaml_filepath = experiment_path / Path(yaml_filepath).name
    copyfile(yaml_filepath, target_yaml_filepath)

    logger.info("Result folder initiated")
    return cfg
