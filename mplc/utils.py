# -*- coding: utf-8 -*-
"""
Some utils functions.
"""
from __future__ import print_function

import argparse
import sys
from itertools import product

import tensorflow as tf
from loguru import logger
from ruamel.yaml import YAML

from . import constants


def load_cfg(yaml_filepath):
    """
    Load a YAML configuration file.

    Args:
        yaml_filepath : str

    Returns:
        cfg : dict
    """
    logger.info("Loading experiment yaml file")

    yaml = YAML(typ='safe')
    with open(yaml_filepath, "r") as stream:
        # This will fail if there are duplicated keys in the YAML file
        cfg = yaml.load(stream)
    logger.info(cfg)
    return cfg


def get_scenario_params_list(config):
    """
    Create parameter list for each scenario from the config.

    Parameters
    ----------
    config : dict
        Dictionary of parameters for experiment

    Returns
    -------
    scenario_params_list : list
        list of parameters for each scenario.

    """

    scenario_params_list = []
    # Separate scenarios from different dataset
    config_dataset = []

    for list_scenario in config:
        if isinstance(list_scenario['dataset_name'], dict):
            for dataset_name in list_scenario['dataset_name'].keys():
                # Add path to init model from an existing model
                dataset_scenario = list_scenario.copy()
                dataset_scenario['dataset_name'] = [dataset_name]
                if list_scenario['dataset_name'][dataset_name] is None:
                    dataset_scenario['init_model_from'] = ['random_initialization']
                else:
                    dataset_scenario['init_model_from'] = list_scenario['dataset_name'][dataset_name]
                config_dataset.append(dataset_scenario)
        else:
            config_dataset.append(list_scenario)

    for list_scenario in config_dataset:
        params_name = list_scenario.keys()
        params_list = list(list_scenario.values())
        for el in product(*params_list):
            scenario = dict(zip(params_name, el))
            if scenario['partners_count'] != len(scenario['amounts_per_partner']):
                raise Exception("Length of amounts_per_partner does not match number of partners.")
            if scenario['samples_split_option'][0] == 'advanced':
                if scenario['partners_count'] != len(scenario['samples_split_option'][1]):
                    raise Exception("Length of samples_split_option does not match number of partners.")
                else:
                    scenario['samples_split_configuration'] = scenario['samples_split_option'][1]
                    scenario['samples_split_option'] = scenario['samples_split_option'][0]
            if 'corruption_parameters' in params_name:
                if scenario['partners_count'] != len(scenario['corruption_parameters']):
                    raise Exception("Length of corruption_parameters does not match number of partners.")
            scenario_params_list.append(scenario)

    logger.info(f"Number of scenario(s) configured: {len(scenario_params_list)}")
    return scenario_params_list


def init_gpu_config():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        logger.info(f"Found GPU: {gpus[0].name}")
        try:  # catch error when used on virtual devices
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except ValueError as e:
            logger.warning(str(e))
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=constants.GPU_MEMORY_LIMIT_MB)]
        )
    else:
        logger.info("No GPU found")


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="input config file")
    parser.add_argument("-v", "--verbose", help="verbose output", action="store_true")
    args = parser.parse_args()

    return args


def init_logger(debug=False):
    logger.remove()
    global log_filter
    log_filter = MyFilter("INFO")
    logger.opt(depth=1)
    # Forward logging to standard output
    logger.add(sys.stdout, enqueue=True, filter=log_filter, level=0)
    if debug:
        log_filter.set_to_debug_level()
    else:
        log_filter.set_to_info_level()


class MyFilter:

    def __init__(self, level):
        self.level = level

    def set_to_debug_level(self):
        self.level = "DEBUG"

    def set_to_info_level(self):
        self.level = "INFO"

    def __call__(self, record):
        levelno = logger.level(self.level).no
        return record["level"].no >= levelno


def set_log_file(path):
    logger.remove()
    logger.add(sys.stdout, enqueue=True, filter=log_filter, level=0)
    info_path = path / constants.INFO_LOGGING_FILE_NAME
    debug_path = path / constants.DEBUG_LOGGING_FILE_NAME
    logger.add(info_path, level="INFO")
    logger.add(debug_path, level="DEBUG")
