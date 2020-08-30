# -*- coding: utf-8 -*-
"""
Some utils functions.
"""

from __future__ import print_function
from ruamel.yaml import YAML
from pathlib import Path
from loguru import logger
from shutil import copyfile
from itertools import product
import datetime
import random
import numpy as np

def load_cfg(yaml_filepath):
    """
    Load a YAML configuration file.

    Args:
        yaml_filepath : str

    Returns:
        cfg : dict
    """
    logger.info("Loading experiment yaml file")

    yaml=YAML(typ='safe')
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

    for list_scenario in config:
        params_name = list_scenario.keys()
        params_list = list(list_scenario.values())

        for el in product(*params_list):
            scenario = dict(zip(params_name, el))

            if scenario['partners_count'] != len(scenario['amounts_per_partner']):
                raise Exception("Length of amounts_per_partner does not match number of partners.")

            if scenario['samples_split_option'][0] == 'advanced' \
                    and (scenario['partners_count'] != len(scenario['samples_split_option'][1])):
                raise Exception("Length of samples_split_option does not match number of partners.")

            if 'corrupted_datasets' in params_name:
                if scenario['partners_count'] != len(scenario['corrupted_datasets']):
                    raise Exception("Length of corrupted_datasets does not match number of partners.")

            scenario_params_list.append(scenario)

    logger.info(f"Number of scenario(s) configured: {len(scenario_params_list)}")
    return scenario_params_list


def init_result_folder(yaml_filepath, cfg):
    """
    Init the result folder.

    Args:
        yaml_filepath : str
        cfg

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


"""
function : get_random_index_from_weighted_list
args => weighted_list:list
Take a increasing positive number list as arguments and return a random index of the list 
which probability is weighted:
ex : [0.1,O.4,0.4,0.1] can be cumulate as l = [0.1,0.5,0.9,1]
we then generate a random number r in range [0:1],
and return the first row that satisfy  r<l[i]
"""
def get_random_index_from_weighted_list(weighted_list):

    # Check if list satisfy cumulative constraint
    # 
    for i in range(1,len(weighted_list)):

        if( weighted_list[i-1] > weighted_list[i]  ):

            return None


    # Check positivity constraint
    for i in range(len(weighted_list)):

        if( weighted_list[i] < 0 ):

            return None
            
    #Genration of random index between 0 and max(weighted_list)
    random_value = random.random()

    for i in range(len(weighted_list)):

        if(weighted_list[i] >= random_value):

            return i

    return None

def distance_vector_numpy(a,b):

    return np.sqrt(np.sum((a-b)**2))

def distance_vector_dictionnary(a,b):

    dist = 0

    for label in a.keys():

        dist += (a[label] - b[label])**2

    return np.sqrt(dist)