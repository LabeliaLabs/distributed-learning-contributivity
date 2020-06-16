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
    # Separate scenarios from different dataset
    config_dataset = []

    for list_scenario in config:
        
        if isinstance(list_scenario['dataset_name'], dict):
            for dataset_name in list_scenario['dataset_name'].keys():
                # Add path to init model from an existing model
                dataset_scenario = list_scenario.copy()
                dataset_scenario['dataset_name'] = [dataset_name]
                dataset_scenario['init_model_from'] = list_scenario['dataset_name'][dataset_name]
                
                config_dataset.append(dataset_scenario)
            else:
                config_dataset.append(list_scenario)
    
    for list_scenario in config_dataset:
        params_name = list_scenario.keys()
        params_list = list(list_scenario.values())
        
<<<<<<< HEAD
                params_name = dataset_scenario.keys()
                params_list = list(dataset_scenario.values())
        else:
            params_name = list_scenario.keys()
            params_list = list(list_scenario.values())

=======
        #Òprint(list_scenario)
>>>>>>> Change way to build scenarios
        for el in product(*params_list):
            scenario = dict(zip(params_name, el))
        
            if scenario['partners_count'] != len(scenario['amounts_per_partner']):
                raise Exception("Length of amounts_per_partner does not match number of partners.")

            if isinstance(scenario['samples_split_option'], list) \
                    and (scenario['partners_count'] != len(scenario['samples_split_option'])):
                raise Exception("Length of samples_split_option does not match number of partners.")
        
            if 'corrupted_datasets' in params_name:
                if scenario['partners_count'] != len(scenario['corrupted_datasets']):
                    raise Exception("Length of corrupted_datasets does not match number of partners.")
            
            # keep only init model from if it is a path. 
            try:
                if scenario['init_model_from'] == None:
                    del scenario['init_model_from']
            except KeyError:
                pass
        
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
