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

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab

from . import constants

# Graphics configuration
sns.set_style("white")
pylab.rcParams.update({'font.size': 18})


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
        if isinstance(list_scenario['dataset'], dict):
            for dataset_name in list_scenario['dataset'].keys():
                # Add path to init model from an existing model
                dataset_scenario = list_scenario.copy()
                dataset_scenario['dataset'] = [dataset_name]
                if list_scenario['dataset'][dataset_name] is None:
                    dataset_scenario['init_model_from'] = ['random_initialization']
                else:
                    dataset_scenario['init_model_from'] = list_scenario['dataset'][dataset_name]
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
            if scenario['samples_split_option'][0] in ['advanced', 'flexible']:
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
    logger.add(sys.stdout,
               enqueue=True,
               filter=log_filter,
               format='<blue>{time:YYYY-MM-DD HH:mm:ss}</> | {level: <7} | {message}',
               level=0)
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


def create_data_distribution_graph(name, ratio_partner_per_class,
                                   corrupted_partners={}, save_folder=None, display=False):
    """
    Create a graph of the partners' labels distribution per class

    :param name: str, title of the graph and of the save file
    :param ratio_partner_per_class: numpy array of shape (number of classes, number of partners),
                                    ratio of the number of labels of each class per partner
    :param corrupted_partners: dic (default is {} for no corruption), dictionary with the index of corrupted partners
                                as key and the proportion of corruption as value
    :param save_folder: pathlib Path (default is None, which implies no save of the figure), folder to save the figure
    :param display: boolean (default is False), True to show the plot, False otherwise
    """
    if save_folder is None and not display:
        warnings.warn(
            "utils.create_data_distribution_graph() was called without setting the save_folder or the display option")

    num_classes = len(ratio_partner_per_class)
    num_partners = len(ratio_partner_per_class[0])

    # Configure the plot
    palette = sns.color_palette(palette="Set2", n_colors=num_partners)
    hatches = {0.25: '.', 0.5: '..', 0.75: '...', 1: 'xx'}
    hatch_color = 'darkred'
    fig, ax = plt.subplots(figsize=(2.4 * num_classes, 1.33 * num_partners))

    # Create a dataframe containing the partners' data distribution for each class and plot it
    for i in range(num_classes - 1, -1, -1):
        df = pd.DataFrame(data=ratio_partner_per_class[i, :], columns=['partner_part'])\
               .assign(before_part=lambda df: df['partner_part'].cumsum().shift(fill_value=0))\
               .assign(after_part=lambda df: 1 - df['partner_part'].cumsum())\
               .reindex(['before_part', 'partner_part', 'after_part'], axis=1)\
               .cumsum(axis=1).add(i, axis=1)\
               .rename_axis('partner_num').reset_index()\
               .melt(id_vars=['partner_num'], value_vars=['partner_part', 'before_part', 'after_part'],
                     var_name='chunk', value_name='value')
        for part, cols in zip(["after_part", "partner_part", "before_part"],
                              [['white'] * num_partners, palette, ['white'] * num_partners]):
            bars = sns.barplot(data=df.query('chunk=="' + part + '"'), y='partner_num', x='value',
                               orient='h', palette=cols, ci=None, edgecolor=hatch_color, linewidth=0, )

    # Add corruption hatches
    corr_labels = [''] * num_partners
    for corr, ratio in corrupted_partners.items():
        selection = [c * num_partners * 3 + 1 * num_partners + corr for c in range(num_classes)]
        keys = np.array(list(hatches.keys()))
        hatch = hatches[keys[np.argmin(np.abs(keys - ratio))]]
        for s in selection:
            bars.patches[s].set_hatch(hatch)
        corr_labels[corr] += f'\n({int(ratio * 100)}% corrupted data)'

    # Draw frames around classes
    min_lim = 0.395
    max_lim = 0.39
    for p in range(num_partners):
        plt.hlines(p - min_lim, 0, num_classes, colors='black')
        plt.hlines(p + max_lim, 0, num_classes, colors='black')
        plt.vlines(0, p - min_lim, p + max_lim, colors='black')
        for c in range(num_classes - 1):
            plt.vlines(c + 1, p - min_lim, p + max_lim, colors='black')
        plt.vlines(num_classes, p - min_lim, p + max_lim, colors='black')

    # Rename labels
    plt.xticks(np.arange(0.5, num_classes + 0.5, 1), labels=np.arange(num_classes))
    plt.xlabel('Class')
    y_labels = [f'Partner {i + 1}' for i in range(num_partners)]
    plt.yticks(np.arange(num_partners), labels=[lab + c for lab, c in zip(y_labels, corr_labels)])
    plt.ylabel('')
    plt.title(name)

    # Delete the frame around the plot
    for pos in ['top', 'right', 'bottom', 'left']:
        ax.spines[pos].set_visible(False)

    # Save and/or display fig
    plt.tight_layout()
    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_folder / (name + ".png"))
    if display:
        plt.show()
    plt.close()


def project_onto_the_simplex(v, s=1):
    """
    Project a vector v onto a simplex of radius s.
    The implementation is adapted to be used with numpy from :
    https://github.com/MLOPTPSU/FedTorch/blob
    /ab8068dbc96804a5c1a8b898fd115175cfebfe75/fedtorch/comms/utils/flow_utils.py#L52.
    This algorithm is based on this paper : https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf

    :param v : 1-D numpy array of size n
    :param s : the radius of the simplex on which v will be projected
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape
    # check if we are already on the simplex
    if np.sum(v) == s and np.all((v >= 0)):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted v (in descending order)
    u = np.flip(np.sort(v))
    cssv = np.cumsum(u)

    # get the optimal solution vector
    opt = u * np.arange(1, n+1) > (cssv - s)

    # get the number of non-zero elements of the optimal solution
    non_zero_vector = np.nonzero(opt)
    if len(non_zero_vector) == 0:
        rho = 0.0
    else:
        rho = np.squeeze(non_zero_vector[0][-1])

    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)

    # compute the projection by thresholding v using theta
    w = np.clip(v - theta, a_min=0, a_max=None)
    return w
