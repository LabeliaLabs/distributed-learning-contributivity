# -*- coding: utf-8 -*-
"""
A script for:
    - mocking a multi-partner ML project by splitting data among different partners
    - training a model across multiple partners in a distributed approach
    - measuring the respective contributions of each partner to the model performance (termed "contributivity")
"""

from timeit import default_timer as timer
import matplotlib.pyplot as plt
import numpy as np
import utils
from loguru import logger
import tensorflow as tf
import sys
import contextlib
import shutil

import constants
import contributivity
import scenario
import multi_partner_learning

import argparse

DEFAULT_CONFIG_FILE = "config.yml"

@logger.catch
def main():

    stream, info_logger_id, info_debug_id = init_logger()

    with contextlib.redirect_stdout(stream):
        logger.info("Standard output is sent to added handlers.")

        # Parse config file for scenarios to be experimented
        config = get_config_from_file()
        scenario_params_list = utils.get_scenario_params_list(
            config["scenario_params_list"])

        experiment_path = config["experiment_path"]
        n_repeats = config["n_repeats"]

        for scenario_id, scenario_params in enumerate(scenario_params_list):
            logger.info(f"Scenario {scenario_id+1}/{len(scenario_params_list)}: {scenario_params}")

        # Move log files to experiment folder
        move_log_file_to_experiment_folder(
            info_logger_id, experiment_path, constants.INFO_LOGGING_FILE_NAME
        )
        move_log_file_to_experiment_folder(
            info_debug_id, experiment_path, constants.DEBUG_LOGGING_FILE_NAME
        )

        # GPU config
        init_GPU_config()

        # Close open figures
        plt.close("all")

        # Iterate over repeats of all scenarios experiments
        for i in range(n_repeats):

            logger.info(f"Repeat {i+1}/{n_repeats}")

            for scenario_id, scenario_params in enumerate(scenario_params_list):

                logger.info(f"Scenario {scenario_id + 1}/{len(scenario_params_list)}")
                logger.info("Current params:")
                logger.info(scenario_params)

                current_scenario = scenario.Scenario(scenario_params, 
                                                     experiment_path,
                                                     scenario_id=scenario_id+1,
                                                     n_repeat=i+1)

                run_scenario(current_scenario)

                # Write results to CSV file
                df_results = current_scenario.to_dataframe()
                df_results["random_state"] = i
                df_results["scenario_id"] = scenario_id

                with open(experiment_path / "results.csv", "a") as f:
                    df_results.to_csv(f, header=f.tell() == 0, index=False)
                    logger.info("Results saved")

    return 0


def init_logger():
    logger.remove()
    # Forward all logging to standard output
    logger.add(sys.__stdout__, level="DEBUG")
    stream = StreamToLogger()

    info_logger_id = logger.add(constants.INFO_LOGGING_FILE_NAME, level="INFO")
    info_debug_id = logger.add(constants.DEBUG_LOGGING_FILE_NAME, level="DEBUG")
    return stream, info_logger_id, info_debug_id


def init_GPU_config():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        logger.info(f"Found GPU: {gpus[0].name}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=constants.GPU_MEMORY_LIMIT_MB)])
    else:
        logger.info("No GPU found")


def move_log_file_to_experiment_folder(logger_id, experiment_path, filename):
    logger.remove(logger_id)
    new_log_path = experiment_path / filename
    shutil.move(filename, new_log_path)
    logger.add(new_log_path)


def run_scenario(current_scenario):

    current_scenario.instantiate_scenario_partners()
    # Split data according to scenario and then pre-process successively...
    # ... train data, early stopping validation data, test data
    if isinstance(current_scenario.samples_split_option, list):
        current_scenario.split_data_advanced()
    else:
        current_scenario.split_data()
    current_scenario.plot_data_distribution()
    current_scenario.compute_batch_sizes()
    current_scenario.preprocess_scenarios_data()

    # Train and eval on all partners according to scenario
    is_save_fig = True
    start = timer()
    mpl = multi_partner_learning.init_multipartnerlearning_from_scenario(current_scenario)
    current_scenario.federated_test_result = mpl.compute_test_score()
    end = timer()
    current_scenario.federated_computation_time_sec = end - start

    for method in current_scenario.methods:
        logger.info(f"{method}")
        contrib = contributivity.Contributivity(scenario=current_scenario)
        contrib.compute_contributivity(method, current_scenario)
        current_scenario.append_contributivity(contrib)
        logger.info(f"## Evaluating contributivity with {method}: {contrib}")

    return 0


def get_config_from_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="input config file")
    args = parser.parse_args()
    if args.file:
        logger.info(f"Using provided config file: {args.file}")
        config_filepath = args.file
    else:
        logger.info(f"Using default config file: {DEFAULT_CONFIG_FILE}")
        config_filepath = DEFAULT_CONFIG_FILE

    config = utils.load_cfg(config_filepath)
    config = utils.init_result_folder(config_filepath, config)

    return config


class StreamToLogger:
    def __init__(self, level="INFO"):
        self._level = level

    def write(self, buffer):
        for line in buffer.rstrip().splitlines():
            logger.opt(depth=1).log(self._level, line.rstrip())

    def flush(self):
        pass


if __name__ == "__main__":
    main()
