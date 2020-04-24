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

import constants
import contributivity
import fl_training
import scenario

import argparse

DEFAULT_CONFIG_FILE = "config.yml"

class StreamToLogger:

    def __init__(self, level="INFO"):
        self._level = level

    def write(self, buffer):
        for line in buffer.rstrip().splitlines():
            logger.opt(depth=1).log(self._level, line.rstrip())

    def flush(self):
        pass


def main():

    logger.remove()
    logger.add(sys.__stdout__)
    stream = StreamToLogger()

    logger.add("experiment.log")

    with contextlib.redirect_stdout(stream):
        print("Standard output is sent to added handlers.")

    # Parse config file for scenarios to be experimented
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
    experiment_path = config["experiment_path"]

    scenario_params_list = config["scenario_params_list"]
    n_repeats = config["n_repeats"]

    # GPU config
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        logger.info(f"Found GPU: {gpus[0].name}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=constants.GPU_MEMORY_LIMIT_MB)])
    else:
        logger.info("No GPU found")

    # Close open figures
    plt.close("all")

    # Iterate over repeats of all scenarios experiments
    for i in range(n_repeats):

        logger.info(f"Repeat {i+1}/{n_repeats}")

        for scenario_id, scenario_params in enumerate(scenario_params_list):

            logger.info("Current params:")
            logger.info(scenario_params)

            print(type(scenario_params["amounts_per_partner"]))

            current_scenario = scenario.Scenario(scenario_params, experiment_path)
            print(current_scenario.to_dataframe())

            run_scenario(current_scenario)

            # Write results to CSV file
            df_results = current_scenario.to_dataframe()
            df_results["random_state"] = i
            df_results["scenario_id"] = scenario_id

            with open(experiment_path / "results.csv", "a") as f:
                df_results.to_csv(f, header=f.tell() == 0, index=False)
                logger.info("Results saved")

    return 0


def run_scenario(current_scenario):

    # Split data according to scenario and then pre-process successively...
    # ... train data, early stopping validation data, test data
    current_scenario.split_data()
    current_scenario.plot_data_distribution()
    current_scenario = fl_training.preprocess_scenarios_data(current_scenario)

    # Train and eval on all partners according to scenario
    is_save_fig = True
    start = timer()
    current_scenario.federated_test_score = fl_training.compute_test_score_with_scenario(
        current_scenario, is_save_fig
    )
    end = timer()
    current_scenario.federated_computation_time_sec = end - start

    for method in current_scenario.methods:
        print(method)
        contrib = contributivity.Contributivity(scenario=current_scenario)
        contrib.compute_contributivity(method, current_scenario)
        current_scenario.append_contributivity(contrib)
        print("\n## Evaluating contributivity with " + method + ":")
        print(contrib)

    # Save results to file
    current_scenario.to_file()

    return 0


if __name__ == "__main__":
    main()
