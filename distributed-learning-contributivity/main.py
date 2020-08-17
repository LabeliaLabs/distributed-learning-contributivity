# -*- coding: utf-8 -*-
"""
A script for:
    - mocking a multi-partner ML project by splitting data among different partners
    - training a model across multiple partners in a distributed approach
    - measuring the respective contributions of each partner to the model performance (termed "contributivity")
"""

import matplotlib.pyplot as plt
from loguru import logger
import tensorflow as tf

import argparse
import contextlib
import os
import shutil
import sys

import constants
import contributivity
import multi_partner_learning
import scenario
import utils

DEFAULT_CONFIG_FILE = "config.yml"

@logger.catch
def main():

    args = parse_command_line_arguments()

    stream, info_logger_id, info_debug_id = init_logger(args)

    with contextlib.redirect_stdout(stream):
        logger.debug("Standard output is sent to added handlers.")

        config = get_config_from_file(args)
        scenario_params_list = utils.get_scenario_params_list(
            config["scenario_params_list"])

        experiment_path = config["experiment_path"]
        n_repeats = config["n_repeats"]

        validate_scenario_list(scenario_params_list, experiment_path)

        for scenario_id, scenario_params in enumerate(scenario_params_list):
            logger.info(f"Scenario {scenario_id+1}/{len(scenario_params_list)}: {scenario_params}")

        # Move log files to experiment folder
        move_log_file_to_experiment_folder(info_logger_id, experiment_path, constants.INFO_LOGGING_FILE_NAME, "INFO")
        move_log_file_to_experiment_folder(info_debug_id, experiment_path, constants.DEBUG_LOGGING_FILE_NAME, "DEBUG")

        # GPU config
        init_gpu_config()

        # Close open figures
        plt.close("all")

        # Iterate over repeats of all scenarios experiments
        for i in range(n_repeats):

            logger.info(f"Repeat {i+1}/{n_repeats}")

            for scenario_id, scenario_params in enumerate(scenario_params_list):

                logger.info(f"Scenario {scenario_id + 1}/{len(scenario_params_list)}")
                logger.info("Current params:")
                logger.info(scenario_params)

                current_scenario = scenario.Scenario(
                    scenario_params,
                    experiment_path,
                    scenario_id=scenario_id+1,
                    n_repeat=i+1
                )

                scenario.run_scenario(current_scenario)

                # Write results to CSV file
                df_results = current_scenario.to_dataframe()
                df_results["random_state"] = i
                df_results["scenario_id"] = scenario_id

                with open(experiment_path / "results.csv", "a") as f:
                    df_results.to_csv(f, header=f.tell() == 0, index=False)
                    logger.info(f"Results saved to {os.path.relpath(experiment_path)}/results.csv")

    return 0


def init_logger(args):
    logger.remove()

    # Forward logging to standard output
    if args.verbose:
        logger.add(sys.__stdout__, level="DEBUG")
    else:
        logger.add(sys.__stdout__, level="INFO")

    stream = StreamToLogger()

    info_logger_id = logger.add(constants.INFO_LOGGING_FILE_NAME, level="INFO")
    info_debug_id = logger.add(constants.DEBUG_LOGGING_FILE_NAME, level="DEBUG")
    return stream, info_logger_id, info_debug_id

def init_gpu_config():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        logger.info(f"Found GPU: {gpus[0].name}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=constants.GPU_MEMORY_LIMIT_MB)])
    else:
        logger.info("No GPU found")


def move_log_file_to_experiment_folder(logger_id, experiment_path, filename, level):
    logger.remove(logger_id)
    new_log_path = experiment_path / filename
    shutil.move(filename, new_log_path)
    logger.add(new_log_path, level=level)


def validate_scenario_list(scenario_params_list, experiment_path):
    """Instantiate every scenario without running it to check if
    every scenario is correctly specified. This prevents scenario initialization errors during the experiment"""

    logger.debug("Starting to validate scenarios")

    for scenario_id, scenario_params in enumerate(scenario_params_list):

        logger.debug(f"Validation scenario {scenario_id + 1}/{len(scenario_params_list)}")

        # TODO: we should not create scenario folder at this point
        current_scenario = scenario.Scenario(scenario_params, experiment_path, is_dry_run=True)
        current_scenario.instantiate_scenario_partners()

        if current_scenario.samples_split_type == 'basic':
            current_scenario.split_data(is_logging_enabled=False)
        elif current_scenario.samples_split_type == 'advanced':
            current_scenario.split_data_advanced(is_logging_enabled=False)

    logger.debug("All scenario have been validated")


def parse_command_line_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="input config file")
    parser.add_argument("-v", "--verbose", help="verbose output", action="store_true")
    args = parser.parse_args()

    return args


def get_config_from_file(args):

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
