# -*- coding: utf-8 -*-
"""
A script for:
    - mocking a multi-partner ML project by splitting data among different partners
    - training a model across multiple partners in a distributed approach
    - measuring the respective contributions of each partner to the model performance (termed "contributivity")
"""

from loguru import logger

from mplc import utils
from mplc.experiment import init_experiment_from_config_file
from mplc.utils import parse_command_line_arguments

DEFAULT_CONFIG_FILE = "./config.yml"


@logger.catch
def main():

    # Get arguments
    args = parse_command_line_arguments()

    # Initialize log level
    utils.init_logger(args.verbose)
    logger.debug("Standard output is sent to added handlers.")

    # Initialize GPU configuration
    utils.init_gpu_config()

    # Initialize experiment from configuration file
    config_filepath = args.file if args.file else DEFAULT_CONFIG_FILE
    if not args.file:
        logger.info(f"No config file specified, using default config file: {DEFAULT_CONFIG_FILE}")
    main_experiment = init_experiment_from_config_file(config_filepath)

    # Run the experiment
    main_experiment.run()

    return 0


if __name__ == "__main__":
    main()
