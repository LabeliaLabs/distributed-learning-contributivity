# -*- coding: utf-8 -*-
"""
A script for:
    - mocking a multi-partner ML project by splitting data among different partners
    - training a model across multiple partners in a distributed approach
    - measuring the respective contributions of each partner to the model performance (termed "contributivity")
"""

import os

import matplotlib.pyplot as plt
from loguru import logger

from mplc import scenario
from mplc import utils
from mplc.utils import parse_command_line_arguments

DEFAULT_CONFIG_FILE = "./config.yml"


@logger.catch
def main():
    args = parse_command_line_arguments()

    if args.verbose:
        debug = True
    else:
        debug = False

    utils.init_logger(debug)

    logger.debug("Standard output is sent to added handlers.")

    if args.file:
        logger.info(f"Using provided config file: {args.file}")
        config = utils.get_config_from_file(args.file)
    else:
        logger.info(f"Using default config file: {DEFAULT_CONFIG_FILE}")
        config = utils.get_config_from_file(DEFAULT_CONFIG_FILE)

    scenario_params_list = utils.get_scenario_params_list(
        config["scenario_params_list"])

    experiment_path = config["experiment_path"]
    n_repeats = config["n_repeats"]
    validate_scenario_list(scenario_params_list, experiment_path)

    for scenario_id, scenario_params in enumerate(scenario_params_list):
        logger.info(f"Scenario {scenario_id + 1}/{len(scenario_params_list)}: {scenario_params}")

    # Move log files to experiment folder
    utils.set_log_file(experiment_path)

    # GPU config
    utils.init_gpu_config()

    # Close open figures
    plt.close("all")

    # Iterate over repeats of all scenarios experiments
    for i in range(n_repeats):

        logger.info(f"Repeat {i + 1}/{n_repeats}")

        for scenario_id, scenario_params in enumerate(scenario_params_list):
            logger.info(f"Scenario {scenario_id + 1}/{len(scenario_params_list)}")
            logger.info("Current params:")
            logger.info(scenario_params)

            current_scenario = scenario.Scenario(
                **scenario_params,
                experiment_path=experiment_path,
                scenario_id=scenario_id + 1,
                repeats_count=i + 1
            )

            current_scenario.run()

            # Write results to CSV file
            df_results = current_scenario.to_dataframe()
            df_results["random_state"] = i
            df_results["scenario_id"] = scenario_id

            with open(experiment_path / "results.csv", "a") as f:
                df_results.to_csv(f, header=f.tell() == 0, index=False)
                logger.info(f"Results saved to {os.path.relpath(experiment_path)}/results.csv")

    return 0


def validate_scenario_list(scenario_params_list, experiment_path):
    """Instantiate every scenario without running it to check if
    every scenario is correctly specified. This prevents scenario initialization errors during the experiment"""

    logger.debug("Starting to validate scenarios")

    for scenario_id, scenario_params in enumerate(scenario_params_list):

        logger.debug(f"Validation scenario {scenario_id + 1}/{len(scenario_params_list)}")

        # TODO: we should not create scenario folder at this point
        current_scenario = scenario.Scenario(**scenario_params, experiment_path=experiment_path)
        current_scenario.run(is_dry_run=True)

    logger.debug("All scenario have been validated")


if __name__ == "__main__":
    main()
