# -*- coding: utf-8 -*-
"""
An Experiment regroups multiple scenarios and enables to run them and analyze their results.
"""

import os
import datetime
import matplotlib.pyplot as plt
from loguru import logger
from pathlib import Path

from . import scenario as scenario_module
from . import utils, constants


class Experiment:
    def __init__(
            self,
            experiment_name=None,
            nb_repeats=1,
    ):
        """
        :param experiment_name: string, name of the experiment
        :param nb_repeats: int, number of repeats of the experiments (as an experiment includes
                           a number of non-deterministic phenomena. Example: 5
        """

        self.name = experiment_name
        self.scenarios_list = []
        self.nb_repeats = nb_repeats
        self.experiment_path = self.define_experiment_path()

    def define_experiment_path(self):
        """Define the path and create folder for saving results of the experiment"""

        now = datetime.datetime.now()
        now_str = now.strftime("%Y-%m-%d_%Hh%M")

        full_experiment_name = self.name + "_" + now_str
        experiment_path = Path.cwd() / constants.EXPERIMENTS_FOLDER_NAME / full_experiment_name

        # Check if experiment folder already exists
        while experiment_path.exists():
            logger.warning(f"Experiment folder {experiment_path} already exists")
            new_experiment_name = Path(str(experiment_path) + "_bis")
            experiment_path = Path.cwd() / constants.EXPERIMENTS_FOLDER_NAME / new_experiment_name
            logger.warning(f"Experiment folder has been renamed to: {experiment_path}")

        experiment_path.mkdir(parents=True, exist_ok=False)

        return experiment_path

    def add_scenario(self, scenario_to_add):
        """Add a scenario to the list of scenarios to be run"""

        if isinstance(scenario_to_add, scenario_module.Scenario):
            scenario_to_add.experiment_path = self.experiment_path
            self.scenarios_list.append(scenario_to_add)
        else:
            raise Exception(f"The scenario {scenario_to_add} you are trying to add is not an instance of"
                            f"object scenario.Scenario")

    def create_scenarios_from_config_file(self, path_to_config_file):
        """Create scenarios from a config file passed as argument,
           and populates scenarios list accordingly"""

        config = utils.get_config_from_file(path_to_config_file)
        scenario_params_list = utils.get_scenario_params_list(config["scenario_params_list"])

        logger.info(f"Creating scenarios from config file")

        for scenario_params_idx, scenario_params in enumerate(scenario_params_list):

            scenario_params_index_str = f"{scenario_params_idx + 1}/{len(scenario_params_list)}"
            logger.debug(f"Scenario {scenario_params_index_str}: {scenario_params}")

            current_scenario = scenario_module.Scenario(
                **scenario_params,
                experiment_path=self.experiment_path,
                scenario_id=scenario_params_idx+1,
            )

            self.add_scenario(current_scenario)

    def validate_scenarios_list(self):
        """Instantiate every scenario without running it to check if
           every scenario is correctly specified. This prevents scenario initialization errors during the experiment"""

        logger.debug("Starting validation of scenarios prior to running the experiment")

        for scenario_idx, scenario in enumerate(self.scenarios_list):

            scenario.is_dry_run = True

            scenario_index_str = f"{scenario_idx + 1}/{len(self.scenarios_list)}"
            logger.info(f"Scenarios validation: now validating scenario {scenario_index_str} "
                         f"(instantiate partners, split data, compute batch sizes, apply data corruption req.)")

            scenario.run()

            scenario.is_dry_run = False

        logger.debug("Scenarios validation: all scenario have been validated successfully")

    def run_experiment(self):
        """Run the experiment, starting by validating the scenarios first"""

        # Preliminary steps
        plt.close("all")  # Close open figures
        utils.set_log_file(self.experiment_path)  # Move log files to experiment folder

        # Validate scenarios prior to running the experiment
        self.validate_scenarios_list()

        # Loop over nb_repeats
        for repeat_idx in range(self.nb_repeats):

            repeat_index_str = f"{repeat_idx + 1}/{self.nb_repeats}"
            logger.info(f"Now starting repeat {repeat_index_str}")

            # Loop over scenarios in scenarios_list
            for scenario_idx, scenario in enumerate(self.scenarios_list):

                scenario_index_str = f"{scenario_idx + 1}/{len(self.scenarios_list)}"
                logger.info(f"(Repeat {repeat_index_str}) Now running scenario {scenario_index_str}")

                # Run the scenario
                scenario.n_repeat = repeat_idx
                scenario.run()

                # Save scenario results
                df_results = scenario.to_dataframe()
                df_results["repeat_index"] = repeat_idx
                df_results["scenario_index"] = scenario_idx

                with open(self.experiment_path / "results.csv", "a") as f:
                    df_results.to_csv(f, header=f.tell() == 0, index=False)
                    logger.info(f"(Repeat {repeat_index_str}, scenario {scenario_index_str}) "
                                f"Results saved to results.csv "
                                f"in folder {os.path.relpath(self.experiment_path)}")

        # TODO: Produce a default analysis notebook
        pass
