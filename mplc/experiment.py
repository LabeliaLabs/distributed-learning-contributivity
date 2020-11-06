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
            scenarios_list=[],
            nb_repeats=1,
    ):
        """
        :param experiment_name: string, name of the experiment
        :param scenarios_list: [Scenario], lists instances of the Scenario object
        :param nb_repeats: int, number of repeats of the experiments (as an experiment includes
                           a number of non-deterministic phenomena. Example: 5
        """

        self.name = experiment_name
        self.scenarios_list = scenarios_list
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
            self.scenarios_list.append(scenario_to_add)
        else:
            logger.info(f"The scenario {scenario_to_add} you are trying to add is not an instance of"
                        f"object scenario.Scenario")

    def run_experiment(self):
        """Run the experiment, starting by validating the scenarios first"""

        # Preliminary steps
        plt.close("all")  # Close open figures
        utils.set_log_file(self.experiment_path)  # Move log files to experiment folder

        # First: validate scenarios
        pass

        # Loop over nb_repeats
        for i in range(self.nb_repeats):

            logger.info(f"Repeat {i + 1}/{self.nb_repeats}")

            # Loop over scenarios in scenarios_list
            for scenario_id, scenario in enumerate(self.scenarios_list):

                logger.info(f"Now running scenario {scenario_id + 1}/{len(self.scenarios_list)}")

                # Run the scenario
                scenario.run()

                # Store scenario results in a global results object/file
                df_results = scenario.to_dataframe()
                df_results["random_state"] = i
                df_results["scenario_id"] = scenario_id

                with open(self.experiment_path / "results.csv", "a") as f:
                    df_results.to_csv(f, header=f.tell() == 0, index=False)
                    logger.info(f"Results saved to {os.path.relpath(self.experiment_path)}/results.csv")

        # Save the final results object/file
        # Produce a default analysis notebook
        pass
