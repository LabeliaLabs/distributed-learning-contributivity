# -*- coding: utf-8 -*-
"""
An Experiment regroups multiple scenarios and enables to run them and analyze their results.
"""

import datetime
import os
import uuid
from pathlib import Path
from shutil import copyfile

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from loguru import logger

from . import scenario as scenario_module
from . import utils, constants
from .utils import get_scenario_params_list, load_cfg

DEFAULT_CONFIG_FILE = "../config.yml"


class ScenarioList(list):
    def __init__(self, experiment):
        self.experiment = experiment
        super().__init__()

    def append(self, scenario_to_add: scenario_module.Scenario) -> None:
        """Add a scenario to the list of scenarios to be run"""

        if isinstance(scenario_to_add, scenario_module.Scenario):
            new_id = self.__len__()
            scenario_to_add.scenario_name = scenario_to_add.scenario_name.replace(
                f'scenario_{scenario_to_add.scenario_id}',
                f'scenario_{new_id}')
            scenario_to_add.scenario_id = new_id
            if self.experiment.is_save:
                scenario_to_add.save_folder = self.experiment.experiment_path / scenario_to_add.scenario_name
            super(ScenarioList, self).append(scenario_to_add)
        else:
            raise Exception(f"The scenario {scenario_to_add} you are trying to add is not an instance of"
                            f"object scenario.Scenario")

    def extend(self, __iterable) -> None:
        for i in __iterable:
            self.append(i)

    def insert(self, __index: int, __scenario_to_insert) -> None:
        """Insert a scenario in the list of scenarios to be run"""

        if isinstance(__scenario_to_insert, scenario_module.Scenario):
            new_id = self.__len__()
            __scenario_to_insert.scenario_name = __scenario_to_insert.scenario_name.replace(
                f'scenario_{__scenario_to_insert.scenario_id}',
                f'scenario_{new_id}')
            __scenario_to_insert.scenario_id = new_id
            if self.experiment.is_save:
                __scenario_to_insert.save_folder = self.experiment.experiment_path / __scenario_to_insert.scenario_name
            super(ScenarioList, self).insert(__index, __scenario_to_insert)
        else:
            raise Exception(f"The scenario {__scenario_to_insert} you are trying to add is not an instance of"
                            f"object scenario.Scenario")

    def __setitem__(self, key, scenario_to_set):
        if isinstance(scenario_to_set, scenario_module.Scenario):
            new_id = self.__len__()
            scenario_to_set.scenario_name = scenario_to_set.scenario_name.replace(
                f'scenario_{scenario_to_set.scenario_id}',
                f'scenario_{new_id}')
            scenario_to_set.scenario_id = new_id
            if self.experiment.is_save:
                scenario_to_set.save_folder = self.experiment.experiment_path / scenario_to_set.scenario_name
            return super(ScenarioList, self).__setitem__(key, scenario_to_set)
        else:
            raise Exception(f"The scenario {scenario_to_set} you are trying to add is not an instance of"
                            f"object scenario.Scenario")


class Experiment:

    def __init__(
            self,
            experiment_name='experiment',
            nb_repeats=1,
            scenarios_list=[],
            is_save=True,
            **kwargs,
    ):
        """
        :param experiment_name: string, name of the experiment
        :param nb_repeats: int, number of repeats of the experiments (as an experiment includes
                           a number of non-deterministic phenomena). Example: 5
        :param scenarios_list: list, list of scenarios to be run during the experiment.
                               Scenario can also be added via the .add_scenario() method.
        :param is_save: boolean. If set to True, the experiment results will be saved on disk.
        """

        now = datetime.datetime.now()
        now_str = now.strftime("%Y-%m-%d_%Hh%M")
        self.name = experiment_name + "_" + now_str

        self.is_save = is_save
        if is_save:
            self.experiment_path = self.define_experiment_path(**kwargs)

        self.result = pd.DataFrame({})
        self.scenarios_list = ScenarioList(self)
        for scenario in scenarios_list:
            self.add_scenario(scenario)
        self.nb_repeats = nb_repeats

    def define_experiment_path(self, **kwargs):
        """Define the path and create folder for saving results of the experiment"""

        experiment_path = Path.cwd() / kwargs.get('experiment_path', constants.EXPERIMENTS_FOLDER_NAME) / self.name

        # Check if experiment folder already exists
        if experiment_path.exists():
            logger.warning(f"Experiment folder {experiment_path} already exists")
            experiment_path = Path(f"{experiment_path}_{uuid.uuid4().hex[:3]}")  # to distinguish identical names
            logger.warning(f"Experiment folder has been renamed to: {experiment_path}")

        experiment_path.mkdir(parents=True, exist_ok=False)

        return experiment_path

    def add_scenario(self, scenario_to_add):
        self.scenarios_list.append(scenario_to_add)

    def run(self):
        """Run the experiment """

        # Preliminary steps
        logger.info(f"Now running experiment {self.name}")
        plt.close("all")  # Close open figures
        if self.is_save:
            utils.set_log_file(self.experiment_path)  # Move log files to experiment folder

        # Loop over nb_repeats
        for repeat_idx in range(self.nb_repeats):

            repeat_index_str = f"{repeat_idx + 1}/{self.nb_repeats}"
            logger.info(f"(Experiment {self.name}) Now starting repeat {repeat_index_str}")

            # Loop over scenarios in scenarios_list
            for scenario_idx, blank_scenario in enumerate(self.scenarios_list):
                # Remove previous tensorflow graphes to free memory
                tf.keras.backend.clear_session()

                scenario_index_str = f"{scenario_idx + 1}/{len(self.scenarios_list)}"
                logger.info(f"(Experiment {self.name}, repeat {repeat_index_str}) "
                            f"Now running scenario {scenario_index_str}")
                blank_scenario.is_run_as_part_of_an_experiment = True

                # Run the scenario
                if self.is_save:
                    scenario = blank_scenario.copy(repeat_count=repeat_idx, save_path=self.experiment_path)
                else:
                    scenario = blank_scenario.copy(repeat_count=repeat_idx)
                scenario.is_run_as_part_of_an_experiment = True
                scenario.run()
                scenario.plot_data_distribution(save=True, display=False)

                # Save scenario results
                df_results = scenario.to_dataframe()
                df_results["repeat_index"] = repeat_idx
                df_results["scenario_index"] = scenario_idx

                if self.is_save:
                    with open(self.experiment_path / constants.RESULT_FILE_NAME, "a") as f:
                        df_results.to_csv(f, header=f.tell() == 0, index=False)
                        logger.info(
                            f"(Experiment {self.name}, repeat {repeat_index_str}, scenario {scenario_index_str}) "
                            f"Results saved to results.csv in folder {os.path.relpath(self.experiment_path)}")

                self.result = self.result.append(df_results)

        # TODO: Produce a default analysis notebook
        pass


def init_experiment_from_config_file(path_to_config_file):
    """
    Create experiment, populate with scenarios according with a config file passed as argument,
    """
    logger.debug(f"Initializing experiment with config file at path {path_to_config_file}")

    config = load_cfg(path_to_config_file)
    exp_params = {'is_save': True,
                  'experiment_name': config["experiment_name"],
                  'nb_repeats': config["n_repeats"]}

    experiment = Experiment(**exp_params)

    #  copy the config file into the result folder
    target_yaml_filepath = experiment.experiment_path / Path(path_to_config_file).name
    copyfile(path_to_config_file, target_yaml_filepath)

    scenario_params_list = get_scenario_params_list(config["scenario_params_list"])

    logger.info("Creating scenarios from config file")

    for scenario_params_idx, scenario_params in enumerate(scenario_params_list):
        scenario_params_index_str = f"{scenario_params_idx + 1}/{len(scenario_params_list)}"
        logger.debug(f"Scenario {scenario_params_index_str}: {scenario_params}")

        current_scenario = scenario_module.Scenario(**scenario_params, scenario_id=scenario_params_idx + 1)

        experiment.add_scenario(current_scenario)
        logger.info(f"Scenario {current_scenario.scenario_name} successfully validated "
                    f"and added to the experiment")

    logger.info("All scenarios created , successfully validated and added to the experiment")

    return experiment
