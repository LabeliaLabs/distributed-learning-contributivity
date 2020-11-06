# -*- coding: utf-8 -*-
"""
An Experiment regroups multiple scenarios and enables to run them and analyze their results.
"""

from loguru import logger

from . import scenario


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

    def add_scenario(self, scenario_to_add):
        """Add a scenario to the list of scenarios to be run"""

        if isinstance(scenario_to_add, scenario.Scenario):
            self.scenarios_list.append(scenario_to_add)
        else:
            logger.info(f"The scenario {scenario_to_add} you are trying to add is not an instance of"
                        f"object scenario.Scenario")

    def run_experiment(self):
        """Run the experiment, starting by validating the scenarios first"""

        # First: validate scenarios
        pass

        # Loop over nb_repeats
        pass

            # Loop over scenarios in scenarios_list
            pass

                # Run the scenario
                pass

                # Store scenario results in a global results object/file
                pass

        # Save the final results object/file
        # Produce a default analysis notebook
        pass
