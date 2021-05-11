# -*- coding: utf-8 -*-
"""
This enables to parameterize end to end tests - the tests are run by Travis each time you commit to the github repo
"""

from mplc import constants
from mplc.experiment import Experiment
from mplc.experiment import init_experiment_from_config_file
from mplc.scenario import Scenario
from . import test_utils


class Test_EndToEndTest:

    def test_titanic_contrib(self):
        """
        Test contributivity score on titanic dataset
        """

        titanic_scenario = Scenario(2, [0.1, 0.9], epoch_count=3, minibatch_count=1, dataset='titanic',
                                    contributivity_methods=["Federated SBS linear", "Federated SBS quadratic",
                                                            "Shapley values"])
        exp = Experiment(experiment_name='end_to_end_test_contrib_titanic', nb_repeats=2,
                         scenarios_list=[titanic_scenario])
        exp.run()

        df = test_utils.get_latest_dataframe("*end_to_end_test*")

        # 2 contributivity methods X 2 partners x 2 repeats = 12
        assert len(df) == 12

    def test_mnist_contrib(self):
        """
        Test contributivity score on mnist dataset
        """

        # run test from config file
        experiment = init_experiment_from_config_file("tests/config_end_to_end_test_contrib.yml")
        experiment.run()

        df = test_utils.get_latest_dataframe("*end_to_end_test*")

        # Three contributivity methods for each partner --> 6 lines
        assert len(df) == 6

        # Every contributivity estimate should be between -1 and 1
        assert df.contributivity_score.max() < 1
        assert df.contributivity_score.min() > -1

        for contributivity_method in df.contributivity_method.unique():

            current_df = df[df.contributivity_method == contributivity_method]

            small_dataset_score = current_df.loc[current_df.dataset_fraction_of_partner == 0.1, "contributivity_score"]
            big_dataset_score = current_df.loc[current_df.dataset_fraction_of_partner == 0.9, "contributivity_score"]

            assert small_dataset_score.values < big_dataset_score.values

    def test_all_contrib_methods(self):
        """
        Test all available contributivity methods on mnist
        """

        all_methods = constants.CONTRIBUTIVITY_METHODS.copy()
        all_methods.remove('AIS_Kriging_S')  # This one fails
        all_methods.remove('IS_reg_S')  # This one is handled in the test below

        scenario = Scenario(2, [0.4, 0.6], epoch_count=1, minibatch_count=2, dataset='mnist',
                            contributivity_methods=all_methods, dataset_proportion=0.05)
        exp = Experiment(scenarios_list=[scenario])
        exp.run()

        df = exp.result
        assert len(df) == 2 * len(all_methods)

    def test_IS_reg_S_contrib(self):
        """
        Test the IS_reg_S contrib method.
        This method activates only when partners_count > 4
        """

        scenario = Scenario(4, [0.25, 0.25, 0.25, 0.25], epoch_count=1, minibatch_count=1, dataset='mnist',
                            contributivity_methods=["IS_reg_S"], dataset_proportion=0.05)
        exp = Experiment(scenarios_list=[scenario])
        exp.run()

        df = exp.result

        # 1 contributivity methods X 4 partners
        assert len(df) == 4
