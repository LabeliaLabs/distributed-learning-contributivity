# -*- coding: utf-8 -*-
"""
This enables to parameterize end to end tests - the tests are run by Travis each time you commit to the github repo
"""

import subprocess

import numpy as np
import pandas as pd

import mplc.multi_partner_learning as mpl
from mplc.corruption import Duplication
from mplc.experiment import Experiment
from mplc.scenario import Scenario
from . import test_utils


class Test_EndToEndTest:

    def test_mnist(self):
        """
        Test performance on MNIST dataset after one epoch
        """
        # run test
        subprocess.run(["python", "main.py", "-f", "tests/config_end_to_end_test_mnist.yml"])

        df = test_utils.get_latest_dataframe("*end_to_end_test*")

        # Extract score
        min_test_score = df["mpl_test_score"].min()
        assert min_test_score > 0.95

    def test_titanic(self):
        """
        Test performance on titanic dataset
        """
        corruption_1 = Duplication(proportion=0.8, duplicated_partner_id=0)
        titanic_scenario_1 = Scenario(2, [0.4, 0.6], epoch_count=3, minibatch_count=1, dataset='titanic')
        titanic_scenario_2 = Scenario(3, [0.2, 0.2, 0.6],
                                      corruption_parameters=['not-corrupted', corruption_1, 'not-corrupted'],
                                      epoch_count=3, minibatch_count=1, dataset='titanic')
        exp = Experiment(experiment_name='end_to_end_titanic',
                         scenarios_list=[titanic_scenario_1],
                         nb_repeats=2,
                         is_save=True)
        exp.add_scenario(titanic_scenario_2)
        exp.run()
        titanic_scenario_1.run()
        assert np.min(titanic_scenario_1.mpl.history.score) > 0.65
        result = pd.read_csv(exp.experiment_path / 'results.csv')
        assert (result.groupby('scenario_index').mean().mpl_test_score > 0.65).all()

    def test_all_mpl_approaches(self):
        """
        Test all the multi_partner_learning approaches
        """

        exp = Experiment(is_save=False)
        # mpl_approaches = mpl.BASIC_MPL_APPROACHES.copy()
        mpl_approaches = mpl.MULTI_PARTNER_LEARNING_APPROACHES.copy()
        _ = mpl_approaches.pop("single-partner")
        for approach in mpl_approaches:
            exp.add_scenario(Scenario(2, [0.25, 0.75], epoch_count=2, minibatch_count=2, dataset='mnist',
                                      dataset_proportion=0.1, multi_partner_learning_approach=approach,
                                      gradient_updates_per_pass_count=3))
        exp.add_scenario(Scenario(1,
                                  [1],
                                  epoch_count=2,
                                  minibatch_count=2,
                                  dataset='mnist',
                                  dataset_proportion=0.1,
                                  multi_partner_learning_approach='single-partner',
                                  gradient_updates_per_pass_count=3))
        exp.run()

        df = exp.result
        assert len(df) == len(mpl.MULTI_PARTNER_LEARNING_APPROACHES)
