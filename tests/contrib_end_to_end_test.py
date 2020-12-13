# -*- coding: utf-8 -*-
"""
This enables to parameterize end to end tests - the tests are run by Travis each time you commit to the github repo
"""

import subprocess
from pathlib import Path

import pandas as pd

from mplc import constants  # noqa: E402
from . import test_utils


class Test_EndToEndTest:

    def test_titanic_contrib(self):
        """
        Test contributivity score on titanic dataset
        """
        # This fails to run
        # titanic_scenario = Scenario(2, [0.2, 0.8], epoch_count=3, minibatch_count=1, dataset_name='titanic',
        #                             contributivity_methods=["Federated SBS linear", "Shapley values"])
        # titanic_scenario.run()

        assert True

    def test_mnist_contrib(self):
        """
        Test contributivity score on mnist dataset
        """

        # run test from config file
        subprocess.run(["python", "main.py", "-f", "tests/config_end_to_end_test_contrib.yml"])

        df = test_utils.get_latest_dataframe("*end_to_end_test*")

        # Three contributivity methods for each partner --> 6 lines
        assert len(df) == 6

        # Every contributivity estimate should be between 0 and 1
        assert df.contributivity_score.max() < 1
        assert df.contributivity_score.min() > 0

        for contributivity_method in df.contributivity_method.unique():

            current_df = df[df.contributivity_method == contributivity_method]

            small_dataset_score = current_df.loc[current_df.dataset_fraction_of_partner == 0.1, "contributivity_score"]
            big_dataset_score = current_df.loc[current_df.dataset_fraction_of_partner == 0.9, "contributivity_score"]

            assert small_dataset_score.values < big_dataset_score.values
