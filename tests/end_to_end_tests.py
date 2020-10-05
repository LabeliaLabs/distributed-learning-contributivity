# -*- coding: utf-8 -*-
"""
This enables to parameterize end to end tests - the tests are run by Travis each time you commit to the github repo
"""

import pandas as pd
import subprocess
import sys
from pathlib import Path

sys.path.append("..")
import constants  # noqa: E402


class Test_EndToEndTest:

    @staticmethod
    def get_latest_dataframe(pattern):

        # Get latest experiment folder
        root_folder = Path().absolute() / constants.EXPERIMENTS_FOLDER_NAME
        subfolder_list = list(root_folder.glob('*end_to_end_test*'))
        subfolder_list_creation_time = [f.stat().st_ctime for f in subfolder_list]
        latest_subfolder_idx = subfolder_list_creation_time.index(max(subfolder_list_creation_time))

        experiment_path = subfolder_list[latest_subfolder_idx]

        return pd.read_csv(experiment_path / "results.csv")

    def test_mnist(self):
        """
        Test performance on MNIST dataset after one epoch
        """
        # run test
        subprocess.run(["python", "main.py", "-f", "tests/config_end_to_end_test_mnist.yml"])

        df = Test_EndToEndTest.get_latest_dataframe("*end_to_end_test*")

        # Extract score
        min_test_score = df["mpl_test_score"].min()

        assert min_test_score > 0.95

    def test_contrib(self):
        """
        Test contrib score
        """
        # run test
        subprocess.run(["python", "main.py", "-f", "tests/config_end_to_end_test_contrib.yml"])

        df = Test_EndToEndTest.get_latest_dataframe("*end_to_end_test*")

        # 2 contributivity methods for each partner --> 4 lines
        assert len(df) == 4

        for contributivity_method in df.contributivity_method.unique():

            current_df = df[df.contributivity_method == contributivity_method]

            small_dataset_score = current_df.loc[current_df.dataset_fraction_of_partner == 0.1, "contributivity_score"]
            big_dataset_score = current_df.loc[current_df.dataset_fraction_of_partner == 0.9, "contributivity_score"]

            assert small_dataset_score.values < big_dataset_score.values
