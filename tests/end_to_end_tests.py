# -*- coding: utf-8 -*-
"""
This enables to parameterize end to end tests - the tests are run by Travis each time you commit to the github repo
"""

import pytest
import pandas as pd
import subprocess
import sys
sys.path.append("..")
import constants

from pathlib import Path


class Test_EndToEndTest:

    def test_mnist(self):
        """
        Test performance on MNIST dataset after one epoch
        """
        # run test
        subprocess.run(["python", "main.py", "-f", "tests/config_end_to_end_test.yml"])

        # Get latest experiment folder
        root_folder = Path().absolute() / constants.EXPERIMENTS_FOLDER_NAME
        subfolder_list = list(root_folder.glob('*end_to_end_test*'))
        subfolder_list_creation_time = [f.stat().st_ctime for f in subfolder_list]
        latest_subfolder_idx =  subfolder_list_creation_time.index(max(subfolder_list_creation_time))

        experiment_path = subfolder_list[latest_subfolder_idx]
        df = pd.read_csv(experiment_path / "results.csv")

        # Extract score 
        min_test_score = df["mpl_test_score"].min()
        
        assert min_test_score > 0.96
