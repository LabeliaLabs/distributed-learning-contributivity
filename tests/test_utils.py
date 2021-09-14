# -*- coding: utf-8 -*-
"""
This enables to parameterize end to end tests - the tests are run by Travis each time you commit to the github repo
"""

from pathlib import Path

import pandas as pd

from mplc import constants  # noqa: E402


def get_latest_dataframe(pattern, path=constants.EXPERIMENTS_FOLDER_NAME):
    # Get latest experiment folder
    root_folder = Path().absolute() / path
    subfolder_list = list(root_folder.glob('*end_to_end_test*'))
    subfolder_list_creation_time = [f.stat().st_ctime for f in subfolder_list]
    latest_subfolder_idx = subfolder_list_creation_time.index(max(subfolder_list_creation_time))

    experiment_path = subfolder_list[latest_subfolder_idx]

    return pd.read_csv(experiment_path / constants.RESULT_FILE_NAME)
