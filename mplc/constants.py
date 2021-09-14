# -*- coding: utf-8 -*-
"""
Declaration of constants.
"""

# ML constants
DEFAULT_BATCH_SIZE = 256
MAX_BATCH_SIZE = 2 ** 20
DEFAULT_GRADIENT_UPDATES_PER_PASS_COUNT = 8
PATIENCE = 10  # patience for early stopping
MIN_DELTA_FOR_EARLY_STOPPING = 0
DEFAULT_BATCH_COUNT = 20
DEFAULT_EPOCH_COUNT = 40
# GPU
GPU_MEMORY_LIMIT_MB = 4096

# Logging
INFO_LOGGING_FILE_NAME = "info.log"
DEBUG_LOGGING_FILE_NAME = "debug.log"

# Paths
EXPERIMENTS_FOLDER_NAME = "experiments"
SINGLE_SCENARIOS_FOLDER_NAME = "standalone_scenarios"
RESULT_FILE_NAME = "results.csv"

# Number of samples for quick_demo
TRAIN_SET_MAX_SIZE_QUICK_DEMO = 1000
VAL_SET_MAX_SIZE_QUICK_DEMO = 500
TEST_SET_MAX_SIZE_QUICK_DEMO = 500
# Contributivity contributivity_methods names
CONTRIBUTIVITY_METHODS = [
    "Shapley values",
    "Independent scores",
    "TMCS",
    "ITMCS",
    "IS_lin_S",
    "IS_reg_S",
    "AIS_Kriging_S",
    "SMCS",
    "WR_SMC",
    "Federated SBS linear",
    "Federated SBS quadratic",
    "Federated SBS constant",
    "S-Model",
    "PVRL",
]

# Datasets' Tags
MNIST = "mnist"
CIFAR10 = "cifar10"
TITANIC = "titanic"
ESC50 = "esc50"
IMDB = 'imdb'
FMNIST = "fmnist"
# Supported datasets
SUPPORTED_DATASETS_NAMES = [MNIST, CIFAR10, TITANIC, ESC50, IMDB, FMNIST]

# Number of attempts allowed before raising an error while trying to download dataset
NUMBER_OF_DOWNLOAD_ATTEMPTS = 3
