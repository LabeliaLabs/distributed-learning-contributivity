# -*- coding: utf-8 -*-
"""
Declaration of constants.
"""

# ML constants
DEFAULT_BATCH_SIZE = 256
MAX_BATCH_SIZE = 2 ** 20
DEFAULT_GRADIENT_UPDATES_PER_PASS_COUNT = 8
PATIENCE = 10  # patience for early stopping

# GPU
GPU_MEMORY_LIMIT_MB = 4096

# Logging
INFO_LOGGING_FILE_NAME = "info.log"
DEBUG_LOGGING_FILE_NAME = "debug.log"

# Paths
EXPERIMENTS_FOLDER_NAME = "experiments"

# Contributivity methods names
# TODO

# Datasets' Tags
MNIST = "mnist"
CIFAR10 = "cifar10"
TITANIC = "titanic"
ESC50 = "esc50"

# Supported datasets
SUPPORTED_DATASETS_NAMES = [MNIST, CIFAR10, TITANIC, ESC50]

# Number of attempts allowed before raising an error while trying to download dataset
NUMBER_OF_DOWNLOAD_ATTEMPTS = 3
