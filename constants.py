# -*- coding: utf-8 -*-
"""
Declaration of constants.
"""

# Dataset
DATASET = str

# Data constants
NUM_CLASSES = 10

# input image dimensions
IMG_ROWS, IMG_COLS = 28, 28
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)

# ML constants
DEFAULT_BATCH_SIZE = 256
DEFAULT_GRADIENT_UPDATES_PER_PASS_COUNT = 8
PATIENCE = 4  # patience for early stopping

# GPU
GPU_MEMORY_LIMIT_MB = 4096

# Logging
INFO_LOGGING_FILE_NAME = "info.log"
DEBUG_LOGGING_FILE_NAME = "debug.log"

# Contributivity methods names
# TODO
