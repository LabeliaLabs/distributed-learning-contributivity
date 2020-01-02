# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:04:48 2019

@author: RGOUSSAULT
"""

#%% Data constants

NUM_CLASSES = 10

# input image dimensions
IMG_ROWS, IMG_COLS = 28, 28
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 1)


#%% ML constants

BATCH_SIZE = 4096
PATIENCE = 4 # patience for early stopping
