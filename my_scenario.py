# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:43:26 2019

This enables to parameterize a desired scenario of data splitting amond nodes.
The scenario is then processed by data_splitting.py

@author: @bowni
"""

import keras
from keras.datasets import mnist

# Define dataset of choice
(X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = mnist.load_data()
DATASET_NAME = 'MNIST' # For log printing purpose only

# Truncate dataset for quicker debugging/testing
X_TRAIN = X_TRAIN[:200]
Y_TRAIN = Y_TRAIN[:200]

# Define the desired number of independant nodes
# Nodes mock different partners in a collaborative data science project
NODES_COUNT = 3

# Configure the desired respective datasets sizes of the nodes
# Should the nodes receive an equivalent amount of samples each...
# ... or receive different amounts?
# Define the percentages of samples per node
# Sum has to equal 1 and number of items has to equal NODES_COUNT
AMOUNTS_PER_NODE = [0.2, 0.3, 0.5]

# Configure if nodes get overlapping or distinct samples
# Should the nodes receive data from distinct categories...
# ... or just random samples?
OVERLAP_OR_DISTINCT = 'Distinct' # Toggle between 'Overlap' and 'Distinct'

# Define if test data should be split between nodes...
# ... or if each node should refer to the complete test set
TESTSET_OPTION = 'Global' # Toggle between 'Global' and 'Split'