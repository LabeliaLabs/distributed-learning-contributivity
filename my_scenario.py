# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:43:26 2019

This enables to parameterize a desired scenario of data splitting amond nodes.
The scenario is then processed by data_splitting.py

@author: @bowni
"""

# Define the desired number of independant nodes
# Nodes mock different partners in a collaborative data science project
NODES_COUNT = 3

# Configure the desired respective datasets sizes of the nodes
# Should the nodes receive an equivalent amount of samples each...
# ... or receive different amounts?
# Define the percentages of samples per node
# Sum has to equal 1 and number of items has to equal NODES_COUNT
AMOUNTS_PER_NODE = [0.1, 0.25, 0.65]

# Configure if nodes get overlapping or distinct samples
# Should the nodes receive data from distinct categories...
# ... or just random samples?
OVERLAP_OR_DISTINCT = 'Overlap' # Change to 'Overlap' if desired