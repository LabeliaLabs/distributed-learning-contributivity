# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:09:07 2019

This processes the split of data among nodes according to scenario defined in my_scenario.py

@author: @bowni
"""

from __future__ import print_function
import keras
from keras.datasets import mnist

import matplotlib.pyplot as plt
import numpy as np

import my_scenario

from node import Node


#%% Load data

# load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('\n### MNIST data loaded:')
print('- ' + str(len(x_train)) + ' train data')
print('- ' + str(len(x_test)) + ' test data')


#%% Constants

# Define the desired number of independant nodes
nodes_count = my_scenario.NODES_COUNT
print('\n### Description of data scenario configured:')
print('- Number of nodes defined:', nodes_count)

#%% Configure the desired splitting scenario - Datasets sizes
# Should the nodes receive an equivalent amount of samples each...
# ... or receive different amounts?

# First we define the percentages of samples per node
amounts_per_node = my_scenario.AMOUNTS_PER_NODE
assert(len(amounts_per_node) == nodes_count)
assert(np.sum(amounts_per_node) == 1)

# Then we parameterize this via the splitting_indices to be passed to np.split
splitting_indices = np.empty((nodes_count-1,))
splitting_indices[0] = amounts_per_node[0]
for i in range(nodes_count-2):
    splitting_indices[i+1] = splitting_indices[i] + amounts_per_node[i+1]
splitting_indices = (splitting_indices * len(y_train)).astype(int)
print('- Splitting indices defined:', splitting_indices)


#%% Configure the desired splitting scenario - Overlapping or distinct samples
# Should the nodes receive data from distinct regions of space...
# ... or from the complete universe?

# First we indicate which scenario we want
overlap_or_distinct = my_scenario.OVERLAP_OR_DISTINCT
print('- Data distribution scenario chosen:', overlap_or_distinct)

# Create a list of indexes of the samples
train_idx = np.arange(len(y_train))
test_idx = np.arange(len(y_test))

# In the 'Distinct' scenario we sort MNIST by labels
if overlap_or_distinct == 'Distinct':
    
    # Sort MNIST by labels
    y_sorted_idx = y_train.argsort()
    y_train = y_train[y_sorted_idx]
    x_train = x_train[y_sorted_idx]
    
    # Print and plot for controlling
    print('\nFirst 10 elements of y_train:' + str(y_train[:10]))
    print('First image:')
    first_image = x_train[0,:,:]
    plt.gray()
    plt.imshow(first_image)
    plt.show()
    print('Last 10 elements of y_train' + str(y_train[-10:]))
    print('Last image:')
    last_image = x_train[-1,:,:]
    plt.gray()
    plt.imshow(last_image)
    plt.show()
    
# In the 'Overlap' scenario we shuffle randomly the indexes
elif overlap_or_distinct == 'Overlap':
    np.random.seed(42)
    np.random.shuffle(train_idx)

# If neither 'Distinct' nor 'Overlap', we quit
else:
    print('This overlap_or_distinct scenario is not recognized. Quitting with quit()')
    quit()
    
    
#%% Do the splitting among nodes according to desired scenarios of...
# ... data amount per node and overlap/distinct distribution

# Split data between nodes
train_idx_idx_list = np.split(train_idx, splitting_indices)
test_idx_idx_list = np.split(test_idx, splitting_indices)    

# Populate nodes
node_list = []

for train_idx, test_idx in zip(train_idx_idx_list, test_idx_idx_list):
    
    x_node_train = x_train[train_idx, :]
    x_node_test = x_test[test_idx]
    y_node_train = y_train[train_idx,]
    y_node_test = y_test[test_idx]
    
    node = Node(x_node_train, x_node_test, y_node_train, y_node_test)
    node_list.append(node)

# Check coherence of node_list versus nodes_count   
assert(len(node_list) == nodes_count)

# Print and plot for controlling
for node_index, node in enumerate(node_list):
    print('\nNode #' + str(node_index + 1) + ':')
    print('First 10 elements of y_train:' + str(node.y_train[:10]))
    print('First image:')
    first_image = node.x_train[0,:,:]
    plt.gray()
    plt.imshow(first_image)
    plt.show()
    print('Last 10 elements of y_train:' + str(node.y_train[-10:]))
    print('Last image:')
    last_image = node.x_train[-1,:,:]
    plt.gray()
    plt.imshow(last_image)
    plt.show()