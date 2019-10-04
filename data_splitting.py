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


def process_data_splitting_scenario():
    """Return a list of nodes populated with their train and test data (not pre-processed)"""

    #%% Fetch parameters of scenario
    
    # Fetch data
    x_train = my_scenario.X_TRAIN
    y_train = my_scenario.Y_TRAIN
    x_test = my_scenario.X_TEST
    y_test = my_scenario.Y_TEST
    print('\n### Data loaded: ', my_scenario.DATASET_NAME)
    print('- ' + str(len(x_train)) + ' train data with ' + str(len(y_train)) + ' labels')
    print('- ' + str(len(x_test)) + ' test data ' + str(len(y_test)) + ' labels')
    
    # Fetch the number of independant nodes
    nodes_count = my_scenario.NODES_COUNT
    print('\n### Description of data scenario configured:')
    print('- Number of nodes defined:', nodes_count)
    
    
    #%% Configure the desired splitting scenario - Datasets sizes
    # Should the nodes receive an equivalent amount of samples each...
    # ... or receive different amounts?
    
    # Fetch the percentages of samples per node and control its coherence
    amounts_per_node = my_scenario.AMOUNTS_PER_NODE
    assert(len(amounts_per_node) == nodes_count)
    assert(np.sum(amounts_per_node) == 1)
    
    # Then we parameterize this via the splitting_indices to be passed to np.split
    splitting_indices = np.empty((nodes_count-1,))
    splitting_indices[0] = amounts_per_node[0]
    for i in range(nodes_count-2):
        splitting_indices[i+1] = splitting_indices[i] + amounts_per_node[i+1]
    splitting_indices_train = (splitting_indices * len(y_train)).astype(int)
    splitting_indices_test = (splitting_indices * len(y_test)).astype(int)
    print('- Splitting indices defined (for train data):', splitting_indices_train)
    
    
    #%% Configure the desired splitting scenario - Overlapping or distinct samples
    # Should the nodes receive data from distinct regions of space...
    # ... or from the complete universe?
    
    # Fetch the type of distribution chosen
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
    # ... data amount per node and overlap/distinct distribution and...
    # ... test data distribution
    
    # Split data between nodes
    train_idx_idx_list = np.split(train_idx, splitting_indices_train)
    test_idx_idx_list = np.split(test_idx, splitting_indices_test)
    
    # Fetch test data distribution scenario
    testset_option = my_scenario.TESTSET_OPTION
    print('- Test data distribution scenario chosen:', testset_option)
  
    # Populate nodes
    node_list = []
    for train_idx, test_idx in zip(train_idx_idx_list, test_idx_idx_list):
        
        # Train data
        x_node_train = x_train[train_idx, :]
        y_node_train = y_train[train_idx,]
        
        # Test data
        if testset_option == 'Split':
            x_node_test = x_test[test_idx]
            y_node_test = y_test[test_idx]
        elif testset_option == 'Global':
            x_node_test = my_scenario.X_TEST
            y_node_test = my_scenario.Y_TEST
        else:
            print('This testset_option scenario is not recognized. Quitting with quit()')
            quit()
        
        node = Node(x_node_train, x_node_test, y_node_train, y_node_test)
        node_list.append(node)
    
    # Check coherence of node_list versus nodes_count   
    assert(len(node_list) == nodes_count)
    
    # Print and plot for controlling
    for node_index, node in enumerate(node_list):
        print('\nNode #' + str(node_index) + ':')
        print('First 10 elements of y_train:' + str(node.y_train[:10]))
#        print('First image:')
#        first_image = node.x_train[0,:,:]
#        plt.gray()
#        plt.imshow(first_image)
#        plt.show()
        print('Last 10 elements of y_train:' + str(node.y_train[-10:]))
#        print('Last image:')
#        last_image = node.x_train[-1,:,:]
#        plt.gray()
#        plt.imshow(last_image)
#        plt.show()
        print('Number of test samples: ' + str(len(node.x_test)))
        
    # Return list of nodes
    return node_list