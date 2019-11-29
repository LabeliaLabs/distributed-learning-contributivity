# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:09:07 2019

This processes the split of data among nodes according to scenario passed as argument

@author: @bowni
"""

from __future__ import print_function

import numpy as np

from node import Node

# TODO this method should propably be in the Scenario class
def process_data_splitting_scenario(scenario):
    """Return a list of nodes populated with their train and test data (not pre-processed)"""

    #%% Fetch parameters of scenario
    
    x_train = scenario.x_train
    y_train = scenario.y_train
    x_test = scenario.x_test
    y_test = scenario.y_test
    
    # Describe data
    print('\n### Data loaded: ', scenario.dataset_name)
    print('- ' + str(len(x_train)) + ' train data with ' + str(len(y_train)) + ' labels')
    print('- ' + str(len(x_test)) + ' test data ' + str(len(y_test)) + ' labels')
    
    # Describe number of independant nodes
    print('\n### Description of data scenario configured:')
    print('- Number of nodes defined:', scenario.nodes_count)
    
    
    #%% Configure the desired splitting scenario - Datasets sizes
    # Should the nodes receive an equivalent amount of samples each...
    # ... or receive different amounts?
    
    # Check the percentages of samples per node and control its coherence
    assert(len(scenario.amounts_per_node) == scenario.nodes_count)
    assert(np.sum(scenario.amounts_per_node) == 1)
    
    # Then we parameterize this via the splitting_indices to be passed to np.split
    # This is to transform the % from my_scenario into indices where to split the data
    splitting_indices = np.empty((scenario.nodes_count-1,))
    splitting_indices[0] = scenario.amounts_per_node[0]
    for i in range(scenario.nodes_count-2):
        splitting_indices[i+1] = splitting_indices[i] + scenario.amounts_per_node[i+1]
    splitting_indices_train = (splitting_indices * len(y_train)).astype(int)
    splitting_indices_test = (splitting_indices * len(y_test)).astype(int)
    # print('- Splitting indices defined (for train data):', splitting_indices_train) # VERBOSE
    
    
    #%% Configure the desired data distribution scenario
    
    # Describe the type of distribution chosen
    print('- Data distribution scenario chosen:', scenario.samples_split_option)
    
    # Create a list of indexes of the samples
    train_idx = np.arange(len(y_train))
    test_idx = np.arange(len(y_test))
    
    # In the 'Stratified' scenario we sort MNIST by labels
    if scenario.samples_split_option == 'Stratified':
        
        # Sort MNIST by labels
        y_sorted_idx = y_train.argsort()
        y_train = y_train[y_sorted_idx]
        x_train = x_train[y_sorted_idx]
        
#        # Print and plot for controlling
#        print('\nFirst 10 elements of y_train:' + str(y_train[:10]))
#        print('First image:')
#        first_image = x_train[0,:,:]
#        plt.gray()
#        plt.imshow(first_image)
#        plt.show()
#        print('Last 10 elements of y_train' + str(y_train[-10:]))
#        print('Last image:')
#        last_image = x_train[-1,:,:]
#        plt.gray()
#        plt.imshow(last_image)
#        plt.show()
        
    # In the 'Random' scenario we shuffle randomly the indexes
    elif scenario.samples_split_option == 'Random':
        np.random.seed(42)
        np.random.shuffle(train_idx)
    
    # If neither 'Stratified' nor 'Random', we raise an exception
    else:
        raise NameError('This samples_split_option scenario [' + scenario.samples_split_option + '] is not recognized')
        
        
    #%% Do the splitting among nodes according to desired scenarios
    
    # Split data between nodes
    train_idx_idx_list = np.split(train_idx, splitting_indices_train)
    test_idx_idx_list = np.split(test_idx, splitting_indices_test)
    
    # Describe test data distribution scenario
    print('- Test data distribution scenario chosen:', scenario.testset_option)
  
    # Populate nodes
    node_list = []
    node_id = 0
    for train_idx, test_idx in zip(train_idx_idx_list, test_idx_idx_list):
        
        # Train data
        x_node_train = x_train[train_idx, :]
        y_node_train = y_train[train_idx,]
        
        # Test data
        if scenario.testset_option == 'Distributed':
            x_node_test = x_test[test_idx]
            y_node_test = y_test[test_idx]
        elif scenario.testset_option == 'Centralised':
            x_node_test = x_test
            y_node_test = y_test
        else:
            raise NameError('This testset_option [' + scenario.testset_option + '] scenario is not recognized')
            
        node = Node(x_node_train, x_node_test, y_node_train, y_node_test, 
                    str(node_id))
        node_list.append(node)
        node_id += 1
    
    # Check coherence of node_list versus nodes_count   
    assert(len(node_list) == scenario.nodes_count)
    
    # Print and plot for controlling
    print('\n### Splitting data among nodes:')
    for node_index, node in enumerate(node_list):
        print('- Node #' + str(node_index) + ':')
        print('  - Number of samples:' + str(len(node.x_train)) + ' train, ' + str(len(node.x_val)) + ' val, ' + str(len(node.x_test)) + ' test')
        print('  - y_train first 10 values:' + str(node.y_train[:10]))
        print('  - y_train last 10 values:' + str(node.y_train[-10:]))
#        print('First image:')
#        first_image = node.x_train[0,:,:]
#        plt.gray()
#        plt.imshow(first_image)
#        plt.show()
#        print('Last image:')
#        last_image = node.x_train[-1,:,:]
#        plt.gray()
#        plt.imshow(last_image)
#        plt.show()
        
    # Return list of nodes
    return node_list