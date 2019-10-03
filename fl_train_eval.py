# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:33:15 2019

@author: RGOUSSAULT
inspired from: https://keras.io/examples/mnist_cnn/
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

import utils
import constants
import my_scenario
import data_splitting
from node import Node

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


#%% Fetch data splitting scenario

nodes_count = my_scenario.NODES_COUNT
node_list = data_splitting.process_data_splitting_scenario()


#%% Pre-process data for ML training

def preprocess_node_list(node_list):
    """Return node_list preprocessed"""

    for node_index, node in enumerate(node_list):
        
        # Preprocess input (x) data
        node.preprocess_data()
        
        # Crete validation dataset
        x_node_train, x_node_val, y_node_train, y_node_val = train_test_split(node.x_train, node.y_train, test_size = 0.1, random_state=42)
        node.x_train = x_node_train
        node.x_val = x_node_val
        node.y_train = y_node_train
        node.y_val = y_node_val
        
        # Align variable names
        x_node_test = node.x_test
        y_node_test = node.y_test
        
        print(str(len(x_node_train)) + ' train data for node ' + str(node_index))
        print(str(len(x_node_val)) + ' val data for node ' + str(node_index))
        print(str(len(x_node_test)) + ' test data for node ' + str(node_index))   
        
    return node_list


#%% Federated training
        
def fl_train(nodes_count, node_list):
    """Return a final aggregated model trained in a federated way on each node"""

    model_list = [None] * nodes_count
    epochs = 2
    score_matrix = np.zeros(shape=(epochs, nodes_count))
    val_acc_epoch = []
    acc_epoch = []
    
    for epoch in range(epochs):
    
        print('\n=============================================')
        print('Epoch #' + str(epoch + 1) + ' out of ' + str(epochs) + ' total epochs')
        is_first_epoch = epoch == 0
        
        
        # Aggregation phase
        if is_first_epoch:
            # First epoch
            print('First epoch, generate model from scratch')
            
        else:
            print('Aggregating models weights to build a new model')
            # Aggregating phase : averaging the weights
            weights = [model.get_weights() for model in model_list]
            new_weights = list()
            
            # TODO : make this clearer
            for weights_list_tuple in zip(*weights):
                new_weights.append(
                    [np.array(weights_).mean(axis=0)\
                        for weights_ in zip(*weights_list_tuple)])    
            
    
        # Training phase
        val_acc_list = []
        acc_list = []
        for node_index, node in enumerate(node_list):
            
            print('\nTraining on node '+ str(node_index))
            node_model = utils.generate_new_cnn_model()
            
            # Model weights are the averaged weights
            if not is_first_epoch:
                node_model.set_weights(new_weights)
                node_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
            
            # Train on whole node local data set
            history = node_model.fit(node.x_train, node.y_train,
                      batch_size=constants.BATCH_SIZE,
                      epochs=1,
                      verbose=1,
                      validation_data=(node.x_val, node.y_val))
            
            val_acc_list.append(history.history['val_acc'])
            acc_list.append(history.history['acc'])
            
            model_list[node_index] = node_model
        
        val_acc_epoch.append(np.median(val_acc_list))
        acc_epoch.append(np.median(acc_list))
    
    
    # Final aggregation : averaging the weights
    weights = [model.get_weights() for model in model_list]
    new_weights = list()
    for weights_list_tuple in zip(*weights):
        new_weights.append(
            [np.array(weights_).mean(axis=0)\
                for weights_ in zip(*weights_list_tuple)])
    
    final_model = utils.generate_new_cnn_model()
    final_model.set_weights(new_weights)
    final_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # Evaluate model
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = utils.preprocess_input(x_test)
    y_test = keras.utils.to_categorical(y_test, constants.NUM_CLASSES)
    model_eval_score = final_model.evaluate(x_test, y_test, batch_size=constants.BATCH_SIZE,
                         verbose=1)
    print('\nModel metrics names: ', final_model.metrics_names)
    
    # Return 
    return model_eval_score
