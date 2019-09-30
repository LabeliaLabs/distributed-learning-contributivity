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
from node import Node

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#%% Constants

nodes_count = 10

#%% Preprocess data

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(str(len(x_train)) + ' train data')
print(str(len(x_test)) + ' test data\n')

# Shuffle train data
train_idx = np.arange(len(y_train))
np.random.seed(42)
np.random.shuffle(train_idx)
train_idx_idx_list = np.array_split(train_idx, nodes_count)

# Shuffle test data
test_idx = np.arange(len(y_test))
np.random.seed(42)
np.random.shuffle(test_idx)
test_idx_idx_list = np.array_split(test_idx, nodes_count)

# Split data between nodes
node_list = []

for train_idx, test_idx in zip(train_idx_idx_list, test_idx_idx_list):
    
    x_node_train = x_train[train_idx, :]
    x_node_test = x_test[test_idx]
    y_node_train = y_train[train_idx,]
    y_node_test = y_test[test_idx]
    
    node = Node(x_node_train, x_node_test, y_node_train, y_node_test)
    node_list.append(node)
    
assert(len(node_list) == nodes_count)


# Now that the data split between the node has been done, we consider that the data
# stay on each node

#%% For each node preprocess data
    
for node_index, node in enumerate(node_list):
    
    # Preprocess input (x) data
    node.preprocess_data()
    
    # Crete validation dataset
    x_node_train, x_node_val, y_node_train, y_node_val = train_test_split(node.x_train, node.y_train, test_size = 0.1, random_state=42)
    node.x_train = x_node_train
    node.x_val = x_node_val
    node.y_train = y_node_train
    node.y_val = y_node_val
    
    print(str(len(x_node_train)) + ' train data for node ' + str(node_index))
    print(str(len(x_node_val)) + ' val data for node ' + str(node_index))
    print(str(len(x_node_test)) + ' test data for node ' + str(node_index))   


#%% For each node build model

model_list = []
for i in range(nodes_count):
    
    model = utils.generate_new_cnn_model()
    model_list.append(model)

#print(model_list[0].summary())


#%% Federated training

epochs = 30
score_matrix = np.zeros(shape=(epochs, nodes_count))
val_acc_epoch = []
acc_epoch = []

for epoch in range(epochs):

    print('\n=============================================')
    print('Epoch ' + str(epoch))
    is_first_epoch = epoch == 0
    
    
    # Aggregation phase
    if is_first_epoch:
        # First epoch
        print('First epoch, generate model from scratch')
        base_model = utils.generate_new_cnn_model()
        
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

    # TODO Evaluation phase: evaluate data on every node test set

# TODO Compute contributivity score 
    
    
#%% Plot history


plt.figure()
plt.plot(acc_epoch,'+-')
plt.plot(val_acc_epoch,'+-')
plt.title('Accuracy')
plt.ylabel('Acc')
plt.xlabel('Epoch')
plt.legend(('Train', 'Val'))
plt.show()