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


#%% Constants

epochs = 3
nodes_count = 3


#%% Preprocess data

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# merge everything, we will do our own train/test split later --> Do we want to do that ??
#x = np.concatenate([x_train, x_test])
#y = np.concatenate([y_train, y_test])

# We want the same data count for every node --> Do we want to do that ??
#data_count = len(x) - (len(x) % nodes_count)
#assert(data_count % nodes_count == 0)

# Shuffle train data
train_idx = np.arange(len(y_train))
np.random.shuffle(train_idx)
train_idx_idx_list = np.array_split(train_idx, nodes_count)

# Shuffle test data
test_idx = np.arange(len(y_test))
np.random.shuffle(test_idx)
test_idx_idx_list = np.array_split(test_idx, nodes_count)

# Split data between nodes
x_node_list = []
y_node_list = []

for train_idx, test_idx in zip(train_idx_idx_list, test_idx_idx_list):
    
    x_node_train = x_train[train_idx, :]
    x_node_test = x_test[test_idx]
    x_node_list.append((x_train, x_test))
    
    y_node_train = y_train[train_idx,]
    y_node_test = y_test[test_idx]
    y_node_list.append((y_train, y_test))

assert(len(x_node_list) == nodes_count)
assert(len(y_node_list) == nodes_count)


# Now that the data split between the node has been done, we consider that the data
# stay on each node

#%% For each node preprocess data
    
for i in range(nodes_count):
    
    # Preprocess input (x) data
    (x_node_train, x_node_test) = x_node_list[i]
    x_node_train = utils.preprocess_input(x_node_train)
    x_node_test = utils.preprocess_input(x_node_test)
    
    # Preprocess labels (y) data
    (y_node_train, y_node_test) = y_node_list[i]
    y_node_train = keras.utils.to_categorical(y_node_train, constants.NUM_CLASSES)
    y_node_test = keras.utils.to_categorical(y_node_test, constants.NUM_CLASSES)
    
    # Crete validation dataset
    x_node_train, x_node_val, y_node_train, y_node_val = train_test_split(x_node_train, y_node_train, test_size = 0.1, random_state=42)
   
    x_node_list[i] = (x_node_train, x_node_val, x_node_test)
    y_node_list[i] = (y_node_train, y_node_val, y_node_test)


#%% For each node build model

model_list = []
for i in range(nodes_count):
    
    model = utils.generate_new_cnn_model()
    model_list.append(model)

print(model_list[0].summary())


#%% Federated training

# Training phase
for node_index in range(nodes_count):
    
    print('\nTraning on node '+ str(node_index))
    
    node_x = x_node_list[node_index]
    node_y = y_node_list[node_index]
    node_model = model_list[node_index]
    
     
    (x_node_train, x_node_val, x_node_test) = node_x
    (y_node_train, y_node_val, y_node_test) = node_y
    
    
    # At each step, train on whole local data set
    history = node_model.fit(x_node_train, y_node_train,
              batch_size=constants.BATCH_SIZE,
              epochs=1,
              verbose=1,
              validation_data=(x_node_val, y_node_val))


# Aggregating phase : averaging the weights

# Evaluation: evaluate data on every node test set

# Compute contributivykeras score 