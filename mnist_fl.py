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


#%% ML Constants

batch_size = 128
epochs = 6

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)


#%% Federated learning constants

nodes_count = 3


#%% Preprocess data

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# merge everything, we will do our own train/test split later --> Do we want to do that ??
x = np.concatenate([x_train, x_test])
y = np.concatenate([y_train, y_test])

# We want the same data count for every node
data_count = len(x) - (len(x) % nodes_count)
assert(data_count % nodes_count == 0)

# Shuffle data
idx = np.arange(data_count)
np.random.shuffle(idx)
idx_list = np.split(idx, 3)

# Split data between nodes
x_list = []
y_list = []
for idx in idx_list:
    x_list.append(x[idx,:])
    y_list.append(y[idx])

# For each node, train_test_split


#%% Build model

model = utils.generate_new_cnn_model(batch_size=batch_size, epochs=epochs)
print(model.summary())


#%% Federated training

# Start training :
# At each step only one batch (defined by batch size) or whole local data set ?
# What kind of aggregating function ? Averaging the weights ?

# Evaluation: evaluate data on every node test set

# Compute contributivy score 