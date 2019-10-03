# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:02:13 2019

A script to run a FL training and contributivity measurement simulation

@author: bowni
"""

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

from itertools import combinations

import constants
import utils
import my_scenario
import data_splitting
import fl_train_eval
import contributivity_measures


#%% Fetch data splitting scenario
nodes_count = my_scenario.NODES_COUNT
node_list = data_splitting.process_data_splitting_scenario()
# Dirty trick for quicker test: nodes_count_bis = nodes_count - 1
# Dirty trick for quicker test: del node_list[3]

#%% Preprocess data
preprocessed_node_list = fl_train_eval.preprocess_node_list(node_list)

#%% Train and eval according to scenario
score = fl_train_eval.fl_train(nodes_count, preprocessed_node_list)
print('Score:', score)

#%% Contributivity measurement
    # TODO