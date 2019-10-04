# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:02:13 2019

A script to run a FL training and contributivity measurement simulations

@author: bowni
"""

from __future__ import print_function

import my_scenario
import data_splitting
import fl_train_eval
import contributivity_measures


#%% Fetch data splitting scenario
node_list = data_splitting.process_data_splitting_scenario()
nodes_count = my_scenario.NODES_COUNT

#%% Preprocess data
preprocessed_node_list = fl_train_eval.preprocess_node_list(node_list)

#%% Train and eval according to scenario
# score = fl_train_eval.fl_train(preprocessed_node_list)
# print('Score:', score)

#%% Contributivity measurement
sv_0 = contributivity_measures.compute_SV_3partners(0, preprocessed_node_list)
# sv_1 = contributivity_measures.compute_SV_3partners(1, preprocessed_node_list)
# sv_2 = contributivity_measures.compute_SV_3partners(2, preprocessed_node_list)
print('\nShapley Value (for 3 partners only):')
print('sv_0: ', sv_0)
print('sv_1: ', sv_1)
print('sv_2: ', sv_2)