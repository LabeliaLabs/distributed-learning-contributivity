# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:02:13 2019

A script to configure and run simulations of:
    - splitting data among different nodes to mock a multi-partner ML project
    - train a model
    - measure contributivity of each node to the model performance

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
fl_train_eval.fl_train(preprocessed_node_list)
# fl_train_eval.single_train(preprocessed_node_list[0])


#%% Contributivity measurement
sv_0 = contributivity_measures.compute_SV_3partners(0, preprocessed_node_list)
# sv_1 = contributivity_measures.compute_SV_3partners(1, preprocessed_node_list)
# sv_2 = contributivity_measures.compute_SV_3partners(2, preprocessed_node_list)
print('\nShapley Value (for 3 partners only):')
print('sv_0: ', sv_0)
print('sv_1: ', sv_1)
print('sv_2: ', sv_2)