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
parameters_dict = my_scenario.PARAMETERS_DICT
node_list = data_splitting.process_data_splitting_scenario(**parameters_dict)


#%% Preprocess data
preprocessed_node_list = fl_train_eval.preprocess_node_list(node_list)


#%% Train and eval according to scenario
#fl_train_eval.fl_train_score(preprocessed_node_list)
# fl_train_eval.single_train_score(preprocessed_node_list[0])


#%% Contributivity measurement
list_shapley_value = contributivity_measures.compute_SV(preprocessed_node_list)
print('\nShapley value for each node: ', list_shapley_value)
