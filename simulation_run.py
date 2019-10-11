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


#%% Preprocess data for compatibility with keras CNN models
preprocessed_node_list = fl_train_eval.preprocess_node_list(node_list)


#%% Train and eval on all nodes according to scenario
fl_score = fl_train_eval.fl_train_score(preprocessed_node_list)[1]


#%% Get performance scores of models trained independently on each node
list_perf_scores = contributivity_measures.compute_independent_scores(preprocessed_node_list, fl_score)
print('\nIndependent perf. scores (raw and normalized additively):')
print('- raw: ', [ '%.3f' % elem for elem in list_perf_scores[0] ] )
print('- normalized additively: ', [ '%.3f' % elem for elem in list_perf_scores[1] ])
print('- (reminder: fl_score ' + ('%.3f' % fl_score) + ')')


#%% Baseline contributivity measurement (Shapley Value)
list_shapley_value = contributivity_measures.compute_SV(preprocessed_node_list)
print('\nShapley value for each node: ', list_shapley_value)
