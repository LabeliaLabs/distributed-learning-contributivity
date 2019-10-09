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

import scenario
import data_splitting
import fl_train_eval
import contributivity_measures


#%% Fetch data splitting scenario

my_basic_scenario = scenario.Scenario()
node_list = data_splitting.process_data_splitting_scenario(my_basic_scenario)


#%% Preprocess data

preprocessed_node_list = fl_train_eval.preprocess_node_list(node_list)


#%% Train and eval according to scenario
# fl_train_eval.fl_train_score(preprocessed_node_list)
# fl_train_eval.single_train_score(preprocessed_node_list[0])


#%% Get performance scores of models trained independently on each node
list_perf_scores = contributivity_measures.compute_independent_scores(preprocessed_node_list, 0.9)
print('\nIndependent perf. scores (raw and softmaxed * target):')
print('- raw: ', list_perf_scores[0])
print('- softmaxed * target: ', list_perf_scores[1])


#%% Contributivity measurement
list_shapley_value = contributivity_measures.compute_SV(preprocessed_node_list)
print('\nShapley value for each node: ', list_shapley_value)


#%% Save results
my_basic_scenario.to_file()