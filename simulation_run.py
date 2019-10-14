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
import contributivity
import data_splitting
import fl_train_eval
import contributivity_measures
from timeit import default_timer as timer
import numpy as np


#%% Fetch data splitting scenario

my_basic_scenario = scenario.Scenario()
node_list = data_splitting.process_data_splitting_scenario(my_basic_scenario)


#%% Preprocess data for compatibility with keras CNN models

preprocessed_node_list = fl_train_eval.preprocess_node_list(node_list)


#%% Train and eval on all nodes according to scenario

fl_score = fl_train_eval.fl_train_score(preprocessed_node_list)[1]


#%% Contributivity 1: Baseline contributivity measurement (Shapley Value)

shapley_contrib = contributivity.Contributivity('Shapley values')

start = timer()
shapley_contrib.contributivity_scores = contributivity_measures.compute_SV(preprocessed_node_list)
end = timer()

shapley_contrib.computation_time = np.round(end - start)

my_basic_scenario.append_contributivity(shapley_contrib)
print(shapley_contrib)

      
#%% Contributivity 1 : performance scores of models trained independently on each node

independant_raw_contrib = contributivity.Contributivity('Independant scores raw')
independant_additiv_contrib = contributivity.Contributivity('Independant scores additiv')

start = timer()
scores = contributivity_measures.compute_independent_scores(preprocessed_node_list, fl_score)
end = timer()

independant_computation_time = np.round(end - start)
independant_raw_contrib.computation_time = independant_computation_time
independant_additiv_contrib.computation_time = independant_computation_time

# TODO use dict instead of 0/1 indexes
independant_raw_contrib.contributivity_scores = scores[0]
independant_additiv_contrib.contributivity_scores = scores[1]

my_basic_scenario.append_contributivity(independant_raw_contrib)
my_basic_scenario.append_contributivity(independant_additiv_contrib)
print(independant_raw_contrib)
print('')
print(independant_additiv_contrib)

      
#%% Save results to file

my_basic_scenario.to_file()