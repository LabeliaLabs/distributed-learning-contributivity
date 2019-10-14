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

contrib_name = 'shapley_values'
shapley_contrib = contributivity.Contributivity(contrib_name)

start = timer()
shapley_contrib.contributivity_scores = contributivity_measures.compute_SV(preprocessed_node_list)
end = timer()
shapley_contrib.computation_time = np.round(end - start)
print(shapley_contrib.computation_time , 's to compute', contrib_name)
print('\n### Shapley value for each node: ', [ '%.3f' % elem for elem in shapley_contrib.contributivity_scores])

      
#%% Contributivity 1 : performance scores of models trained independently on each node

start = timer()
perf_scores = contributivity_measures.compute_independent_scores(preprocessed_node_list, fl_score)
end = timer()
time_taken = np.round(end - start)

print('\n### Independent perf. scores (raw and normalized additively):')
print('- raw: ', [ '%.3f' % elem for elem in perf_scores[0] ] )
print('- normalized additively: ', [ '%.3f' % elem for elem in perf_scores[1] ])
print('- (reminder: fl_score ' + ('%.3f' % fl_score) + ')')

      
#%% Save results to file

my_basic_scenario.to_file()