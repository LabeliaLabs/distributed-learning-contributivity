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
import fl_training
import contributivity_measures

from timeit import default_timer as timer
import numpy as np


#%% Create scenarii

# Create a custom scenario and comment the main scenario parameters (see scenario.py for more comments)
my_custom_scenario = scenario.Scenario(is_quick_demo=True)
my_custom_scenario.nodes_count = 3 # Number of nodes in the collaborative ML project simulated
my_custom_scenario.amounts_per_node = [0.20, 0.30, 0.5] # Percentages of the data samples for each node
my_custom_scenario.samples_split_option = 'Stratified' # If data are split randomly between nodes or stratified to be distinct (toggle between 'Random' and 'Stratified')
my_custom_scenario.testset_option = 'Centralised' # If test data are distributed between nodes or stays a central testset (toggle between 'Centralised' and 'Distributed')

# Gather scenarii in a list
scenarii_list = []
# scenarii_list.append(my_default_scenario)
scenarii_list.append(my_custom_scenario)


#%% Run the scenarii

for current_scenario in scenarii_list:
    
    current_scenario.split_data()
    
    current_scenario.nodes_list = fl_training.preprocess_node_list(current_scenario.nodes_list)
    
    
    #%% Train and eval on all nodes according to scenario
    
    fl_score = fl_training.compute_test_score(current_scenario.nodes_list)
    
    
    #%% Contributivity 1: Baseline contributivity measurement (Shapley Value)
    
    shapley_contrib = contributivity.Contributivity('Shapley values')
    
    start = timer()
    shapley_contrib.contributivity_scores = contributivity_measures.compute_SV(current_scenario.nodes_list)
    end = timer()
    
    shapley_contrib.computation_time = np.round(end - start)
    
    current_scenario.append_contributivity(shapley_contrib)
    print(shapley_contrib)
    
          
    #%% Contributivity 2: Performance scores of models trained independently on each node
    
    independant_raw_contrib = contributivity.Contributivity('Independant scores raw')
    independant_additiv_contrib = contributivity.Contributivity('Independant scores additiv')
    
    start = timer()
    scores = contributivity_measures.compute_independent_scores(current_scenario.node_list, fl_score)
    end = timer()
    
    independant_computation_time = np.round(end - start)
    independant_raw_contrib.computation_time = independant_computation_time
    independant_additiv_contrib.computation_time = independant_computation_time
    
    # TODO use dict instead of 0/1 indexes
    independant_raw_contrib.contributivity_scores = scores[0]
    independant_additiv_contrib.contributivity_scores = scores[1]
    
    current_scenario.append_contributivity(independant_raw_contrib)
    current_scenario.append_contributivity(independant_additiv_contrib)
    print(independant_raw_contrib)
    print('')
    print(independant_additiv_contrib)
    
          
    #%% Save results to file
    
    current_scenario.to_file()