# -*- coding: utf-8 -*-
"""
A script to configure and run simulations of:
    - splitting data among different nodes to mock a multi-partner ML project
    - train a model across multiple nodes
    - measure contributivity of each node to the model performance
"""

from __future__ import print_function

import scenario
import contributivity
import fl_training
import contributivity_measures

from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

#%% Create scenarii

# Create a custom scenario and comment the main scenario parameters (see scenario.py for more comments)
my_custom_scenario = scenario.Scenario(is_quick_demo=True)
my_custom_scenario.nodes_count = 3 # Number of nodes in the collaborative ML project simulated
my_custom_scenario.amounts_per_node = [0.20, 0.30, 0.5] # Percentages of the data samples for each node
my_custom_scenario.samples_split_option = 'Random' # If data are split randomly between nodes or stratified to be distinct (toggle between 'Random' and 'Stratified')
my_custom_scenario.testset_option = 'Centralised' # If test data are distributed between nodes or stays a central testset (toggle between 'Centralised' and 'Distributed')

# Gather scenarii in a list
scenarii_list = []
# scenarii_list.append(my_default_scenario)
scenarii_list.append(my_custom_scenario)




#%% Run the scenarii

for current_scenario in scenarii_list:

    current_scenario.split_data()
    current_scenario.plot_data_distribution()

    # Pre-process successively train data, early stopping validation data, test data
    current_scenario.node_list = fl_training.preprocess_node_list(current_scenario.node_list)
    current_scenario.x_esval, current_scenario.y_esval = fl_training.preprocess_test_data(current_scenario.x_esval, current_scenario.y_esval)
    current_scenario.x_test, current_scenario.y_test = fl_training.preprocess_test_data(current_scenario.x_test, current_scenario.y_test)


    #%% Train and eval on all nodes according to scenario

    is_save_fig = True
    current_scenario.federated_test_score = fl_training.compute_test_score_with_scenario(current_scenario, is_save_fig)


    #%% Contributivity 1: Baseline contributivity measurement (Shapley Value)

    start = timer()
    (contributivity_scores,scores_var) = contributivity_measures.compute_SV(current_scenario.node_list, current_scenario.epoch_count, current_scenario.x_esval, current_scenario.y_esval, current_scenario.x_test, current_scenario.y_test)
    end = timer()

    shapley_contrib = contributivity.Contributivity('Shapley values',contributivity_scores,scores_var,np.round(end - start))


    current_scenario.append_contributivity(shapley_contrib)
    print('\n## Evaluating contributivity with Shapley:')
    print(shapley_contrib)


    #%% Contributivity 2: Performance scores of models trained independently on each node



    start = timer()
    scores = contributivity_measures.compute_independent_scores(current_scenario.node_list, current_scenario.epoch_count, current_scenario.federated_test_score)
    end = timer()
    # TODO use dict instead of 0/1 indexes
    independant_raw_contrib = contributivity.Contributivity('Independant scores raw',scores[0],np.repeat (0.0,len(scores[0]) ) )
    independant_additiv_contrib = contributivity.Contributivity('Independant scores additive',scores[1],np.repeat (0.0,len(scores[1]) ) )

    independant_computation_time = np.round(end - start)
    independant_raw_contrib.computation_time = independant_computation_time
    independant_additiv_contrib.computation_time = independant_computation_time



    current_scenario.append_contributivity(independant_raw_contrib)
    current_scenario.append_contributivity(independant_additiv_contrib)
    print('\n## Evaluating contributivity with independent single partner models:')
    print(independant_raw_contrib)
    print(independant_additiv_contrib)


    #%% Save results to file

    current_scenario.to_file()