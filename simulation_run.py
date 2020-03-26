# -*- coding: utf-8 -*-
"""
A script to configure and run simulations of:
    - splitting data among different nodes to mock a multi-partner ML project
    - train a model across multiple nodes
    - measure contributivity of each node to the model performance
"""

from __future__ import print_function

# GPU config
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
from timeit import default_timer as timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import contributivity
import contributivity_measures
import fl_training
import scenario

# gpus = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(gpus[0], True)

plt.close("all")

#%% Create scenarii
scenarii_list = []
IS_QUICK_DEMO = True

# Create a custom scenario and comment the main scenario parameters (see scenario.py for more comments)
my_custom_scenario = scenario.Scenario(is_quick_demo=IS_QUICK_DEMO)
my_custom_scenario.nodes_count = 3 # Number of nodes in the collaborative ML project simulated
my_custom_scenario.amounts_per_node = [0.33, 0.33, 0.34] # Percentages of the data samples for each node
my_custom_scenario.samples_split_option = 'Stratified' # If data are split randomly between nodes or stratified to be distinct (toggle between 'Random' and 'Stratified')
my_custom_scenario.testset_option = 'Distributed' # If test data are distributed between nodes or stays a central testset (toggle between 'Centralised' and 'Distributed')
my_custom_scenario.corrupted_nodes = ["not-corrupted"]*my_custom_scenario.nodes_count # no corrupted node
scenarii_list.append(my_custom_scenario)


# my_custom_scenario = scenario.Scenario(is_quick_demo=IS_QUICK_DEMO)
# my_custom_scenario.nodes_count = 3 # Number of nodes in the collaborative ML project simulated
# my_custom_scenario.amounts_per_node = [0.33, 0.33, 0.34] # Percentages of the data samples for each node
# my_custom_scenario.samples_split_option = 'Random' # If data are split randomly between nodes or stratified to be distinct (toggle between 'Random' and 'Stratified')
# my_custom_scenario.testset_option = 'Centralised' # If test data are distributed between nodes or stays a central testset (toggle between 'Centralised' and 'Distributed')
# my_custom_scenario.corrupted_nodes = ["corrupted","not-corrupted","not-corrupted"] # First node is corrupted (label are wrong)
# scenarii_list.append(my_custom_scenario)
#
#
# my_custom_scenario = scenario.Scenario(is_quick_demo=IS_QUICK_DEMO)
# my_custom_scenario.nodes_count = 3 # Number of nodes in the collaborative ML project simulated
# my_custom_scenario.amounts_per_node = [0.33, 0.33, 0.34] # Percentages of the data samples for each node
# my_custom_scenario.samples_split_option = 'Random' # If data are split randomly between nodes or stratified to be distinct (toggle between 'Random' and 'Stratified')
# my_custom_scenario.testset_option = 'Centralised' # If test data are distributed between nodes or stays a central testset (toggle between 'Centralised' and 'Distributed')
# my_custom_scenario.corrupted_nodes = ["shuffled","not-corrupted","not-corrupted"] # First node is corrupted (label are wrong)
# scenarii_list.append(my_custom_scenario)
#
# my_custom_scenario = scenario.Scenario(is_quick_demo=IS_QUICK_DEMO)
# my_custom_scenario.nodes_count = 3 # Number of nodes in the collaborative ML project simulated
# my_custom_scenario.amounts_per_node = [0.33, 0.33, 0.34] # Percentages of the data samples for each node
# my_custom_scenario.samples_split_option = 'Stratified' # If data are split randomly between nodes or stratified to be distinct (toggle between 'Random' and 'Stratified')
# my_custom_scenario.testset_option = 'Centralised' # If test data are distributed between nodes or stays a central testset (toggle between 'Centralised' and 'Distributed')
# my_custom_scenario.corrupted_nodes = ["not-corrupted"]*my_custom_scenario.nodes_count # no corrupted node
# scenarii_list.append(my_custom_scenario)
#
# my_custom_scenario = scenario.Scenario(is_quick_demo=IS_QUICK_DEMO)
# my_custom_scenario.nodes_count = 3 # Number of nodes in the collaborative ML project simulated
# my_custom_scenario.amounts_per_node = [0.2, 0.2, 0.6] # Percentages of the data samples for each node
# my_custom_scenario.samples_split_option = 'Random' # If data are split randomly between nodes or stratified to be distinct (toggle between 'Random' and 'Stratified')
# my_custom_scenario.testset_option = 'Centralised' # If test data are distributed between nodes or stays a central testset (toggle between 'Centralised' and 'Distributed')
# my_custom_scenario.corrupted_nodes = ["not-corrupted"]*my_custom_scenario.nodes_count # no corrupted node
# scenarii_list.append(my_custom_scenario)
#
# my_custom_scenario = scenario.Scenario(is_quick_demo=IS_QUICK_DEMO)
# my_custom_scenario.nodes_count = 3 # Number of nodes in the collaborative ML project simulated
# my_custom_scenario.amounts_per_node = [0.2, 0.2, 0.6] # Percentages of the data samples for each node
# my_custom_scenario.samples_split_option = 'Stratified' # If data are split randomly between nodes or stratified to be distinct (toggle between 'Random' and 'Stratified')
# my_custom_scenario.testset_option = 'Centralised' # If test data are distributed between nodes or stays a central testset (toggle between 'Centralised' and 'Distributed')
# my_custom_scenario.corrupted_nodes = ["not-corrupted"]*my_custom_scenario.nodes_count # no corrupted node
# scenarii_list.append(my_custom_scenario)
#
# my_custom_scenario = scenario.Scenario(is_quick_demo=IS_QUICK_DEMO)
# my_custom_scenario.nodes_count = 4 # Number of nodes in the collaborative ML project simulated
# my_custom_scenario.amounts_per_node = [0.25, 0.25, 0.25, 0.25] # Percentages of the data samples for each node
# my_custom_scenario.samples_split_option = 'Random' # If data are split randomly between nodes or stratified to be distinct (toggle between 'Random' and 'Stratified')
# my_custom_scenario.testset_option = 'Centralised' # If test data are distributed between nodes or stays a central testset (toggle between 'Centralised' and 'Distributed')
# my_custom_scenario.corrupted_nodes = ["not-corrupted"]*my_custom_scenario.nodes_count # no corrupted node
# scenarii_list.append(my_custom_scenario)
#
# my_custom_scenario = scenario.Scenario(is_quick_demo=IS_QUICK_DEMO)
# my_custom_scenario.nodes_count = 4 # Number of nodes in the collaborative ML project simulated
# my_custom_scenario.amounts_per_node = [0.25, 0.25, 0.25, 0.25] # Percentages of the data samples for each node
# my_custom_scenario.samples_split_option = 'Stratified' # If data are split randomly between nodes or stratified to be distinct (toggle between 'Random' and 'Stratified')
# my_custom_scenario.testset_option = 'Centralised' # If test data are distributed between nodes or stays a central testset (toggle between 'Centralised' and 'Distributed')
# my_custom_scenario.corrupted_nodes = ["not-corrupted"]*my_custom_scenario.nodes_count # no corrupted node
# scenarii_list.append(my_custom_scenario)
#
# my_custom_scenario = scenario.Scenario(is_quick_demo=IS_QUICK_DEMO)
# my_custom_scenario.nodes_count = 4 # Number of nodes in the collaborative ML project simulated
# my_custom_scenario.amounts_per_node = [0.1, 0.15, 0.3, 0.45] # Percentages of the data samples for each node
# my_custom_scenario.samples_split_option = 'Random' # If data are split randomly between nodes or stratified to be distinct (toggle between 'Random' and 'Stratified')
# my_custom_scenario.testset_option = 'Centralised' # If test data are distributed between nodes or stays a central testset (toggle between 'Centralised' and 'Distributed')
# my_custom_scenario.corrupted_nodes = ["not-corrupted"]*my_custom_scenario.nodes_count # no corrupted node
# scenarii_list.append(my_custom_scenario)
#
# my_custom_scenario = scenario.Scenario(is_quick_demo=IS_QUICK_DEMO)
# my_custom_scenario.nodes_count = 4 # Number of nodes in the collaborative ML project simulated
# my_custom_scenario.amounts_per_node = [0.1, 0.15, 0.3, 0.45] # Percentages of the data samples for each node
# my_custom_scenario.samples_split_option = 'Stratified' # If data are split randomly between nodes or stratified to be distinct (toggle between 'Random' and 'Stratified')
# my_custom_scenario.testset_option = 'Centralised' # If test data are distributed between nodes or stays a central testset (toggle between 'Centralised' and 'Distributed')
# my_custom_scenario.corrupted_nodes = ["not-corrupted"]*my_custom_scenario.nodes_count # no corrupted node
# scenarii_list.append(my_custom_scenario)
#
# my_custom_scenario = scenario.Scenario(is_quick_demo=IS_QUICK_DEMO)
# my_custom_scenario.nodes_count = 5 # Number of nodes in the collaborative ML project simulated
# my_custom_scenario.amounts_per_node = [0.2, 0.2, 0.2, 0.2, 0.2] # Percentages of the data samples for each node
# my_custom_scenario.samples_split_option = 'Random' # If data are split randomly between nodes or stratified to be distinct (toggle between 'Random' and 'Stratified')
# my_custom_scenario.testset_option = 'Centralised' # If test data are distributed between nodes or stays a central testset (toggle between 'Centralised' and 'Distributed')
# my_custom_scenario.corrupted_nodes = ["not-corrupted"]*my_custom_scenario.nodes_count # no corrupted node
# scenarii_list.append(my_custom_scenario)
#
# my_custom_scenario = scenario.Scenario(is_quick_demo=IS_QUICK_DEMO)
# my_custom_scenario.nodes_count = 5 # Number of nodes in the collaborative ML project simulated
# my_custom_scenario.amounts_per_node = [0.2, 0.2, 0.2, 0.2, 0.2] # Percentages of the data samples for each node
# my_custom_scenario.samples_split_option = 'Stratified' # If data are split randomly between nodes or stratified to be distinct (toggle between 'Random' and 'Stratified')
# my_custom_scenario.testset_option = 'Centralised' # If test data are distributed between nodes or stays a central testset (toggle between 'Centralised' and 'Distributed')
# my_custom_scenario.corrupted_nodes = ["not-corrupted"]*my_custom_scenario.nodes_count # no corrupted node
# scenarii_list.append(my_custom_scenario)
#
# my_custom_scenario = scenario.Scenario(is_quick_demo=IS_QUICK_DEMO)
# my_custom_scenario.nodes_count = 5 # Number of nodes in the collaborative ML project simulated
# my_custom_scenario.amounts_per_node = [0.1, 0.1, 0.2, 0.2, 0.4] # Percentages of the data samples for each node
# my_custom_scenario.samples_split_option = 'Random' # If data are split randomly between nodes or stratified to be distinct (toggle between 'Random' and 'Stratified')
# my_custom_scenario.testset_option = 'Centralised' # If test data are distributed between nodes or stays a central testset (toggle between 'Centralised' and 'Distributed')
# my_custom_scenario.corrupted_nodes = ["not-corrupted"]*my_custom_scenario.nodes_count # no corrupted node
# scenarii_list.append(my_custom_scenario)
#
# my_custom_scenario = scenario.Scenario(is_quick_demo=IS_QUICK_DEMO)
# my_custom_scenario.nodes_count = 5 # Number of nodes in the collaborative ML project simulated
# my_custom_scenario.amounts_per_node = [0.1, 0.1, 0.2, 0.2, 0.4] # Percentages of the data samples for each node
# my_custom_scenario.samples_split_option = 'Stratified' # If data are split randomly between nodes or stratified to be distinct (toggle between 'Random' and 'Stratified')
# my_custom_scenario.testset_option = 'Centralised' # If test data are distributed between nodes or stays a central testset (toggle between 'Centralised' and 'Distributed')
# my_custom_scenario.corrupted_nodes = ["not-corrupted"]*my_custom_scenario.nodes_count # no corrupted node
# scenarii_list.append(my_custom_scenario)


#%% Run the scenarii

for current_scenario in scenarii_list:

    #%% Split data according to scenario and then pre-process successively train data, early stopping validation data, test data

    current_scenario.split_data()
    current_scenario.plot_data_distribution()
    current_scenario = fl_training.preprocess_scenarios_data(current_scenario)


    #%% Corrupt the node's label in needed

    for i, node in enumerate(current_scenario.node_list):
        if current_scenario.corrupted_nodes[i] == "corrupted":
            print("corruption of node " + str(i) + "\n")
            node.corrupt_labels()
        elif current_scenario.corrupted_nodes[i] == "shuffled":
            print("shuffleling of node " + str(i) + "\n")
            node.shuffle_labels()
        elif current_scenario.corrupted_nodes[i] == "not-corrupted":
            pass
        else:
            print("unexpeted label of corruption")

    #%% Train and eval on all nodes according to scenario

    is_save_fig = True
    current_scenario.federated_test_score = fl_training.compute_test_score_with_scenario(
        current_scenario, is_save_fig
    )

    #%% Contributivity 1: Baseline contributivity measurement (Shapley Value)

    start = timer()
    (contributivity_scores, scores_var) = contributivity_measures.compute_SV(
        current_scenario.node_list,
        current_scenario.epoch_count,
        current_scenario.x_valearlystop,
        current_scenario.y_valearlystop,
        current_scenario.x_test,
        current_scenario.y_test,
    )
    end = timer()

    shapley_contrib = contributivity.Contributivity(
        "Shapley values", contributivity_scores, scores_var, np.round(end - start)
    )

    current_scenario.append_contributivity(shapley_contrib)
    print("\n## Evaluating contributivity with Shapley:")
    print(shapley_contrib)

    #%% Contributivity 2: Performance scores of models trained independently on each node

    start = timer()
    scores = contributivity_measures.compute_independent_scores(
        current_scenario.node_list,
        current_scenario.epoch_count,
        current_scenario.federated_test_score,
        current_scenario.testset_option,
        current_scenario.x_test,
        current_scenario.y_test,
    )
    end = timer()
    # TODO use dict instead of 0/1 indexes
    independant_raw_contrib = contributivity.Contributivity(
        "Independant scores raw", scores[0], np.repeat(0.0, len(scores[0]))
    )
    independant_additiv_contrib = contributivity.Contributivity(
        "Independant scores additive", scores[1], np.repeat(0.0, len(scores[1]))
    )

    independant_computation_time = np.round(end - start)
    independant_raw_contrib.computation_time = independant_computation_time
    independant_additiv_contrib.computation_time = independant_computation_time

    current_scenario.append_contributivity(independant_raw_contrib)
    current_scenario.append_contributivity(independant_additiv_contrib)
    print("\n## Evaluating contributivity with independent single partner models:")
    print(independant_raw_contrib)
    print(independant_additiv_contrib)

    #%% Contributivity 3: Truncated Monte Carlo Shapley

    start = timer()
    tmcs_results = contributivity_measures.truncated_MC(
        current_scenario, sv_accuracy=0.01, alpha=0.9, contrib_accuracy=0.05
    )
    end = timer()

    tmcs_contrib = contributivity.Contributivity(
        "TMCS values", tmcs_results["sv"], tmcs_results["std_sv"], np.round(end - start)
    )

    current_scenario.append_contributivity(tmcs_contrib)
    print("\n## Evaluating contributivity with Truncated Monte Carlo Shapley (TMCS):")
    print(tmcs_contrib)

    #%% Save results to file

    current_scenario.to_file()
