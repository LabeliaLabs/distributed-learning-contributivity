# -*- coding: utf-8 -*-
"""
Implement multiple contributivity measurement approaches
"""

from __future__ import print_function

import numpy as np
from scipy.stats import norm
from itertools import combinations

from scipy.special import softmax
from scipy.stats import norm
import fl_training
import scenario
import shapley_value.shapley as sv


#%% Compute independent performance scores of models trained independently on each node

def compute_contributivity(method_to_compute, 
                           current_scenario,
                           sv_accuracy=0.01, 
                           alpha=0.9, 
                           contrib_accuracy=0.05):
    
    score_dict = {}
    
    if method_to_compute == "Shapley values":
        # Contributivity 1: Baseline contributivity measurement (Shapley Value)
        (contributivity_scores, scores_var) = compute_SV(
            current_scenario.node_list,
            current_scenario.epoch_count,
            current_scenario.x_val,
            current_scenario.y_val,
            current_scenario.x_test,
            current_scenario.y_test,
            current_scenario.aggregation_weighting,
            current_scenario.minibatch_count,
            current_scenario.is_early_stopping,
            current_scenario.single_partner_test_mode
        )
        score_dict = {"Shapley values": (contributivity_scores, scores_var)}
    elif method_to_compute == "Independant scores":
        # Contributivity 2: Performance scores of models trained independently on each node
        scores = compute_independent_scores(
            current_scenario.node_list,
            current_scenario.epoch_count,
            current_scenario.federated_test_score,
            current_scenario.single_partner_test_mode,
            current_scenario.x_test,
            current_scenario.y_test,
        )
        score_dict = {"Independant scores raw": (scores[0], np.repeat(0.0, len(scores[0]))),
                      "Independant scores additive": (scores[1], np.repeat(0.0, len(scores[1])))}
    elif method_to_compute == "TMCS":
        # Contributivity 3: Truncated Monte Carlo Shapley
        tmcs_results = truncated_MC(
                        current_scenario, sv_accuracy, 
                        alpha, contrib_accuracy
                        )
        score_dict = {"TMCS": (tmcs_results["sv"], 
                                      tmcs_results["std_sv"])}
    
    return score_dict


def compute_independent_scores(
    node_list,
    epoch_count,
    collaborative_score,
    single_partner_test_mode,
    global_x_test,
    global_y_test,
):

    print(
        "\n# Launching computation of perf. scores of models trained independently on each node"
    )

    # Initialize a list of performance scores
    performance_scores = []

    # Train models independently on each node and append perf. score to list of perf. scores
    for node in node_list:
        performance_scores.append(
            fl_training.compute_test_score_for_single_node(
                node, epoch_count, single_partner_test_mode, global_x_test, global_y_test
            )
        )

    # Compute 'regularized' values of performance scores so that they are additive and their sum...
    # ... amount to the collaborative performance score obtained by the coalition of all players (nodes)
    perf_scores_additive = softmax(performance_scores) * collaborative_score

    # Return performance scores both raw and additively regularized
    return [np.array(performance_scores), np.array(perf_scores_additive)]


#%% Generalization of Shapley Value computation


def compute_SV(node_list, epoch_count, x_val_global, y_val_global, x_test, y_test, aggregation_weighting, minibatch_count,is_early_stopping,single_partner_test_mode):

    print("\n# Launching computation of Shapley Value of all nodes")

    # Initialize list of all players (nodes) indexes
    nodes_count = len(node_list)
    nodes_idx = np.arange(nodes_count)
    # print('All players (nodes) indexes: ', nodes_idx) # VERBOSE

    # Define all possible coalitions of players
    coalitions = [
        list(j) for i in range(len(nodes_idx)) for j in combinations(nodes_idx, i + 1)
    ]
    # print('All possible coalitions of players (nodes): ', coalitions) # VERBOSE

    # For each coalition, obtain value of characteristic function...
    # ... i.e.: train and evaluate model on nodes part of the given coalition
    characteristic_function = []

    for coalition in coalitions:
        coalition_nodes = list(node_list[i] for i in coalition)
        # print('\nComputing characteristic function on coalition ', coalition) # VERBOSE
        characteristic_function.append(
            fl_training.compute_test_score(
                coalition_nodes, epoch_count,
                x_val_global,
                y_val_global,
                x_test, y_test,
                aggregation_weighting,
                minibatch_count,
                is_early_stopping,
                single_partner_test_mode
            )
        )
    # print('\nValue of characteristic function for all coalitions: ', characteristic_function) # VERBOSE

    # Compute Shapley Value for each node
    # We are using this python implementation: https://github.com/susobhang70/shapley_value
    # It requires coalitions to be ordered - see README of https://github.com/susobhang70/shapley_value
    list_shapley_value = sv.main(nodes_count, characteristic_function)

    # Return SV of each node
    return (np.array(list_shapley_value), np.repeat(0.0, len(list_shapley_value)))


#%% compute Shapley values with the truncated Monte-carlo method


def truncated_MC(scenario, sv_accuracy=0.01, alpha=0.9, contrib_accuracy=0.05):
    """Return the vector of approximated Shapley value corresponding to a list of node and a characteristic function using the Truncated Monte-Carlo method."""

    preprocessed_node_list = scenario.node_list
    n = len(preprocessed_node_list)

    # We store the value of the characteristic function in order to avoid recomputing it twice
    char_value_dict = {(): 0}  # the dictionary that will countain the values

    # Return the characteristic function of the nodelist associated to the ensemble permut, without recomputing it if it was already computed
    def not_twice_characteristic(permut):
        # Sort permut
        permut = np.sort(permut)
        try:  # Return the characteristic_func(permut) if it was already computed
            return char_value_dict[tuple(permut)]
        except KeyError:  # Characteristic_func(permut) has not been computed yet, so we compute, store, and return characteristic_func(permut)
            small_node_list = np.array([preprocessed_node_list[i] for i in permut])
            char_value_dict[tuple(permut)] = fl_training.compute_test_score(
                small_node_list,
                scenario.epoch_count,
                scenario.x_val,
                scenario.y_val,
                scenario.x_test,
                scenario.y_test,
                scenario.aggregation_weighting,
                scenario.minibatch_count,
                scenario.is_early_stopping,
                scenario.single_partner_test_mode,
                save_folder=scenario.save_folder,
            )

            return char_value_dict[tuple(permut)]

    characteristic_all_node = not_twice_characteristic(
        np.arange(n)
    )  # Characteristic function on all nodes
    if n == 1:
        return {
            "sv": characteristic_all_node,
            "std_sv": np.array([0]),
            "prop": np.array([1]),
            "computed_val": char_value_dict,
        }
    else:
        contributions = np.array([[]])
        permutation = np.zeros(n)  # Store the current permutation
        t = 0
        q = norm.ppf((1 - alpha) / 2, loc=0, scale=1)
        v_max = 0
        while (
            t < 100 or t < q ** 2 * v_max / (sv_accuracy * characteristic_all_node) ** 2
        ):  # Check if the length of the confidence interval  is below the value of sv_accuracy*characteristic_all_node
            t += 1

            if t == 1:
                contributions = np.array([np.zeros(n)])
            else:
                contributions = np.vstack((contributions, np.zeros(n)))

            permutation = np.random.permutation(n)  # Store the current permutation
            char_nodelists = np.zeros(
                n + 1
            )  # Store the characteristic function on each ensemble built with the first elements of the permutation
            char_nodelists[-1] = characteristic_all_node
            for j in range(n):
                if abs(characteristic_all_node - char_nodelists[j]) < contrib_accuracy:
                    char_nodelists[j + 1] = char_nodelists[j]
                else:
                    char_nodelists[j + 1] = not_twice_characteristic(
                        permutation[: j + 1]
                    )
                contributions[-1][permutation[j]] = (
                    char_nodelists[j + 1] - char_nodelists[j]
                )
            v_max = np.max(np.var(contributions, axis=0))
        sv = np.mean(contributions, axis=0)

    return {
        "sv": sv,
        "std_sv": np.std(contributions, axis=0) / np.sqrt(t - 1),
        "prop": sv / np.sum(sv),
        "computed_val": char_value_dict,
    }
