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
import utils
from loguru import logger


import contributivity
import contributivity_measures
import fl_training
import scenario


def main():

    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    plt.close("all")

    yaml_filepath = "config.yml"
    config = utils.load_cfg(yaml_filepath)
    config = utils.init_result_folder(yaml_filepath, config)
    experiment_path = config["experiment_path"]

    scenario_params_list = config["scenario_params_list"]
    n_repeats = config["n_repeats"]

    for i in range(n_repeats):

        logger.info(f"Repeat {i+1}/{n_repeats}")

        for scenario_params in scenario_params_list:

            logger.info("Current params:")
            logger.info(scenario_params)

            current_scenario = scenario.Scenario(scenario_params, experiment_path)
            run_scenario(current_scenario)

            # Write results to CSV file
            df_results = current_scenario.to_dataframe()
            df_results["random_state"] = i

            with open(experiment_path / 'results.csv', "a") as f:
                df_results.to_csv(f, header=f.tell() == 0, index=False)
                logger.info("Results saved")

    return 0


def run_scenario(current_scenario):

    current_scenario.split_data()
    current_scenario.plot_data_distribution()

    # Pre-process successively train data, early stopping validation data, test data
    current_scenario.node_list = fl_training.preprocess_node_list(
        current_scenario.node_list
    )
    (
        current_scenario.x_esval,
        current_scenario.y_esval,
    ) = fl_training.preprocess_test_data(
        current_scenario.x_esval, current_scenario.y_esval
    )
    current_scenario.x_test, current_scenario.y_test = fl_training.preprocess_test_data(
        current_scenario.x_test, current_scenario.y_test
    )

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
        current_scenario.x_esval,
        current_scenario.y_esval,
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

    # #%% Contributivity 3: Truncated Monte Carlo Shapley

    # start = timer()
    # tmcs_results = contributivity_measures.truncated_MC(
    #     current_scenario, sv_accuracy=0.01, alpha=0.9, contrib_accuracy=0.05
    # )
    # end = timer()

    # tmcs_contrib = contributivity.Contributivity(
    #     "TMCS values", tmcs_results["sv"], tmcs_results["std_sv"], np.round(end - start)
    # )

    # current_scenario.append_contributivity(tmcs_contrib)
    # print("\n## Evaluating contributivity with Truncated Monte Carlo Shapley (TMCS):")
    # print(tmcs_contrib)

    #%% Save results to file

    current_scenario.to_file()

    return 0


if __name__ == "__main__":
    main()
