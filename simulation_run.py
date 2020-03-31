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
import fl_training
import scenario

import argparse

DEFAULT_CONFIG_FILE = "config.yml"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="input config file")
    args = parser.parse_args()
    if args.file:
        logger.info(f"Using provided config file: {args.file}")
        config_filepath = args.file
    else:
        logger.info(f"Using default config file: {DEFAULT_CONFIG_FILE}")
        config_filepath = DEFAULT_CONFIG_FILE

    config = utils.load_cfg(config_filepath)
    config = utils.init_result_folder(config_filepath, config)
    experiment_path = config["experiment_path"]

    scenario_params_list = config["scenario_params_list"]
    n_repeats = config["n_repeats"]

    # GPU config
    # gpus = tf.config.experimental.list_physical_devices("GPU")
    # tf.config.experimental.set_memory_growth(gpus[0], True)

    # Close open figures
    plt.close("all")

    for i in range(n_repeats):

        logger.info(f"Repeat {i+1}/{n_repeats}")

        for scenario_id, scenario_params in enumerate(scenario_params_list):

            logger.info("Current params:")
            logger.info(scenario_params)

            print(type(scenario_params["amounts_per_node"]))

            current_scenario = scenario.Scenario(scenario_params, experiment_path)
            print(current_scenario.to_dataframe())

            run_scenario(current_scenario)

            # Write results to CSV file
            df_results = current_scenario.to_dataframe()
            df_results["random_state"] = i
            df_results["scenario_id"] = scenario_id

            with open(experiment_path / "results.csv", "a") as f:
                df_results.to_csv(f, header=f.tell() == 0, index=False)
                logger.info("Results saved")

    return 0


def run_scenario(current_scenario):

    #%% Split data according to scenario and then pre-process successively train data, early stopping validation data, test data

    current_scenario.split_data()
    current_scenario.plot_data_distribution()
    current_scenario = fl_training.preprocess_scenarios_data(current_scenario)

    # Train and eval on all nodes according to scenario
    is_save_fig = True
    current_scenario.federated_test_score = fl_training.compute_test_score_with_scenario(
        current_scenario, is_save_fig
    )

    # Contributivity 1: Baseline contributivity measurement (Shapley Value)

    shapley_contrib = contributivity.Contributivity(scenario=current_scenario)
    shapley_contrib.compute_SV(current_scenario)

    current_scenario.append_contributivity(shapley_contrib)
    print("\n## Evaluating contributivity with Shapley:")
    print(shapley_contrib)

    # Contributivity 2: Performance scores of models trained independently on each node

    independant_raw_contrib = contributivity.Contributivity(scenario=current_scenario)
    independant_raw_contrib.compute_independent_scores_raws(current_scenario)

    current_scenario.append_contributivity(independant_raw_contrib)
    print("\n## Evaluating contributivity with independent single partner models:")
    print(independant_raw_contrib)

    # Contributivity 3: Performance scores of models trained independently on each node

    independant_additiv_contrib = contributivity.Contributivity(
        scenario=current_scenario
    )
    independant_additiv_contrib.compute_independent_scores_additive(current_scenario)

    current_scenario.append_contributivity(independant_additiv_contrib)
    print(
        "\n## Evaluating contributivity with independent single partner models (additive contrib):"
    )
    print(independant_additiv_contrib)

    # Contributivity 4: Truncated Monte Carlo Shapley

    tmcs_contrib = contributivity.Contributivity(scenario=current_scenario)
    tmcs_contrib.truncated_MC(current_scenario)

    current_scenario.append_contributivity(tmcs_contrib)
    print("\n## Evaluating contributivity with Truncated Monte Carlo Shapley:")
    print(tmcs_contrib)

    # Contributivity 5: interpolated monte-carlo

    itmcs_contrib = contributivity.Contributivity(scenario=current_scenario)
    itmcs_contrib.interpol_trunc_MC(current_scenario)

    current_scenario.append_contributivity(itmcs_contrib)
    print(
        "\n## Evaluating contributivity with interpolated truncated Monte Carlo Shapley:"
    )
    print(itmcs_contrib)

    # Contributivity 6:   importance sampling with linear interpolation model
    IS_lin_contrib = contributivity.Contributivity(scenario=current_scenario)
    IS_lin_contrib.IS_lin(current_scenario)

    current_scenario.append_contributivity(IS_lin_contrib)
    print(
        "\n## Evaluating contributivity with importance sampling with linear interpolation model:"
    )
    print(IS_lin_contrib)

    # Contributivity 7: mportance sampling with regression model
    IS_reg_contrib = contributivity.Contributivity(scenario=current_scenario)
    IS_reg_contrib.IS_reg(current_scenario)

    current_scenario.append_contributivity(IS_reg_contrib)
    print(
        "\n## Evaluating contributivity with importance sampling with regresion model:"
    )
    print(IS_reg_contrib)

    # Contributivity 8: Adaptative importance sampling with kriging model
    AISS_contib = contributivity.Contributivity(scenario=current_scenario)
    AISS_contib.AIS_Kriging(current_scenario)

    current_scenario.append_contributivity(AISS_contib)
    print("\n## Evaluating contributivity with Adaptative importance sampling (AISS):")
    print(AISS_contib)

    # # Contributivity 9:  Stratified Monte Carlo
    SMC_contib = contributivity.Contributivity(scenario=current_scenario)
    SMC_contib.Stratified_MC(current_scenario)

    current_scenario.append_contributivity(SMC_contib)
    print("\n## Evaluating contributivity with stratified Monte Carlo Shapley:")
    print(SMC_contib)

    # # # Contributivity 10:  Stratified Monte Carlo
    Support_contib = contributivity.Contributivity(scenario=current_scenario)
    Support_contib.support(current_scenario)

    current_scenario.append_contributivity(Support_contib)
    print("\n## Evaluating contributivity with support stratified Monte Carlo Shapley:")
    print(Support_contib)

    # Save results to file

    current_scenario.to_file()

    return 0


if __name__ == "__main__":
    main()
