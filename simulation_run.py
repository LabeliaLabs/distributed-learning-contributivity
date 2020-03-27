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
    #gpus = tf.config.experimental.list_physical_devices("GPU")
    #tf.config.experimental.set_memory_growth(gpus[0], True)

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
    start = timer()
    (contributivity_scores, scores_var,fit_count) = contributivity_measures.compute_SV(
        current_scenario.node_list,
        current_scenario.epoch_count,
        current_scenario.x_val,
        current_scenario.y_val,
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
    print("\n## Number of fits with Shapley:" )
    print(fit_count)

    # Contributivity 2: Performance scores of models trained independently on each node

    start = timer()
    scores = contributivity_measures.compute_independent_scores(
        current_scenario.node_list,
        current_scenario.epoch_count,
        current_scenario.federated_test_score,
        current_scenario.single_partner_test_mode,
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
    print("\n## Number of fits with independent single partner models:" )
    print(scores[2])


    # Contributivity 3: Truncated Monte Carlo Shapley

    start = timer()
    tmcs_results = contributivity_measures.truncated_MC(
        current_scenario, sv_accuracy=0.01, alpha=0.95, truncation=0.05
    )
    end = timer()

    tmcs_contrib = contributivity.Contributivity(
        "TMCS values", tmcs_results["sv"], tmcs_results["std_sv"], np.round(end - start)
    )

    current_scenario.append_contributivity(tmcs_contrib)
    print("\n## Evaluating contributivity with Truncated Monte Carlo Shapley (TMCS):")
    print(tmcs_contrib)


    # Contributivity 4: imterpolated monte-carlo

    start = timer()
    IMCS_results = contributivity_measures.interpol_trunc_MC(
        current_scenario, sv_accuracy=0.01, alpha=0.95
    )
    end = timer()

    IMC_contib = contributivity.Contributivity(
        "IMCS values", IMCS_results["sv"], IMCS_results["std_sv"], np.round(end - start)
    )

    current_scenario.append_contributivity(IMC_contib)
    print("\n## Evaluating contributivity with imterpolated Monte Carlo Shapley (IMCS):")
    print(IMC_contib)
    
    # Contributivity 5: Stratified Monte Carlo  

    start = timer()
    SMCS_results = contributivity_measures.Stratified_MC(
        current_scenario, sv_accuracy=0.01, alpha=0.95
    )
    end = timer()

    SMCS_contib = contributivity.Contributivity(
        "SMCS values", SMCS_results["sv"], SMCS_results["std_sv"], np.round(end - start)
    )

    current_scenario.append_contributivity(SMCS_contib)
    print("\n## Evaluating contributivity with Stratified Monte Carlo Shapley (SMCS):")
    print(SMCS_contib)
    
    # Contributivity 6: Adaptative importance sampling with kriging model  

    start = timer()
    AISS_results = contributivity_measures.AIS_Kriging(
        current_scenario, sv_accuracy=0.01, alpha=0.95
    )
    end = timer()

    AISS_contib = contributivity.Contributivity(
        "AISS values", AISS_results["sv"], AISS_results["std_sv"], np.round(end - start)
    )

    current_scenario.append_contributivity(AISS_contib)
    print("\n## Evaluating contributivity with Adaptative importance sampling (AISS):")
    print(AISS_contib)
    
    
    # Contributivity 7:   importance sampling with linear interpolation model  

    start = timer()
    ISS_lin_results = contributivity_measures.IS_lin(
        current_scenario, sv_accuracy=0.01, alpha=0.95
    )
    end = timer()

    ISS_lin_contib = contributivity.Contributivity(
        "ISS_lin values", ISS_lin_results["sv"], ISS_lin_results["std_sv"], np.round(end - start)
    )

    current_scenario.append_contributivity(ISS_lin_contib)
    print("\n## Evaluating contributivity with  importance sampling Shapley with linear interpolation model  (ISS_lin):")
    print(ISS_lin_contib)
    
    # Contributivity 8:   importance sampling with linear regresion model  

    start = timer()
    ISS_reg_results = contributivity_measures.IS_lin(
        current_scenario, sv_accuracy=0.01, alpha=0.95
    )
    end = timer()

    ISS_reg_contib = contributivity.Contributivity(
        "ISS_reg values", ISS_reg_results["sv"], ISS_reg_results["std_sv"], np.round(end - start)
    )

    current_scenario.append_contributivity(ISS_reg_contib)
    print("\n## Evaluating contributivity with  importance sampling Shapley with regression model  (ISS_reg):")
    print(ISS_reg_contib)
    # Save results to file

    current_scenario.to_file()

    return 0


if __name__ == "__main__":
    main()
