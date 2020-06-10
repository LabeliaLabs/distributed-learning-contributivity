# -*- coding: utf-8 -*-
"""
This enables to parameterize a desired scenario to mock a multi-partner ML project.
"""

from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import datetime
import os
import keras
import numpy as np
import matplotlib.pyplot as plt
import uuid
import pandas as pd
from loguru import logger
import operator
import random

import constants
from partner import Partner
import utils


class Scenario:
    def __init__(self, params, experiment_path, scenario_id=1, n_repeat=1):

        # Identify and get a dataset for running experiments
        self.dataset_name = "MNIST"
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.nb_samples_used = len(x_train)
        self.final_relative_nb_samples = []

        # The train set has to be split into a train set and a validation set for early stopping
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )
        self.x_test = x_test
        self.y_test = y_test

        # List of all partners defined in the scenario
        self.partners_list = []

        # List of contributivity measures selected and computed in the scenario
        self.contributivity_list = []

        # --------------------------------------
        #  Definition of collaborative scenarios
        # --------------------------------------

        # partners mock different partners in a collaborative data science project
        # For defining the number of partners
        self.partners_count = params["partners_count"]

        # For configuring the respective sizes of the partners' datasets
        # Should the partners receive an equivalent amount of samples each or receive different amounts?
        # Define the percentages of samples per partner
        # Sum has to equal 1 and number of items has to equal partners_count
        self.amounts_per_partner = params["amounts_per_partner"]

        # For configuring if data samples are split between partners randomly or in a stratified way...
        # ... so that they cover distinct areas of the samples space
        if "samples_split_option" in params:
            self.samples_split_option = params["samples_split_option"]
        else:
            self.samples_split_option = "random"  # default

        # For configuring if the data of the partners are corrupted or not (useful for testing contributivity measures)
        if "corrupted_datasets" in params:
            self.corrupted_datasets = params["corrupted_datasets"]
        else:
            self.corrupted_datasets = ["not_corrupted"] * self.partners_count  # default

        # ---------------------------------------------------
        #  Configuration of the distributed learning approach
        # ---------------------------------------------------

        self.mpl = None

        # Multi-partner learning approach
        multi_partner_learning_approaches_list = [
            "fedavg",
            "seq-pure",
            "seq-with-final-agg",
            "seqavg",
        ]

        if "multi_partner_learning_approach" in params:
            approach = params["multi_partner_learning_approach"]
            if approach in multi_partner_learning_approaches_list:
                self.multi_partner_learning_approach = approach
            else:
                raise Exception(f"Multi-partner learning approach '{approach}' is not a valid approach.")
        else:
            self.multi_partner_learning_approach = 'fedavg'  # default

        # When training on a single partner,
        # the test set can be either the local partner test set or the global test set
        if "single_partner_test_mode" in params:
            self.single_partner_test_mode = params[
                "single_partner_test_mode"
            ]  # Toggle between 'local' and 'global'
        else:
            self.single_partner_test_mode = "global"  # default

        # Define how federated learning aggregation steps are weighted. Toggle between 'uniform' and 'data_volume'
        # Default is 'uniform'
        if "aggregation_weighting" in params:
            self.aggregation_weighting = params["aggregation_weighting"]
        else:
            self.aggregation_weighting = "uniform"  # default

        # Number of epochs, mini-batches and fit_batches in ML training
        if "epoch_count" in params:
            self.epoch_count = params["epoch_count"]
            assert self.epoch_count > 0
        else:
            self.epoch_count = 40  # default

        if "minibatch_count" in params:
            self.minibatch_count = params["minibatch_count"]
            assert self.minibatch_count > 0
        else:
            self.minibatch_count = 20  # default

        if "gradient_updates_per_pass_count" in params:
            self.gradient_updates_per_pass_count = params["gradient_updates_per_pass_count"]
            assert self.gradient_updates_per_pass_count > 0
        else:
            self.gradient_updates_per_pass_count = constants.DEFAULT_GRADIENT_UPDATES_PER_PASS_COUNT

        # Early stopping stops ML training when performance increase is not significant anymore
        # It is used to optimize the number of epochs and the execution time
        if "is_early_stopping" in params:
            self.is_early_stopping = params["is_early_stopping"]
        else:
            self.is_early_stopping = True  # default

        # -----------------------------------------------------------------
        #  Configuration of contributivity measurement methods to be tested
        # -----------------------------------------------------------------

        # Contributivity methods
        contributivity_methods_list = [
            "Shapley values",
            "Independent scores",
            "TMCS",
            "ITMCS",
            "IS_lin_S",
            "IS_reg_S",
            "AIS_Kriging_S",
            "SMCS",
            "WR_SMC",
            "DVRL"
        ]

        self.methods = []
        if "methods" in params and params["methods"]:
            for method in params["methods"]:
                if method in contributivity_methods_list:
                    self.methods.append(method)
                else:
                    raise Exception(f"Contributivity method '{method}' is not in methods list.")

        # -------------
        # Miscellaneous
        # -------------

        # Scenario id and number of repetition
        self.scenario_id = scenario_id
        self.n_repeat = n_repeat

        # The quick demo parameters overwrites previously defined parameters to make the scenario faster to compute
        if "is_quick_demo" in params and params["is_quick_demo"]:
            # Use less data and less epochs to speed up the computations
            logger.info("Quick demo: limit number of data and number of epochs.")
            self.x_train = self.x_train[:1000]
            self.y_train = self.y_train[:1000]
            self.x_val = self.x_val[:500]
            self.y_val = self.y_val[:500]
            self.x_test = self.x_test[:500]
            self.y_test = self.y_test[:500]
            self.epoch_count = 3
            self.minibatch_count = 2

        # -------
        # Outputs
        # -------

        now = datetime.datetime.now()
        now_str = now.strftime("%Y-%m-%d_%Hh%M")
        self.scenario_name = (
                "scenario_"
                + str(self.scenario_id)
                + "_"
                + "repeat"
                + "_"
                + str(self.n_repeat)
                + "_"
                + now_str
                + "_"
                + uuid.uuid4().hex[
                  :3
                  ]  # This is to be sure 2 distinct scenarios do no have the same name
        )

        self.short_scenario_name = (
                str(self.partners_count)
                + " "
                + str(self.amounts_per_partner)
        )

        self.save_folder = experiment_path / self.scenario_name

        self.save_folder.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------
        # Print the description of the scenario configured
        # ------------------------------------------------

        # Describe scenario
        logger.info("### Description of data scenario configured:")
        logger.info(f"   Number of partners defined: {self.partners_count}")
        logger.info(f"   Data distribution scenario chosen: {self.samples_split_option}")
        logger.info(f"   Test data distribution scenario chosen: {self.single_partner_test_mode}")
        logger.info(f"   Multi-partner learning approach: {self.multi_partner_learning_approach}")
        logger.info(f"   Weighting option: {self.aggregation_weighting}")
        logger.info(f"   Iterations parameters: "
                    f"{self.epoch_count} epochs > "
                    f"{self.minibatch_count} mini-batches > "
                    f"{self.gradient_updates_per_pass_count} gradient updates per pass")

        # Describe data
        logger.info(f"### Data loaded: {self.dataset_name}")
        logger.info(f"   {len(self.x_train)} train data with {len(self.y_train)} labels")
        logger.info(f"   {len(self.x_val)} train data with {len(self.y_val)} labels")
        logger.info(f"   {len(self.x_test)} train data with {len(self.y_test)} labels")

    def append_contributivity(self, contributivity):

        self.contributivity_list.append(contributivity)

    def instantiate_scenario_partners(self):

        self.partners_list = [Partner(i) for i in range(self.partners_count)]

    def split_data_advanced(self):
        """Advanced split: Populates the partners with their train and test data (not pre-processed)"""

        x_train = self.x_train
        y_train = self.y_train
        x_test = self.x_test
        y_test = self.y_test
        partners_list = self.partners_list
        amounts_per_partner = self.amounts_per_partner
        advanced_split_option = self.samples_split_option

        # Compose the lists of partners with data samples from shared clusters and those with specific clusters
        for p in partners_list:
            p.cluster_count = int(advanced_split_option[p.id][0])
            p.cluster_split_option = advanced_split_option[p.id][1]
        partners_with_shared_clusters = [p for p in partners_list if p.cluster_split_option == 'shared']
        partners_with_specific_clusters = [p for p in partners_list if p.cluster_split_option == 'specific']
        partners_with_shared_clusters.sort(key=operator.attrgetter("cluster_count"), reverse=True)
        partners_with_specific_clusters.sort(key=operator.attrgetter("cluster_count"), reverse=True)

        # Compose the list of different labels in the dataset
        labels = list(set(y_train))
        random.seed(42)
        random.shuffle(labels)

        # Check coherence of the split option:
        nb_diff_labels = len(labels)
        specific_clusters_count = sum([p.cluster_count for p in partners_with_specific_clusters])
        if partners_with_shared_clusters:
            shared_clusters_count = max([p.cluster_count for p in partners_with_shared_clusters])
        else:
            shared_clusters_count = 0
        assert specific_clusters_count + shared_clusters_count <= nb_diff_labels

        # Stratify the dataset into clusters per labels
        x_train_for_cluster, y_train_for_cluster, nb_samples_per_cluster = {}, {}, {}
        for label in labels:
            idx_in_full_trainset = np.where(y_train == label)
            x_train_for_cluster[label] = x_train[idx_in_full_trainset]
            y_train_for_cluster[label] = y_train[idx_in_full_trainset]
            nb_samples_per_cluster[label] = len(y_train_for_cluster[label])

        # For each partner compose the list of clusters from which they will draw data samples
        index = 0
        for p in partners_with_specific_clusters:
            p.clusters_list = labels[index:index + p.cluster_count]
            index += p.cluster_count

        shared_clusters = labels[index:index + shared_clusters_count]
        for p in partners_with_shared_clusters:
            p.clusters_list = random.sample(shared_clusters, k=p.cluster_count)

        # We need to enforce the relative data amounts configured.
        # It might not be possible to distribute all data samples, depending on...
        # ... the coherence of the relative data amounts and the split option.
        # We will compute a resize factor to determine the total nb of samples to be distributed per partner

        # For partners getting data samples from specific clusters...
        # ... compare the nb of available samples vs. the nb of samples initially configured
        resize_factor_specific = 1
        for p in partners_with_specific_clusters:
            nb_available_samples = sum([nb_samples_per_cluster[cl] for cl in p.clusters_list])
            nb_samples_requested = int(amounts_per_partner[p.id] * len(y_train))
            ratio = nb_available_samples / nb_samples_requested
            resize_factor_specific = min(resize_factor_specific, ratio)

        # For each partner getting data samples from shared clusters:
        # ... compute the nb of samples initially configured and resize it,
        # ... then sum per cluster how many samples are needed.
        # Then, find if a cluster is requested more samples than it has, and if yes by which factor
        resize_factor_shared = 1
        nb_samples_needed_per_cluster = dict.fromkeys(shared_clusters, 0)
        for p in partners_with_shared_clusters:
            initial_amount_resized = int(amounts_per_partner[p.id] * len(y_train) * resize_factor_specific)
            initial_amount_resized_per_cluster = int(initial_amount_resized / p.cluster_count)
            for cl in p.clusters_list:
                nb_samples_needed_per_cluster[cl] += initial_amount_resized_per_cluster
        for cl in nb_samples_needed_per_cluster:
            resize_factor_shared = min(resize_factor_shared,
                                       nb_samples_per_cluster[cl] / nb_samples_needed_per_cluster[cl],
                                       )

        # Compute the final resize factor
        final_resize_factor = resize_factor_specific * resize_factor_shared

        # Size correctly each partner's subset. For each partner:
        for p in partners_list:
            p.final_nb_samples = int(amounts_per_partner[p.id] * len(y_train) * final_resize_factor)
            p.final_nb_samples_p_cluster = int(p.final_nb_samples / p.cluster_count)
        self.nb_samples_used = sum([p.final_nb_samples for p in partners_list])
        self.final_relative_nb_samples = [p.final_nb_samples / self.nb_samples_used for p in partners_list]

        # Partners receive their subsets
        shared_clusters_index = dict.fromkeys(shared_clusters, 0)
        for p in partners_list:
            list_arrays_x, list_arrays_y = [], []
            if p in partners_with_shared_clusters:
                for cl in p.clusters_list:
                    idx = shared_clusters_index[cl]
                    list_arrays_x.append(x_train_for_cluster[cl][idx:idx + p.final_nb_samples_p_cluster])
                    list_arrays_y.append(y_train_for_cluster[cl][idx:idx + p.final_nb_samples_p_cluster])
                    shared_clusters_index[cl] += p.final_nb_samples_p_cluster
            elif p in partners_with_specific_clusters:
                for cl in p.clusters_list:
                    list_arrays_x.append(x_train_for_cluster[cl][:p.final_nb_samples_p_cluster])
                    list_arrays_y.append(y_train_for_cluster[cl][:p.final_nb_samples_p_cluster])
            p.x_train = np.concatenate(list_arrays_x)
            p.y_train = np.concatenate(list_arrays_y)
            p.x_test = x_test
            p.y_test = y_test

        # Check coherence of number of mini-batches versus partner with small dataset
        assert self.minibatch_count <= min([len(p.x_train) for p in self.partners_list])

        # Print for controlling
        logger.info("### Splitting data among partners:")
        logger.info(f"   Advanced split performed.")
        logger.info(f"   Nb of samples split amongst partners: {self.nb_samples_used}")
        logger.info(f"   Partners' relative nb of samples: {[round(p, 2) for p in self.final_relative_nb_samples]} "
                    f"   (versus initially configured: {amounts_per_partner})")
        for partner in self.partners_list:
            logger.info(f"   Partner #{partner.id}: {len(partner.x_train)} samples with labels {partner.clusters_list}")

        return 0

    def split_data(self):
        """Populates the partners with their train and test data (not pre-processed)"""

        # Fetch parameters of scenario
        x_train = self.x_train
        y_train = self.y_train
        x_test = self.x_test
        y_test = self.y_test

        # Configure the desired splitting scenario - Datasets sizes
        # Should the partners receive an equivalent amount of samples each...
        # ... or receive different amounts?

        # Check the percentages of samples per partner and control its coherence
        assert len(self.amounts_per_partner) == self.partners_count
        assert np.sum(self.amounts_per_partner) == 1

        # Then we parameterize this via the splitting_indices to be passed to np.split
        # This is to transform the percentages from the scenario configuration into indices where to split the data
        if self.partners_count == 1:
            splitting_indices_train = 1
            splitting_indices_test = 1
        else:
            splitting_indices = np.empty((self.partners_count - 1,))
            splitting_indices[0] = self.amounts_per_partner[0]
            for i in range(self.partners_count - 2):
                splitting_indices[i + 1] = (
                        splitting_indices[i] + self.amounts_per_partner[i + 1]
                )
            splitting_indices_train = (splitting_indices * len(y_train)).astype(int)
            splitting_indices_test = (splitting_indices * len(y_test)).astype(int)

        # Configure the desired data distribution scenario

        # Create a list of indexes of the samples
        train_idx = np.arange(len(y_train))
        test_idx = np.arange(len(y_test))

        # In the 'stratified' scenario we sort MNIST by labels
        if self.samples_split_option == "stratified":
            # Sort MNIST by labels
            y_sorted_idx = y_train.argsort()
            y_train = y_train[y_sorted_idx]
            x_train = x_train[y_sorted_idx]

        # In the 'random' scenario we shuffle randomly the indexes
        elif self.samples_split_option == "random":
            np.random.seed(42)
            np.random.shuffle(train_idx)

        # If neither 'stratified' nor 'random', we raise an exception
        else:
            raise NameError(
                "This samples_split_option scenario ["
                + self.samples_split_option
                + "] is not recognized."
            )

        # Do the splitting among partners according to desired scenarios

        # Split data between partners
        train_idx_idx_list = np.split(train_idx, splitting_indices_train)
        test_idx_idx_list = np.split(test_idx, splitting_indices_test)

        # Populate partners
        partner_idx = 0
        for train_idx, test_idx in zip(train_idx_idx_list, test_idx_idx_list):
            current_partner = self.partners_list[partner_idx]

            # Train data
            x_partner_train = x_train[train_idx, :]
            y_partner_train = y_train[
                train_idx,
            ]

            # Test data (for use in scenarios with single_partner_test_mode == 'local')
            x_partner_test = x_test[test_idx]
            y_partner_test = y_test[test_idx]

            current_partner.x_train = x_partner_train
            current_partner.x_test = x_partner_test
            current_partner.y_train = y_partner_train
            current_partner.y_test = y_partner_test

            current_partner.final_nb_samples = len(current_partner.x_train)
            current_partner.clusters_list = list(set(current_partner.y_train))

            partner_idx += 1

        # Check coherence of number of mini-batches versus smaller partner
        assert self.minibatch_count <= (min(self.amounts_per_partner) * len(x_train))

        self.nb_samples_used = sum([len(p.x_train) for p in self.partners_list])
        self.final_relative_nb_samples = [p.final_nb_samples / self.nb_samples_used for p in self.partners_list]

        # Print for controlling
        logger.info(f"### Splitting data among partners:")
        logger.info(f"   Simple split performed.")
        logger.info(f"   Nb of samples split amongst partners: {self.nb_samples_used}")
        for partner in self.partners_list:
            logger.info(f"   Partner #{partner.id}: "
                        f"{partner.final_nb_samples} samples "
                        f"with labels {partner.clusters_list}")

        return 0

    def plot_data_distribution(self):

        for i, partner in enumerate(self.partners_list):

            plt.subplot(self.partners_count, 1, i + 1)  # TODO share y axis
            data_count = np.bincount(partner.y_train)

            # Fill with 0
            while len(data_count) < 10:
                data_count = np.append(data_count, 0)

            plt.bar(np.arange(0, 10), data_count)
            plt.ylabel("partner " + str(partner.id))

        plt.suptitle("Data distribution")
        plt.xlabel("Digits")

        if not os.path.exists(self.save_folder / 'graphs/'):
            os.makedirs(self.save_folder / 'graphs/')
        plt.savefig(self.save_folder / "graphs/data_distribution.png")
        plt.close()

    def compute_batch_sizes(self):

        # For each partner we compute the batch size in multi-partner and single-partner setups
        BATCH_SIZE_MIN = 1
        BATCH_SIZE_MAX = constants.MAX_BATCH_SIZE

        if self.partners_count == 1:
            p = self.partners_list[0]
            batch_size = int(len(p.x_train) / self.gradient_updates_per_pass_count)
            p.batch_size = np.clip(batch_size, BATCH_SIZE_MIN, BATCH_SIZE_MAX)
        else:
            for p in self.partners_list:
                batch_size = int(len(p.x_train) / (self.minibatch_count * self.gradient_updates_per_pass_count))
                p.batch_size = np.clip(batch_size, BATCH_SIZE_MIN, BATCH_SIZE_MAX)

        for p in self.partners_list:
            logger.info(f"   compute_batch_sizes(), partner #{p.id}: {p.batch_size}")

    def preprocess_scenarios_data(self):
        """Return scenario with central datasets (val, test) and distributed datasets (partners) pre-processed"""

        logger.info("## Pre-processing datasets of the scenario for keras CNN:")

        # First, datasets of each partner
        for partner_index, partner in enumerate(self.partners_list):

            # Preprocess inputs (x) data
            partner.x_train = utils.preprocess_input(partner.x_train)
            partner.x_test = utils.preprocess_input(partner.x_test)

            # Preprocess labels (y) data
            partner.y_train = keras.utils.to_categorical(partner.y_train, constants.NUM_CLASSES)
            partner.y_test = keras.utils.to_categorical(partner.y_test, constants.NUM_CLASSES)

            # Create validation dataset
            partner.x_train, partner.x_val, partner.y_train, partner.y_val = train_test_split(
                partner.x_train, partner.y_train, test_size=0.1, random_state=42
            )

            if self.corrupted_datasets[partner_index] == "corrupted":
                logger.info(f"   ... Corrupting data (offsetting labels) of partner #{partner.id}")
                partner.corrupt_labels()
            elif self.corrupted_datasets[partner_index] == "shuffled":
                logger.info(f"   ... Corrupting data (shuffling labels) of partner #{partner.id}")
                partner.shuffle_labels()
            elif self.corrupted_datasets[partner_index] == "not_corrupted":
                pass
            else:
                logger.info("Unexpected label of corruption, not corruption performed!")

            logger.info(f"   Partner #{partner.id}: done.")

        # Then the scenario central dataset of the scenario
        self.x_val = utils.preprocess_input(self.x_val)
        self.y_val = keras.utils.to_categorical(self.y_val, constants.NUM_CLASSES)
        logger.info("   Central early stopping validation set: done.")
        self.x_test = utils.preprocess_input(self.x_test)
        self.y_test = keras.utils.to_categorical(self.y_test, constants.NUM_CLASSES)
        logger.info("   Central testset: done.")

    def to_dataframe(self):

        df = pd.DataFrame()
        dict_results = {}

        # Scenario definition parameters
        dict_results["scenario_name"] = self.scenario_name
        dict_results["short_scenario_name"] = self.short_scenario_name
        dict_results["dataset_name"] = self.dataset_name
        dict_results["train_data_samples_count"] = len(self.x_train)
        dict_results["test_data_samples_count"] = len(self.x_test)
        dict_results["partners_count"] = self.partners_count
        dict_results["amounts_per_partner"] = self.amounts_per_partner
        dict_results["samples_split_option"] = self.samples_split_option
        dict_results["nb_samples_used"] = self.nb_samples_used
        dict_results["final_relative_nb_samples"] = self.final_relative_nb_samples

        # Multi-partner learning approach parameters
        dict_results["multi_partner_learning_approach"] = self.multi_partner_learning_approach
        dict_results["aggregation_weighting"] = self.aggregation_weighting
        dict_results["single_partner_test_mode"] = self.single_partner_test_mode
        dict_results["epoch_count"] = self.epoch_count
        dict_results["minibatch_count"] = self.minibatch_count
        dict_results["gradient_updates_per_pass_count"] = self.gradient_updates_per_pass_count
        dict_results["is_early_stopping"] = self.is_early_stopping
        dict_results["mpl_test_score"] = self.mpl.test_score
        dict_results["mpl_nb_epochs_done"] = self.mpl.nb_epochs_done
        dict_results["learning_computation_time_sec"] = self.mpl.learning_computation_time

        if not self.contributivity_list:
            df = df.append(dict_results, ignore_index=True)

        for contrib in self.contributivity_list:

            # Contributivity data
            dict_results["contributivity_method"] = contrib.name
            dict_results["contributivity_scores"] = contrib.contributivity_scores
            dict_results["contributivity_stds"] = contrib.scores_std
            dict_results["computation_time_sec"] = contrib.computation_time_sec
            dict_results["first_characteristic_calls_count"] = contrib.first_charac_fct_calls_count

            for i in range(self.partners_count):

                # Partner-specific data
                dict_results["partner_id"] = i
                dict_results["amount_per_partner"] = self.amounts_per_partner[i]
                dict_results["contributivity_score"] = contrib.contributivity_scores[i]
                dict_results["contributivity_std"] = contrib.scores_std[i]

                df = df.append(dict_results, ignore_index=True)

        return df
