# -*- coding: utf-8 -*-
"""
This enables to parameterize a desired scenario to mock a multi-partner ML project.
"""

from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import os
import sys
import datetime
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import uuid
import pandas as pd
from loguru import logger

from node import Node


class Scenario:
    def __init__(self, params, experiment_path):

        # Identify and get a dataset for running experiments
        self.dataset_name = "MNIST"
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # The train set has to be split into a train set and a validation set for early stopping
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )
        self.x_test = x_test
        self.y_test = y_test

        # Nodes mock different partners in a collaborative data science project
        # For defining the number of nodes
        self.nodes_count = params["nodes_counts"]

        # For configuring the respective sizes of the nodes' datasets
        # Should the nodes receive an equivalent amount of samples each or receive different amounts?
        # Define the percentages of samples per node
        # Sum has to equal 1 and number of items has to equal NODES_COUNT
        self.amounts_per_node = params["amounts_per_node"]

        # For configuring if data samples are split between nodes randomly or in a stratified way...
        # ... so that they cover distinct areas of the samples space
        self.samples_split_option = params[
            "samples_split_option"
        ]  # Toggle between 'Random' and 'Stratified'

        # For configuring if the data of the nodes are corrupted or not (useful for testing contributivity measures)
        if "corrupted_nodes" in params:
            self.corrupted_nodes = params["corrupted_nodes"]
        else:
            self.corrupted_nodes = ["not_corrupted"] * self.nodes_count

        # When training on a single node, the test set can be either the local node test set or the global test set
        self.single_partner_test_mode = params[
            "single_partner_test_mode"
        ]  # Toggle between 'local' and 'global'

        # Performance of the model trained in a distributed way on all nodes
        self.federated_test_score = int

        self.federated_computation_time = int

        # Define how federated learning aggregation steps are weighted. Toggle between 'uniform' and 'data_volume'
        # Default is 'uniform'
        if "aggregation_weighting" in params:
            self.aggregation_weighting = params["aggregation_weighting"]
        else:
            self.aggregation_weighting = "uniform"

        # List of all nodes defined in the scenario
        self.node_list = []

        # List of contributivity measures selected and computed in the scenario
        self.contributivity_list = []

        # Number of epochs and mini-batches in ML training
        if 'epoch_count' in params:
            self.epoch_count = params['epoch_count']
            assert(self.epoch_count > 0)
        else:
            self.epoch_count = 40
        
        if 'minibatch_count' in params:
            self.minibatch_count = params['minibatch_count']
            assert self.minibatch_count > 0
        else:
            self.minibatch_count = 20
        
        # Contributivity methods
        methods_default = [
            "Shapley values",
            "Independant scores",
            "TMCS values"
        ]
        
        self.methods = []
        if not 'methods' in params:
            self.methods = methods_default
        else:
            if params['methods']:
                for el in params['methods']:
                    if el in methods_default:
                        self.methods.append(el)
                    else:
                        raise Exception('method ' + el + ' is not in methods list.')
            else:
                raise Exception("No contributivity method given in config file")
        
        # Early stopping stops ML training when performance increase is not significant anymore
        # It is used to optimize the number of epochs and the execution time
        self.is_early_stopping = True # Toggle between True and False

        now = datetime.datetime.now()
        now_str = now.strftime("%Y-%m-%d_%Hh%M")
        self.scenario_name = (
            self.samples_split_option
            + "_"
            + str(self.nodes_count)
            + "_"
            + str(self.amounts_per_node)
            + "_"
            + str(self.corrupted_nodes)
            + "_"
            + str(self.single_partner_test_mode)
            + "_"
            + now_str
            + "_"
            + uuid.uuid4().hex[
                :3
            ]  # This is to be sure 2 distinct scenarios do no have the same name
        )

        self.short_scenario_name = (
            self.samples_split_option
            + " "
            + str(self.nodes_count)
            + " "
            + str(self.amounts_per_node)
        )

        self.save_folder = experiment_path / self.scenario_name

        self.save_folder.mkdir(parents=True, exist_ok=True)

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

    def append_contributivity(self, contributivity):

        self.contributivity_list.append(contributivity)

    def split_data(self):
        """Populates the nodes with their train and test data (not pre-processed)"""

        # Fetch parameters of scenario
        x_train = self.x_train
        y_train = self.y_train
        x_val = self.x_val
        y_val = self.y_val
        x_test = self.x_test
        y_test = self.y_test

        # Describe data
        print("\n### Data loaded: ", self.dataset_name)
        print("- " + str(len(x_train)) + " train data with " + str(len(y_train)) + " labels")
        print("- " + str(len(x_val)) + " val data with " + str(len(y_val)) + " labels")
        print("- " + str(len(x_test)) + " test data " + str(len(y_test)) + " labels")

        # Configure the desired splitting scenario - Datasets sizes
        # Should the nodes receive an equivalent amount of samples each...
        # ... or receive different amounts?
        # Check the percentages of samples per node and control its coherence
        assert len(self.amounts_per_node) == self.nodes_count
        assert np.sum(self.amounts_per_node) == 1

        # Then we parameterize this via the splitting_indices to be passed to np.split
        # This is to transform the percentages from the scenario configuration into indices where to split the data
        splitting_indices = np.empty((self.nodes_count - 1,))
        splitting_indices[0] = self.amounts_per_node[0]
        for i in range(self.nodes_count - 2):
            splitting_indices[i + 1] = (
                splitting_indices[i] + self.amounts_per_node[i + 1]
            )
        splitting_indices_train = (splitting_indices * len(y_train)).astype(int)
        splitting_indices_test = (splitting_indices * len(y_test)).astype(int)
        # print('- Splitting indices defined (for train data):', splitting_indices_train) # VERBOSE

        # Configure the desired data distribution scenario

        # Create a list of indexes of the samples
        train_idx = np.arange(len(y_train))
        test_idx = np.arange(len(y_test))

        # In the 'Stratified' scenario we sort MNIST by labels
        if self.samples_split_option == "Stratified":
            # Sort MNIST by labels
            y_sorted_idx = y_train.argsort()
            y_train = y_train[y_sorted_idx]
            x_train = x_train[y_sorted_idx]

        # In the 'Random' scenario we shuffle randomly the indexes
        elif self.samples_split_option == "Random":
            np.random.seed(42)
            np.random.shuffle(train_idx)

        # If neither 'Stratified' nor 'Random', we raise an exception
        else:
            raise NameError(
                "This samples_split_option scenario ["
                + self.samples_split_option
                + "] is not recognized."
            )

        # Do the splitting among nodes according to desired scenarios

        # Split data between nodes
        train_idx_idx_list = np.split(train_idx, splitting_indices_train)
        test_idx_idx_list = np.split(test_idx, splitting_indices_test)

        # Populate nodes
        node_id = 0
        for train_idx, test_idx in zip(train_idx_idx_list, test_idx_idx_list):

            # Train data
            x_node_train = x_train[train_idx, :]
            y_node_train = y_train[
                train_idx,
            ]

            # Test data (for use in scenarios with single_partner_test_mode == 'local')
            x_node_test = x_test[test_idx]
            y_node_test = y_test[test_idx]

            node = Node(
                x_node_train, x_node_test, y_node_train, y_node_test, str(node_id)
            )
            self.node_list.append(node)
            node_id += 1

        # Check coherence of node_list versus nodes_count
        assert len(self.node_list) == self.nodes_count

        # Check coherence of number of mini-batches versus smaller node
        assert self.minibatch_count <= (min(self.amounts_per_node) * len(x_train))

        # Print the description of the scenario configured
        print("\n### Description of data scenario configured:")
        print("- Number of nodes defined:", self.nodes_count)
        print("- Data distribution scenario chosen:", self.samples_split_option)
        print("- Test data distribution scenario chosen:", self.single_partner_test_mode)
        print("- Weighting option:", self.aggregation_weighting)
        print("- Number of epochs and mini-batches: " + str(self.epoch_count) + " epochs and " + str(self.minibatch_count) + " mini-batches")

        # Print and plot for controlling
        print("\n### Splitting data among nodes:")
        for node_index, node in enumerate(self.node_list):
            print("- Node #" + str(node_index) + ":")
            print("  - Number of samples:" + str(len(node.x_train)) + " train samples")
            print("  - y_train first 10 values:" + str(node.y_train[:10]))
            print("  - y_train last 10 values:" + str(node.y_train[-10:]))

        return 0

    def plot_data_distribution(self):

        for i, node in enumerate(self.node_list):

            plt.subplot(self.nodes_count, 1, i + 1)  # TODO share y axis
            # print(node.y_train) # VERBOSE
            # data = np.argmax(node.y_train, axis=1)
            data_count = np.bincount(node.y_train)

            # Fill with 0
            while len(data_count) < 10:
                data_count = np.append(data_count, 0)

            plt.bar(np.arange(0, 10), data_count)
            plt.ylabel("Node " + str(i))

        plt.suptitle("Data distribution")
        plt.xlabel("Digits")
        plt.savefig(self.save_folder / "data_distribution.png")

    def to_file(self):

        out = ""
        out += "Dataset name: " + self.dataset_name + "\n"
        out += "Number of data samples - train: " + str(len(self.x_train)) + "\n"
        out += "Number of data samples - test: " + str(len(self.x_test)) + "\n"
        out += "Nodes count: " + str(self.nodes_count) + "\n"
        out += (
            "Percentages of data samples per node: " + str(self.amounts_per_node) + "\n"
        )
        out += (
            "Random or stratified split of data samples: "
            + self.samples_split_option
            + "\n"
        )
        out += (
            "When training on a single node, global or local testset: "
            + self.single_partner_test_mode
            + "\n"
        )
        out += "Number of epochs: " + str(self.epoch_count) + "\n"
        out += "Number of mini-batches: " + str(self.minibatch_count) + "\n"
        out += "Early stopping on? " + str(self.is_early_stopping) + "\n"
        out += (
            "Test score of federated training: " + str(self.federated_test_score) + "\n"
        )
        out += "\n"

        out += str(len(self.contributivity_list)) + " contributivity methods: " + "\n"

        for contrib in self.contributivity_list:
            out += str(contrib) + "\n\n"

        target_file_path = self.save_folder / "results_summary.txt"

        with open(target_file_path, "w", encoding="utf-8") as f:
            f.write(out)

    def to_dataframe(self):

        df = pd.DataFrame()

        for contrib in self.contributivity_list:

            dict_results = {}

            for i in range(self.nodes_count):

                # Scenario data
                dict_results["dataset_name"] = self.dataset_name
                dict_results["train_data_samples_count"] = len(self.x_train)
                dict_results["test_data_samples_count"] = len(self.x_test)
                dict_results["nodes_count"] = self.nodes_count
                dict_results["amounts_per_node"] = self.amounts_per_node
                dict_results["samples_split_option"] = self.samples_split_option
                dict_results["single_partner_test_mode"] = self.single_partner_test_mode
                dict_results["epoch_count"] = self.epoch_count
                dict_results["is_early_stopping"] = self.is_early_stopping
                dict_results["federated_test_score"] = self.federated_test_score
                dict_results["federated_computation_time"] = self.federated_computation_time
                dict_results["scenario_name"] = self.scenario_name
                dict_results["short_scenario_name"] = self.short_scenario_name
                dict_results["minibatch_count"] = self.minibatch_count

                # Contributivity data
                dict_results["contributivity_method"] = contrib.name
                dict_results["contributivity_scores"] = contrib.contributivity_scores
                dict_results["contributivity_stds"] = contrib.scores_std
                dict_results["computation_time"] = contrib.computation_time

                # Node data
                dict_results["node_id"] = i
                dict_results["amount_per_node"] = self.amounts_per_node[i]
                dict_results["contributivity_score"] = contrib.contributivity_scores[i]
                dict_results["contributivity_std"] = contrib.scores_std[i]

                df = df.append(dict_results, ignore_index=True)

        df.info()

        return df
