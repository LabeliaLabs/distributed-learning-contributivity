# -*- coding: utf-8 -*-
"""
This enables to parameterize a desired scenario to mock a multi-partner ML project.
"""

from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import datetime
import numpy as np
import matplotlib.pyplot as plt
import uuid
import pandas as pd
from loguru import logger

from partner import Partner


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

        # Performance of the model trained in a distributed way on all partners
        self.federated_test_score = int
        self.federated_computation_time_sec = int

        # List of all partners defined in the scenario
        self.partners_list = []

        # List of contributivity measures selected and computed in the scenario
        self.contributivity_list = []

        # -------------------------------------
        # Definition of collaborative scenarios
        # -------------------------------------

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
        self.samples_split_option = params[
            "samples_split_option"
        ]  # Toggle between 'random' and 'stratified'

        # For configuring if the data of the partners are corrupted or not (useful for testing contributivity measures)
        if "corrupted_datasets" in params:
            self.corrupted_datasets = params["corrupted_datasets"]
        else:
            self.corrupted_datasets = ["not_corrupted"] * self.partners_count

        # ---------------------------------------------------
        #  Configuration of the distributed learning approach
        # ---------------------------------------------------

        # When training on a single partner, the test set can be either the local partner test set or the global test set
        if "single_partner_test_mode" in params:
            self.single_partner_test_mode = params[
                "single_partner_test_mode"
            ]  # Toggle between 'local' and 'global'
        else:
            self.single_partner_test_mode = "global"

        # Define how federated learning aggregation steps are weighted. Toggle between 'uniform' and 'data_volume'
        # Default is 'uniform'
        if "aggregation_weighting" in params:
            self.aggregation_weighting = params["aggregation_weighting"]
        else:
            self.aggregation_weighting = "uniform"

        # Number of epochs and mini-batches in ML training
        if "epoch_count" in params:
            self.epoch_count = params["epoch_count"]
            assert self.epoch_count > 0
        else:
            self.epoch_count = 40

        if "minibatch_count" in params:
            self.minibatch_count = params["minibatch_count"]
            assert self.minibatch_count > 0
        else:
            self.minibatch_count = 20

        # Early stopping stops ML training when performance increase is not significant anymore
        # It is used to optimize the number of epochs and the execution time
        if "is_early_stopping" in params:
            self.is_early_stopping = params["is_early_stopping"]
        else:
            self.is_early_stopping = True

        # -----------------------------------------------------------------
        #  Configuration of contributivity measurement methods to be tested
        # -----------------------------------------------------------------

        # Contributivity methods
        ALL_METHODS_LIST = [
            "Shapley values",
            "Independant scores",
            "TMCS",
            "ITMCS",
            "IS_lin_S",
            "IS_reg_S",
            "AIS_Kriging_S",
            "SMCS",
            "WR_SMC",
        ]

        # List of Contributivity methods runned by default if no method was given in the config file
        DEFAULT_METHODS_LIST = ["Shapley values", "Independant scores", "TMCS"]

        self.methods = []
        if "methods" in params and params["methods"]:

            for method in params["methods"]:
                if method in ALL_METHODS_LIST:
                    self.methods.append(method)
                else:
                    raise Exception("Method [" + method + "] is not in methods list.")

        else:
            self.methods = DEFAULT_METHODS_LIST

        # -------------
        # Miscellaneous
        # -------------

        # The quick demo parameters overwrites previously defined paramaters to make the scenario faster to compute
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
                self.samples_split_option
                + "_"
                + str(self.partners_count)
                + "_"
                + str(self.amounts_per_partner)
                + "_"
                + str(self.corrupted_datasets)
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
                + str(self.partners_count)
                + " "
                + str(self.amounts_per_partner)
        )

        self.save_folder = experiment_path / self.scenario_name

        self.save_folder.mkdir(parents=True, exist_ok=True)

    def append_contributivity(self, contributivity):

        self.contributivity_list.append(contributivity)

    def split_data(self):
        """Populates the partners with their train and test data (not pre-processed)"""

        # Fetch parameters of scenario
        x_train = self.x_train
        y_train = self.y_train
        x_val = self.x_val
        y_val = self.y_val
        x_test = self.x_test
        y_test = self.y_test

        # Describe data
        print("\n### Data loaded: ", self.dataset_name)
        print(
            "- "
            + str(len(x_train))
            + " train data with "
            + str(len(y_train))
            + " labels"
        )
        print("- " + str(len(x_val)) + " val data with " + str(len(y_val)) + " labels")
        print("- " + str(len(x_test)) + " test data " + str(len(y_test)) + " labels")

        # Configure the desired splitting scenario - Datasets sizes
        # Should the partners receive an equivalent amount of samples each...
        # ... or receive different amounts?

        # Check the percentages of samples per partner and control its coherence
        assert len(self.amounts_per_partner) == self.partners_count
        assert np.sum(self.amounts_per_partner) == 1

        # Then we parameterize this via the splitting_indices to be passed to np.split
        # This is to transform the percentages from the scenario configuration into indices where to split the data
        splitting_indices = np.empty((self.partners_count - 1,))
        splitting_indices[0] = self.amounts_per_partner[0]
        for i in range(self.partners_count - 2):
            splitting_indices[i + 1] = (
                splitting_indices[i] + self.amounts_per_partner[i + 1]
            )
        splitting_indices_train = (splitting_indices * len(y_train)).astype(int)
        splitting_indices_test = (splitting_indices * len(y_test)).astype(int)
        # print('- Splitting indices defined (for train data):', splitting_indices_train) # VERBOSE

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
        partner_id = 0
        for train_idx, test_idx in zip(train_idx_idx_list, test_idx_idx_list):

            # Train data
            x_partner_train = x_train[train_idx, :]
            y_partner_train = y_train[
                train_idx,
            ]

            # Test data (for use in scenarios with single_partner_test_mode == 'local')
            x_partner_test = x_test[test_idx]
            y_partner_test = y_test[test_idx]

            partner = Partner(
                x_partner_train, x_partner_test, y_partner_train, y_partner_test, str(partner_id)
            )
            self.partners_list.append(partner)
            partner_id += 1

        # Check coherence of partners_list versus partners_count
        assert len(self.partners_list) == self.partners_count

        # Check coherence of number of mini-batches versus smaller partner
        assert self.minibatch_count <= (min(self.amounts_per_partner) * len(x_train))

        # Print the description of the scenario configured
        print("\n### Description of data scenario configured:")
        print("- Number of partners defined:", self.partners_count)
        print("- Data distribution scenario chosen:", self.samples_split_option)
        print(
            "- Test data distribution scenario chosen:", self.single_partner_test_mode
        )
        print("- Weighting option:", self.aggregation_weighting)
        print(
            "- Number of epochs and mini-batches: "
            + str(self.epoch_count)
            + " epochs and "
            + str(self.minibatch_count)
            + " mini-batches"
        )

        # Print and plot for controlling
        print("\n### Splitting data among partners:")
        for partner_index, partner in enumerate(self.partners_list):
            print("- partner #" + str(partner_index) + ":")
            print("  - Number of samples: " + str(len(partner.x_train)) + " train samples")
            print("  - y_train first 10 values: " + str(partner.y_train[:10]))
            print("  - y_train last 10 values: " + str(partner.y_train[-10:]))

        return 0

    def plot_data_distribution(self):

        for i, partner in enumerate(self.partners_list):

            plt.subplot(self.partners_count, 1, i + 1)  # TODO share y axis
            # print(partner.y_train) # VERBOSE
            # data = np.argmax(partner.y_train, axis=1)
            data_count = np.bincount(partner.y_train)

            # Fill with 0
            while len(data_count) < 10:
                data_count = np.append(data_count, 0)

            plt.bar(np.arange(0, 10), data_count)
            plt.ylabel("partner " + str(i))

        plt.suptitle("Data distribution")
        plt.xlabel("Digits")
        plt.savefig(self.save_folder / "data_distribution.png")

    def to_file(self):

        out = ""
        out += "Dataset name: " + self.dataset_name + "\n"
        out += "Number of data samples - train: " + str(len(self.x_train)) + "\n"
        out += "Number of data samples - test: " + str(len(self.x_test)) + "\n"
        out += "partners count: " + str(self.partners_count) + "\n"
        out += (
            "Percentages of data samples per partner: " + str(self.amounts_per_partner) + "\n"
        )
        out += (
            "random or stratified split of data samples: "
            + self.samples_split_option
            + "\n"
        )
        out += (
            "When training on a single partner, global or local testset: "
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

            for i in range(self.partners_count):

                # Scenario data
                dict_results["dataset_name"] = self.dataset_name
                dict_results["train_data_samples_count"] = len(self.x_train)
                dict_results["test_data_samples_count"] = len(self.x_test)
                dict_results["partners_count"] = self.partners_count
                dict_results["amounts_per_partner"] = self.amounts_per_partner
                dict_results["samples_split_option"] = self.samples_split_option
                dict_results["single_partner_test_mode"] = self.single_partner_test_mode
                dict_results["epoch_count"] = self.epoch_count
                dict_results["is_early_stopping"] = self.is_early_stopping
                dict_results["federated_test_score"] = self.federated_test_score
                dict_results["federated_computation_time_sec"] = self.federated_computation_time_sec
                dict_results["scenario_name"] = self.scenario_name
                dict_results["short_scenario_name"] = self.short_scenario_name
                dict_results["minibatch_count"] = self.minibatch_count
                dict_results["aggregation_weighting"] = self.aggregation_weighting

                # Contributivity data
                dict_results["contributivity_method"] = contrib.name
                dict_results["contributivity_scores"] = contrib.contributivity_scores
                dict_results["contributivity_stds"] = contrib.scores_std
                dict_results["computation_time_sec"] = contrib.computation_time_sec
                dict_results["first_characteristic_calls_count"] = contrib.first_charac_fct_calls_count
                # partner data
                dict_results["partner_id"] = i
                dict_results["amount_per_partner"] = self.amounts_per_partner[i]
                dict_results["contributivity_score"] = contrib.contributivity_scores[i]
                dict_results["contributivity_std"] = contrib.scores_std[i]

                df = df.append(dict_results, ignore_index=True)

        df.info()

        return df
