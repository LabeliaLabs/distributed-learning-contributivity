# -*- coding: utf-8 -*-
"""
This enables to parameterize a desired scenario to mock a multi-partner ML project.
"""

import datetime
import operator
import random
import re
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from . import contributivity, constants
from . import dataset as dataset_module
from .corruption import Corruption, NoCorruption, IMPLEMENTED_CORRUPTION, Duplication
from .mpl_utils import AGGREGATORS
from .multi_partner_learning import MULTI_PARTNER_LEARNING_APPROACHES
from .partner import Partner


class Scenario:
    def __init__(
            self,
            partners_count,
            amounts_per_partner,
            dataset=None,
            dataset_name=constants.MNIST,
            dataset_proportion=1,
            samples_split_option=None,
            corruption_parameters=None,
            init_model_from="random_initialization",
            multi_partner_learning_approach="fedavg",
            aggregation_weighting="data-volume",
            gradient_updates_per_pass_count=constants.DEFAULT_GRADIENT_UPDATES_PER_PASS_COUNT,
            minibatch_count=constants.DEFAULT_BATCH_COUNT,
            epoch_count=constants.DEFAULT_EPOCH_COUNT,
            is_early_stopping=True,
            contributivity_methods=None,
            is_quick_demo=False,
            save_path=None,
            scenario_id=1,
            **kwargs,
    ):
        """

        :param partners_count: int, number of partners. Example: partners_count = 3
        :param amounts_per_partner:  [float]. Fractions of the
        original dataset each partner receives to mock a collaborative ML scenario where each partner provides data
        for the ML training.
        :param dataset: dataset.Dataset object. Use it if you want to use your own dataset, otherwise use dataset_name.
        :param dataset_name: str. 'mnist', 'cifar10', 'esc50' and 'titanic' are currently supported (default: mnist)
        :param dataset_proportion: float (default: 1)
        :param samples_split_option: ['basic', 'random'] (default),
                                     ['basic', 'stratified']
                                     or ['advanced', [[nb of clusters (int), 'shared' or 'specific']]].
        :param corruption_parameters: list of map. Enables to artificially corrupt the data of one or several partners.
                                   The size of the list must be equal to the number of partners.
                                   Items of the list must be mapping. The possible keys, values are the following:
                                    'corruption_method': 'not_corrupted' (default),
                                                         'duplication',
                                                         'permutation',
                                                         'permutation-circular',
                                                         'random',
                                                         'random-uniform',
                                                         'redundancy'
                                    Indicating the corruption method to use to corrupt the partner's data
                                    'proportion_corrupted': 1. (default), float between 0. and 1. indicating the
                                                                          proportion of partner's data to corrupt
                                    'duplicated_partner_id': Partner_id used by the duplicate corruption method.
                                                          If not provided, a random partner amongst those
                                                          with enough data will be selected

                                   Example with 3 partners.
                                   [{}, {'corruption_method':'permutation'},
                                    {'corruption_method':'duplication',
                                    'proportion_corrupted': 0.4,
                                    'duplicated_partner_id': 0}]
        :param init_model_from: None (default) or path
        :param multi_partner_learning_approach: 'fedavg' (default), 'seq-pure', 'seq-with-final-agg' or 'seqavg'
                                                Define the multi-partner learning approach
        :param aggregation_weighting: 'data_volume' (default), 'uniform' or 'local_score'
        :param gradient_updates_per_pass_count: int
        :param minibatch_count: int
        :param epoch_count: int
        :param is_early_stopping: boolean. Stop the training if scores on val_set reach a plateau
        :param contributivity_methods: A declarative list `[]` of the contributivity measurement methods to be executed.
        :param is_quick_demo: boolean. Useful for debugging
        :param save_path: path where to save the scenario outputs. By default, they are not saved!
        :param scenario_id: str
        :param **kwargs:
        """

        # ---------------------------------------------------------------------
        # Initialization of the dataset defined in the config of the experiment
        # ---------------------------------------------------------------------

        # Raise Exception if unknown parameters in the config of the scenario

        params_known = [
            "dataset",
            "dataset_name",
            "dataset_proportion",
        ]  # Dataset related
        params_known += [
            "contributivity_methods",
            "multi_partner_learning_approach",
            "aggregation",
        ]  # federated learning related
        params_known += [
            "partners_count",
            "amounts_per_partner",
            "corruption_parameters",
            "samples_split_option",
        ]  # Partners related
        params_known += [
            "gradient_updates_per_pass_count",
            "epoch_count",
            "minibatch_count",
            "is_early_stopping",
        ]  # Computation related
        params_known += ["init_model_from"]  # Model related
        params_known += ["is_quick_demo"]
        params_known += ["save_path",
                         "scenario_name",
                         "repeat_count"]

        unrecognised_parameters = [x for x in kwargs.keys() if x not in params_known]
        if len(unrecognised_parameters) > 0:
            for x in unrecognised_parameters:
                logger.debug(f"Unrecognised parameter: {x}")
            raise Exception(
                f"Unrecognised parameters {unrecognised_parameters}, check your configuration"
            )

        # Get and verify which dataset is configured
        if isinstance(dataset, dataset_module.Dataset):
            self.dataset = dataset
        else:
            # Reference the module corresponding to the dataset selected and initialize the Dataset object
            if dataset_name == constants.MNIST:  # default
                self.dataset = dataset_module.Mnist()
            elif dataset_name == constants.CIFAR10:
                self.dataset = dataset_module.Cifar10()
            elif dataset_name == constants.TITANIC:
                self.dataset = dataset_module.Titanic()
            elif dataset_name == constants.ESC50:
                self.dataset = dataset_module.Esc50()
            elif dataset_name == constants.IMDB:
                self.dataset = dataset_module.Imdb()
            else:
                raise Exception(
                    f"Dataset named '{dataset_name}' is not supported (yet). You can construct your own "
                    f"dataset object, or even add it by contributing to the project !"
                )
            logger.debug(f"Dataset selected: {self.dataset.name}")

        # Proportion of the dataset the computation will used
        self.dataset_proportion = dataset_proportion
        assert (
                self.dataset_proportion > 0
        ), "Error in the config file, dataset_proportion should be > 0"
        assert (
                self.dataset_proportion <= 1
        ), "Error in the config file, dataset_proportion should be <= 1"

        if self.dataset_proportion < 1:
            self.dataset.shorten_dataset_proportion(self.dataset_proportion)
        else:
            logger.debug("The full dataset will be used (dataset_proportion is configured to 1)")

        # --------------------------------------
        #  Definition of collaborative scenarios
        # --------------------------------------

        # Partners mock different partners in a collaborative data science project
        self.partners_list = []  # List of all partners defined in the scenario
        self.partners_count = partners_count  # Number of partners in the scenario

        # For configuring the respective sizes of the partners' datasets
        # (% of samples of the dataset for each partner, ...
        # ... has to sum to 1, and number of items has to equal partners_count)
        self.amounts_per_partner = amounts_per_partner

        # For configuring if data samples are split between partners randomly or in a stratified way...
        # ... so that they cover distinct areas of the samples space
        if samples_split_option:
            (self.samples_split_type, self.samples_split_description) = samples_split_option
        else:
            (self.samples_split_type, self.samples_split_description) = ("basic", "random")  # default

        # For configuring if the data of the partners are corrupted or not (useful for testing contributivity measures)
        if corruption_parameters:
            self.corruption_parameters = list(
                map(lambda x: x if isinstance(x, Corruption) else IMPLEMENTED_CORRUPTION[x](),
                    corruption_parameters))
        else:
            self.corruption_parameters = [NoCorruption() for _ in range(self.partners_count)]  # default

        # ---------------------------------------------------
        #  Configuration of the distributed learning approach
        # ---------------------------------------------------

        self.mpl = None

        # Multi-partner learning approach
        self.multi_partner_learning_approach = multi_partner_learning_approach
        try:
            self._multi_partner_learning_approach = MULTI_PARTNER_LEARNING_APPROACHES[
                multi_partner_learning_approach]
        except KeyError:
            text_error = f"Multi-partner learning approach '{multi_partner_learning_approach}' is not a valid "
            text_error += "approach. List of supported approach : "
            for key in MULTI_PARTNER_LEARNING_APPROACHES.keys():
                text_error += f"{key}, "
            raise KeyError(text_error)

        # Define how federated learning aggregation steps are weighted...
        # ... Toggle between 'uniform' (default) and 'data_volume'
        self.aggregation = aggregation_weighting
        try:
            self._aggregation = AGGREGATORS[aggregation_weighting]
        except KeyError:
            raise ValueError(f"aggregation approach '{aggregation_weighting}' is not a valid approach. ")

        # Number of epochs, mini-batches and fit_batches in ML training
        self.epoch_count = epoch_count
        assert (
                self.epoch_count > 0
        ), "Error: in the provided config file, epoch_count should be > 0"

        self.minibatch_count = minibatch_count
        assert (
                self.minibatch_count > 0
        ), "Error: in the provided config file, minibatch_count should be > 0"

        self.gradient_updates_per_pass_count = gradient_updates_per_pass_count
        assert self.gradient_updates_per_pass_count > 0, (
            "Error: in the provided config file, "
            "gradient_updates_per_pass_count should be > 0 "
        )

        # Early stopping stops ML training when performance increase is not significant anymore
        # It is used to optimize the number of epochs and the execution time
        self.is_early_stopping = is_early_stopping

        # Model used to initialise model
        self.init_model_from = init_model_from
        if init_model_from == "random_initialization":
            self.use_saved_weights = False
        else:
            self.use_saved_weights = True

        # -----------------------------------------------------------------
        #  Configuration of contributivity measurement contributivity_methods to be tested
        # -----------------------------------------------------------------

        # List of contributivity measures selected and computed in the scenario
        self.contributivity_list = []

        # Contributivity contributivity_methods
        self.contributivity_methods = []
        if contributivity_methods is not None:
            for method in contributivity_methods:
                if method in constants.CONTRIBUTIVITY_METHODS:
                    self.contributivity_methods.append(method)
                else:
                    raise Exception(f"Contributivity method '{method}' is not in contributivity_methods list.")

        # -------------
        # Miscellaneous
        # -------------

        # Misc.
        self.scenario_id = scenario_id
        self.repeat_count = kwargs.get('repeat_count', 1)

        # The quick demo parameters overwrites previously defined parameters to make the scenario faster to compute
        self.is_quick_demo = is_quick_demo
        if self.is_quick_demo and self.dataset_proportion < 1:
            raise Exception("Don't start a quick_demo without the full dataset")

        if self.is_quick_demo:
            # Use less data and/or less epochs to speed up the computations
            if len(self.dataset.x_train) > constants.TRAIN_SET_MAX_SIZE_QUICK_DEMO:
                index_train = np.random.choice(
                    self.dataset.x_train.shape[0],
                    constants.TRAIN_SET_MAX_SIZE_QUICK_DEMO,
                    replace=False,
                )
                index_val = np.random.choice(
                    self.dataset.x_val.shape[0],
                    constants.VAL_SET_MAX_SIZE_QUICK_DEMO,
                    replace=False,
                )
                index_test = np.random.choice(
                    self.dataset.x_test.shape[0],
                    constants.TEST_SET_MAX_SIZE_QUICK_DEMO,
                    replace=False,
                )
                self.dataset.x_train = self.dataset.x_train[index_train]
                self.dataset.y_train = self.dataset.y_train[index_train]
                self.dataset.x_val = self.dataset.x_val[index_val]
                self.dataset.y_val = self.dataset.y_val[index_val]
                self.dataset.x_test = self.dataset.x_test[index_test]
                self.dataset.y_test = self.dataset.y_test[index_test]
            self.epoch_count = 3
            self.minibatch_count = 2

        # -----------------
        # Output parameters
        # -----------------

        now = datetime.datetime.now()
        now_str = now.strftime("%Y-%m-%d_%Hh%M")
        self.scenario_name = kwargs.get('scenario_name',
                                        f"scenario_{self.scenario_id}_repeat_{self.repeat_count}_{now_str}_"
                                        f"{uuid.uuid4().hex[:3]}")  # to distinguish identical names
        if re.search(r'\s', self.scenario_name):
            raise ValueError(
                f'The scenario name "{self.scenario_name}"cannot be written with space character, please use '
                f'underscore or dash.')
        self.short_scenario_name = f"{self.partners_count}_{self.amounts_per_partner}"

        if save_path is not None:
            self.save_folder = Path(save_path) / self.scenario_name
        else:
            self.save_folder = None
        # -----------------------
        # Provision the scenario
        # -----------------------

        self.instantiate_scenario_partners()
        self.split_data()
        self.compute_batch_sizes()
        self.apply_data_alteration_configuration()
        self.log_scenario_description()

    @property
    def nb_samples_used(self):
        if len(self.partners_list) == 0:
            return len(self.dataset.x_train)
        else:
            return sum([p.final_nb_samples for p in self.partners_list])

    @property
    def final_relative_nb_samples(self):
        return [p.final_nb_samples / self.nb_samples_used for p in self.partners_list]

    def copy(self, **kwargs):
        params = self.__dict__.copy()
        for key in ['partners_list',
                    'samples_split_type',
                    'samples_split_description',
                    'mpl',
                    '_multi_partner_learning_approach',
                    '_aggregation',
                    'use_saved_weights',
                    'contributivity_list',
                    'scenario_name',
                    'short_scenario_name',
                    'save_folder']:
            del params[key]
        if 'is_quick_demo' in kwargs and kwargs['is_quick_demo'] != self.is_quick_demo:
            raise ValueError("Attribute 'is_quick_demo' cannot be modified between copies.")
        params['save_path'] = self.save_folder.parents[0]
        params.update(kwargs)

        return Scenario(**params)

    def log_scenario_description(self):
        """Log the description of the scenario configured"""

        # Describe scenario
        logger.info("Description of data scenario configured:")
        logger.info(f"   Number of partners defined: {self.partners_count}")
        logger.info(f"   Data distribution scenario chosen: {self.samples_split_description}")
        logger.info(f"   Multi-partner learning approach: {self.multi_partner_learning_approach}")
        logger.info(f"   Weighting option: {self.aggregation}")
        logger.info(f"   Iterations parameters: "
                    f"{self.epoch_count} epochs > "
                    f"{self.minibatch_count} mini-batches > "
                    f"{self.gradient_updates_per_pass_count} gradient updates per pass")

        # Describe data
        logger.info(f"Data loaded: {self.dataset.name}")
        if self.is_quick_demo:
            logger.info("   Quick demo configuration: number of data samples and epochs "
                        "are limited to speed up the run")
        logger.info(
            f"   {len(self.dataset.x_train)} train data with {len(self.dataset.y_train)} labels"
        )
        logger.info(
            f"   {len(self.dataset.x_val)} val data with {len(self.dataset.y_val)} labels"
        )
        logger.info(
            f"   {len(self.dataset.x_test)} test data with {len(self.dataset.y_test)} labels"
        )

    def append_contributivity(self, contributivity_method):
        self.contributivity_list.append(contributivity_method)

    def instantiate_scenario_partners(self):
        """Create the partners_list"""
        if len(self.partners_list) > 0:
            raise Exception('Partners have already been initialized')
        self.partners_list = [Partner(i, corruption=self.corruption_parameters[i]) for i in range(self.partners_count)]

    def split_data(self, is_logging_enabled=True):
        if self.samples_split_type == "basic":
            self.split_data_basic(is_logging_enabled)
        elif self.samples_split_type == "advanced":
            self.split_data_advanced(is_logging_enabled)

    def split_data_advanced(self, is_logging_enabled=True):
        """Advanced split: Populates the partners with their train and test data (not pre-processed)"""

        y_train = LabelEncoder().fit_transform([str(y) for y in self.dataset.y_train])
        partners_list = self.partners_list
        amounts_per_partner = self.amounts_per_partner
        advanced_split_description = self.samples_split_description

        # Compose the lists of partners with data samples from shared clusters and those with specific clusters
        for p in partners_list:
            p.cluster_count = int(advanced_split_description[p.id][0])
            p.cluster_split_option = advanced_split_description[p.id][1]
        partners_with_shared_clusters = [
            p for p in partners_list if p.cluster_split_option == "shared"
        ]
        partners_with_specific_clusters = [
            p for p in partners_list if p.cluster_split_option == "specific"
        ]
        partners_with_shared_clusters.sort(
            key=operator.attrgetter("cluster_count"), reverse=True
        )
        partners_with_specific_clusters.sort(
            key=operator.attrgetter("cluster_count"), reverse=True
        )

        # Compose the list of different labels in the dataset
        labels = list(set(y_train))
        random.seed(42)
        random.shuffle(labels)

        # Check coherence of the split option:
        nb_diff_labels = len(labels)
        specific_clusters_count = sum(
            [p.cluster_count for p in partners_with_specific_clusters]
        )
        if partners_with_shared_clusters:
            shared_clusters_count = max(
                [p.cluster_count for p in partners_with_shared_clusters]
            )
        else:
            shared_clusters_count = 0
        assert (
                specific_clusters_count + shared_clusters_count <= nb_diff_labels
        ), "Error: data samples from the \
            initial dataset are split in clusters per data labels - Incompatibility between the split arguments \
            and the dataset provided \
            - Example: ['advanced', [[7, 'shared'], [6, 'shared'], [2, 'specific'], [1, 'specific']]] \
            means 7 shared clusters and 2 + 1 = 3 specific clusters ==> This scenario can't work with a dataset with \
            less than 10 labels"

        # Stratify the dataset into clusters per labels
        x_train_for_cluster, y_train_for_cluster, nb_samples_per_cluster = {}, {}, {}
        for label in labels:
            idx_in_full_trainset = np.where(y_train == label)
            x_train_for_cluster[label] = self.dataset.x_train[idx_in_full_trainset]
            y_train_for_cluster[label] = self.dataset.y_train[idx_in_full_trainset]
            nb_samples_per_cluster[label] = len(y_train_for_cluster[label])

        # For each partner compose the list of clusters from which they will draw data samples
        index = 0
        for p in partners_with_specific_clusters:
            p.clusters_list = labels[index: index + p.cluster_count]
            index += p.cluster_count

        shared_clusters = labels[index: index + shared_clusters_count]
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
            nb_available_samples = sum(
                [nb_samples_per_cluster[cl] for cl in p.clusters_list]
            )
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
            initial_amount_resized = int(
                amounts_per_partner[p.id] * len(y_train) * resize_factor_specific
            )
            initial_amount_resized_per_cluster = int(
                initial_amount_resized / p.cluster_count
            )
            for cl in p.clusters_list:
                nb_samples_needed_per_cluster[cl] += initial_amount_resized_per_cluster
        for cl in nb_samples_needed_per_cluster:
            resize_factor_shared = min(
                resize_factor_shared,
                nb_samples_per_cluster[cl] / nb_samples_needed_per_cluster[cl],
            )

        # Compute the final resize factor
        final_resize_factor = resize_factor_specific * resize_factor_shared

        # Size correctly each partner's subset. For each partner:
        for p in partners_list:
            p.final_nb_samples = int(
                amounts_per_partner[p.id] * len(y_train) * final_resize_factor
            )
            p.final_nb_samples_p_cluster = int(p.final_nb_samples / p.cluster_count)
        # Partners receive their subsets
        shared_clusters_index = dict.fromkeys(shared_clusters, 0)
        for p in partners_list:

            list_arrays_x, list_arrays_y = [], []

            if p in partners_with_shared_clusters:
                for cl in p.clusters_list:
                    idx = shared_clusters_index[cl]
                    list_arrays_x.append(
                        x_train_for_cluster[cl][idx: idx + p.final_nb_samples_p_cluster]
                    )
                    list_arrays_y.append(
                        y_train_for_cluster[cl][idx: idx + p.final_nb_samples_p_cluster]
                    )
                    shared_clusters_index[cl] += p.final_nb_samples_p_cluster
            elif p in partners_with_specific_clusters:
                for cl in p.clusters_list:
                    list_arrays_x.append(
                        x_train_for_cluster[cl][: p.final_nb_samples_p_cluster]
                    )
                    list_arrays_y.append(
                        y_train_for_cluster[cl][: p.final_nb_samples_p_cluster]
                    )

            p.x_train = np.concatenate(list_arrays_x)
            p.y_train = np.concatenate(list_arrays_y)

            # Create local validation and test datasets from the partner train data
            p.x_train, p.x_val, p.y_train, p.y_val = train_test_split(
                p.x_train, p.y_train, test_size=0.1, random_state=42
            )
            p.x_train, p.x_test, p.y_train, p.y_test = train_test_split(
                p.x_train, p.y_train, test_size=0.1, random_state=42
            )

        # Check coherence of number of mini-batches versus partner with small dataset
        assert self.minibatch_count <= min(
            [len(p.x_train) for p in self.partners_list]
        ), "Error: in the provided \
            config file and the provided dataset, a partner doesn't have enough data samples to create the minibatches"

        if is_logging_enabled:
            logger.info("Splitting data among partners (advanced split):")
            logger.info(f"Nb of samples split amongst partners: {self.nb_samples_used}")
            logger.debug(
                f"Partners' relative nb of samples: {[round(p, 2) for p in self.final_relative_nb_samples]} "
                f"(versus initially configured: {amounts_per_partner})"
            )
            for partner in self.partners_list:
                logger.info(
                    f"Partner #{partner.id}: {len(partner.x_train)} "
                    f"samples with labels {partner.clusters_list}"
                )

        return 0

    def split_data_basic(self, is_logging_enabled=True):
        """Populates the partners with their train and test data (not pre-processed)"""

        y_train = LabelEncoder().fit_transform([str(y) for y in self.dataset.y_train])

        # Configure the desired splitting scenario - Datasets sizes
        # Should the partners receive an equivalent amount of samples each...
        # ... or receive different amounts?

        # Check the percentages of samples per partner and control its coherence
        assert (
                len(self.amounts_per_partner) == self.partners_count
        ), "Error: in the provided config file, \
            amounts_per_partner list should have a size equals to partners_count"
        assert (
                np.sum(self.amounts_per_partner) == 1
        ), "Error: in the provided config file, \
            amounts_per_partner argument: the sum of the proportions you provided isn't equal to 1"

        # Then we parameterize this via the splitting_indices to be passed to np.split
        # This is to transform the percentages from the scenario configuration into indices where to split the data
        if self.partners_count == 1:
            splitting_indices_train = 1
        else:
            splitting_indices = np.empty((self.partners_count - 1,))
            splitting_indices[0] = self.amounts_per_partner[0]
            for i in range(self.partners_count - 2):
                splitting_indices[i + 1] = (
                        splitting_indices[i] + self.amounts_per_partner[i + 1]
                )
            splitting_indices_train = (splitting_indices * len(y_train)).astype(int)

        # Configure the desired data distribution scenario
        # In the 'stratified' scenario we sort by labels
        if self.samples_split_description == "stratified":
            # Sort by labels
            train_idx = y_train.argsort()

        # In the 'random' scenario we shuffle randomly the indexes
        elif self.samples_split_description == "random":
            train_idx = np.arange(len(y_train))
            np.random.seed(42)
            np.random.shuffle(train_idx)

        # If neither 'stratified' nor 'random', we raise an exception
        else:
            raise NameError(
                "This samples_split option ["
                + self.samples_split_description
                + "] is not recognized."
            )

        # Do the partitioning among partners according to desired scenarios
        # Split data between partners
        train_idx_idx_list = np.split(train_idx, splitting_indices_train)

        # Populate partners
        partner_idx = 0
        for train_idx in train_idx_idx_list:
            p = self.partners_list[partner_idx]

            # Finalize selection of train data
            # Populate the partner's train dataset
            p.x_train = self.dataset.x_train[train_idx, :]
            p.y_train = self.dataset.y_train[train_idx]

            # Create local validation and test datasets from the partner train data
            (
                p.x_train,
                p.x_test,
                p.y_train,
                p.y_test,
            ) = self.dataset.train_test_split_local(p.x_train, p.y_train)
            p.x_train, p.x_val, p.y_train, p.y_val = self.dataset.train_val_split_local(
                p.x_train, p.y_train
            )

            # Update other attributes from partner
            p.final_nb_samples = len(p.x_train)
            p.clusters_list = list(set(y_train[train_idx]))

            # Move on to the next partner
            partner_idx += 1

        # Check coherence of number of mini-batches versus smaller partner
        assert self.minibatch_count <= (
                min(self.amounts_per_partner) * len(y_train)
        ), "Error: in the provided config \
            file and dataset, a partner doesn't have enough data samples to create the minibatches"

        if is_logging_enabled:
            logger.info("Splitting data among partners (simple split):")
            logger.info(f"Nb of samples split amongst partners: {self.nb_samples_used}")
            for partner in self.partners_list:
                logger.info(
                    f"Partner #{partner.id}: {partner.final_nb_samples} samples with labels {partner.clusters_list}"
                )

        return 0

    def plot_data_distribution(self):
        lb = LabelEncoder().fit([str(y) for y in self.dataset.y_train])
        for i, partner in enumerate(self.partners_list):

            plt.subplot(self.partners_count, 1, i + 1)  # TODO share y axis
            data_count = np.bincount(lb.transform([str(y) for y in partner.y_train]))

            # Fill with 0
            while len(data_count) < self.dataset.num_classes:
                data_count = np.append(data_count, 0)

            plt.bar(np.arange(0, self.dataset.num_classes), data_count)
            plt.ylabel("partner " + str(partner.id))

        plt.suptitle("Data distribution")
        plt.xlabel("Digits")

        (self.save_folder / 'graphs').mkdir(exist_ok=True)
        plt.savefig(self.save_folder / "graphs" / "data_distribution.png")
        plt.close()

    def compute_batch_sizes(self):

        # For each partner we compute the batch size in multi-partner and single-partner setups
        batch_size_min = 1
        batch_size_max = constants.MAX_BATCH_SIZE

        if self.partners_count == 1:
            p = self.partners_list[0]
            batch_size = int(len(p.x_train) / self.gradient_updates_per_pass_count)
            p.batch_size = np.clip(batch_size, batch_size_min, batch_size_max)
        else:
            for p in self.partners_list:
                batch_size = int(
                    len(p.x_train)
                    / (self.minibatch_count * self.gradient_updates_per_pass_count)
                )
                p.batch_size = np.clip(batch_size, batch_size_min, batch_size_max)

        for p in self.partners_list:
            logger.debug(f"   Compute batch sizes, partner #{p.id}: {p.batch_size}")

    def apply_data_alteration_configuration(self):
        """perform corruption on partner if needed"""
        for partner in self.partners_list:
            if isinstance(partner.corruption, Duplication):
                if not partner.corruption.duplicated_partner_id:
                    data_volume = np.array([p.data_volume for p in self.partners_list if p.id != partner.id])
                    ids = np.array([p.id for p in self.partners_list if p.id != partner.id])
                    candidates = ids[data_volume >= partner.data_volume * partner.corruption.proportion]
                    partner.corruption.duplicated_partner_id = np.random.choice(candidates)
                partner.corruption.set_duplicated_partner(self.partners_list)
            partner.corrupt()

    def to_dataframe(self):

        df = pd.DataFrame()
        dict_results = {}

        # Scenario definition parameters
        dict_results["scenario_name"] = self.scenario_name
        dict_results["short_scenario_name"] = self.short_scenario_name
        dict_results["dataset_name"] = self.dataset.name
        dict_results["train_data_samples_count"] = len(self.dataset.x_train)
        dict_results["test_data_samples_count"] = len(self.dataset.x_test)
        dict_results["partners_count"] = self.partners_count
        dict_results["dataset_fraction_per_partner"] = self.amounts_per_partner
        dict_results["samples_split_description"] = self.samples_split_description
        dict_results["nb_samples_used"] = self.nb_samples_used
        dict_results["final_relative_nb_samples"] = self.final_relative_nb_samples

        # Multi-partner learning approach parameters
        dict_results["multi_partner_learning_approach"] = self.multi_partner_learning_approach
        dict_results["aggregation"] = self.aggregation
        dict_results["epoch_count"] = self.epoch_count
        dict_results["minibatch_count"] = self.minibatch_count
        dict_results["gradient_updates_per_pass_count"] = self.gradient_updates_per_pass_count
        dict_results["is_early_stopping"] = self.is_early_stopping
        dict_results["mpl_test_score"] = self.mpl.history.score
        dict_results["mpl_nb_epochs_done"] = self.mpl.history.nb_epochs_done
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
                dict_results["dataset_fraction_of_partner"] = self.amounts_per_partner[i]
                dict_results["contributivity_score"] = contrib.contributivity_scores[i]
                dict_results["contributivity_std"] = contrib.scores_std[i]

                df = df.append(dict_results, ignore_index=True)

        return df

    def run(self):

        # -----------------
        # Preliminary steps
        # -----------------
        if self.save_folder is not None:
            self.save_folder.mkdir()
            self.plot_data_distribution()
        logger.info(f"Now starting running scenario {self.scenario_name}")

        # -----------------------------------------------------
        # Instantiate and run the distributed learning approach
        # -----------------------------------------------------

        self.mpl = self._multi_partner_learning_approach(self, custom_name='main_mpl')
        self.mpl.fit()

        # -------------------------------------------------------------------------
        # Instantiate and run the contributivity measurement contributivity_methods
        # -------------------------------------------------------------------------

        for method in self.contributivity_methods:
            logger.info(f"{method}")
            contrib = contributivity.Contributivity(scenario=self)
            contrib.compute_contributivity(method)
            self.append_contributivity(contrib)
            logger.info(f"Evaluating contributivity with {method}: {contrib}")

        return 0
