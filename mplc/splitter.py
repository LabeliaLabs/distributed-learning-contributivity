# -*- coding: utf-8 -*-
"""
This enables to split the original dataset between the partners.
"""
import operator
import random
from abc import ABC, abstractmethod

import numpy as np
from loguru import logger
from sklearn.preprocessing import LabelEncoder


class Splitter(ABC):
    name = 'Abstract Splitter'

    def __init__(self, amounts_per_partner, val_set='global', test_set='global', **kwargs):

        self.amounts_per_partner = amounts_per_partner
        self.test_set = test_set
        self.val_set = val_set

        self.dataset = None
        self.partners_list = None

    @property
    def partners_count(self):
        return len(self.partners_list)

    def __str__(self):
        return self.name

    def split(self, partners_list, dataset):
        self.dataset = dataset
        self.partners_list = partners_list

        logger.info("Splitting data among partners: starting now.")
        self._test_config_coherence()
        logger.info("Coherence of config parameters: OK.")

        logger.info("Train data split: starting now.")
        self._split_train()

        if self.val_set == 'local':
            logger.info("Validation data split: starting now.")
            self._split_val()

        if self.test_set == 'local':
            logger.info("Test data split: starting now.")
            self._split_test()

        for partner in self.partners_list:
            logger.info(
                f"Partner #{partner.id}: {partner.final_nb_samples} samples with labels {partner.labels}"
            )

    def _test_config_coherence(self):
        self._test_amounts_per_partner_total()
        self._test_amounts_per_partner_length()

    def _test_amounts_per_partner_total(self):
        if np.sum(self.amounts_per_partner) != 1:
            raise ValueError("The sum of the amount per partners you provided isn't equal to 1; it has to.")

    def _test_amounts_per_partner_length(self):
        if len(self.amounts_per_partner) != self.partners_count:
            raise AttributeError(f"The amounts_per_partner list should have a size ({len(self.amounts_per_partner)}) "
                                 f"equals to partners_count ({self.partners_count})")

    def _split_train(self):
        subsets = self._generate_subset(self.dataset.x_train, self.dataset.y_train)
        for idx, p in enumerate(self.partners_list):
            p.x_train, p.y_train = subsets[idx]

    def _split_val(self):
        subsets = self._generate_subset(self.dataset.x_val, self.dataset.y_val)
        for idx, p in enumerate(self.partners_list):
            p.x_val, p.y_val = subsets[idx]

    def _split_test(self):
        subsets = self._generate_subset(self.dataset.x_test, self.dataset.y_test)
        for idx, p in enumerate(self.partners_list):
            p.x_test, p.y_test = subsets[idx]

    @abstractmethod
    def _generate_subset(self, x, y):
        return [(x, y) for _ in self.partners_list]

    def __copy__(self):
        kwargs = self.__dict__.copy()
        del kwargs['dataset']
        del kwargs['partners_list']
        return self.__class__(**kwargs)

    def copy(self):
        return self.__copy__()


class FlexibleSplitter(Splitter):
    name = 'Fully Flexible Splitter'

    def __init__(self, amounts_per_partner, configuration, **kwargs):

        logger.info("Proceeding to a flexible split as requested. Please note that the flexible "
                    "split currently discards the amounts_per_partner (if provided) and infers amounts of samples "
                    "per partner from the samples_split_configuration provided.")

        # First we re-assemble the split configuration per cluster
        self.configuration = configuration
        self.split_configuration = configuration
        self.samples_split_grouped_by_cluster = list(zip(*configuration))

        # Init of the superclass to inherit its methods
        super().__init__(amounts_per_partner, **kwargs)

    def _test_config_coherence(self):

        # First, we test if the splitter configuration is coherent with the number of partners
        if len(self.split_configuration) != self.partners_count:
            raise AttributeError(f"The split configuration should have a size ({len(self.split_configuration)}) "
                                 f"equals to partners_count ({self.partners_count})")

        # Second, we test for each class that the amount of samples split across partners is <= 100%
        for idx, cluster in enumerate(self.samples_split_grouped_by_cluster):
            if np.sum(cluster) > 1:
                raise ValueError(f"Amounts of samples of class {idx} split among partners exceed 100%, "
                                 f"the dataset split cannot be performed.")

    def _generate_subset(self, x, y):

        # Convert raw labels in y to simplify operations on the dataset
        lb = LabelEncoder()
        y_str = lb.fit_transform([str(label) for label in y])
        labels = list(set(y_str))

        # Split the datasets (x and y) into subsets of samples of each label (called "clusters")
        x_for_cluster, y_for_cluster, nb_samples_per_cluster = {}, {}, {}
        for label in labels:
            idx_in_full_set = np.where(y_str == label)
            x_for_cluster[label] = x[idx_in_full_set]
            y_for_cluster[label] = y[idx_in_full_set]
            nb_samples_per_cluster[label] = len(y_for_cluster[label])

        # Assemble datasets per partner by looping over partners and labels
        res = []
        nb_samples_split = []
        for p_idx, p in enumerate(self.partners_list):

            list_arrays_x, list_arrays_y = [], []

            for idx, label in enumerate(labels):
                nb_samples_to_pick = int(nb_samples_per_cluster[label] * self.samples_split_grouped_by_cluster[idx][
                    p_idx])
                list_arrays_x.append(x_for_cluster[label][:nb_samples_to_pick])
                x_for_cluster[label] = x_for_cluster[label][nb_samples_to_pick:]
                list_arrays_y.append(y_for_cluster[label][:nb_samples_to_pick])
                y_for_cluster[label] = y_for_cluster[label][nb_samples_to_pick:]

            res.append((np.concatenate(list_arrays_x), np.concatenate(list_arrays_y)))
            nb_samples_split.append(len(np.concatenate(list_arrays_y)))

        # Log the relative amounts of samples split among partners
        total_nb_samples_split = np.sum(nb_samples_split)
        relative_nb_samples = [round(nb / total_nb_samples_split, 2) for nb in nb_samples_split]
        logger.info(f"Partners' relative number of samples: {relative_nb_samples}")

        return res


class RandomSplitter(Splitter):
    name = 'Random Splitter'

    def _generate_subset(self, x, y):
        if self.partners_count == 1:
            return [(x, y)]
        else:
            splitting_indices = (np.cumsum(self.amounts_per_partner)[:-1] * len(y)).astype(int)
            idxs = np.arange(len(y))
            np.random.shuffle(idxs)
            idx_list = np.split(idxs, splitting_indices)
            res = []
            for slice_idx in idx_list:
                res.append((x[slice_idx], y[slice_idx]))
            return res


class StratifiedSplitter(Splitter):
    name = 'Stratified Splitter'

    def _generate_subset(self, x, y):
        if self.partners_count == 1:
            return [(x, y)]
        else:
            y_str = LabelEncoder().fit_transform([str(label) for label in y])
            splitting_indices = (np.cumsum(self.amounts_per_partner)[:-1] * len(y)).astype(int)
            idxs = y_str.argsort()
            idx_list = np.split(idxs, splitting_indices)
            res = []
            for slice_idx in idx_list:
                res.append((x[slice_idx], y[slice_idx]))
            return res


class AdvancedSplitter(Splitter):
    name = 'Advanced Splitter'

    def __init__(self, amounts_per_partner, configuration, **kwargs):
        self.configuration = configuration
        self.num_clusters, self.specific_shared = list(zip(*configuration))
        super().__init__(amounts_per_partner, **kwargs)

    def __str__(self):
        return self.name + str(list(zip(self.num_clusters, self.specific_shared)))

    def _generate_subset(self, x, y):
        lb = LabelEncoder()
        y_str = lb.fit_transform([str(label) for label in y])
        labels = list(set(y_str))
        np.random.shuffle(labels)
        nb_diff_labels = len(lb.classes_)

        for p_id, p in enumerate(self.partners_list):
            p.cluster_count_param = self.num_clusters[p_id]
            p.cluster_split_option = self.specific_shared[p_id]

        partners_with_specific_clusters = [p for p, option in zip(self.partners_list, self.specific_shared) if
                                           option == 'specific']
        partners_with_specific_clusters.sort(key=operator.attrgetter("cluster_count_param"), reverse=True)
        partners_with_shared_clusters = [p for p, option in zip(self.partners_list, self.specific_shared) if
                                         option == 'shared']
        partners_with_shared_clusters.sort(key=operator.attrgetter("cluster_count_param"), reverse=True)

        specific_clusters_count = sum(
            [p.cluster_count_param for p in partners_with_specific_clusters]
        )
        if partners_with_shared_clusters:
            shared_clusters_count = max(
                [p.cluster_count_param for p in partners_with_shared_clusters]
            )
        else:
            shared_clusters_count = 0
        assert (
                specific_clusters_count + shared_clusters_count <= nb_diff_labels
        ), f"Error: data samples from the initial dataset are split in clusters per data labels - " \
           f"Incompatibility between the split arguments and the dataset provided: " \
           f"{specific_clusters_count + shared_clusters_count}, {nb_diff_labels} " \
           f"- Example: ['advanced', [[7, 'shared'], [6, 'shared'], [2, 'specific'], [1, 'specific']]] means 7" \
           f" shared clusters and 2 + 1 = 3 specific clusters " \
           f"==> This scenario can't work with a dataset with less than 10 labels"

        x_for_cluster, y_for_cluster, nb_samples_per_cluster = {}, {}, {}
        for label in labels:
            idx_in_full_set = np.where(y_str == label)
            x_for_cluster[label] = x[idx_in_full_set]
            y_for_cluster[label] = y[idx_in_full_set]
            nb_samples_per_cluster[label] = len(y_for_cluster[label])

        # For each partner compose the list of clusters from which they will draw data samples
        index = 0
        for p in partners_with_specific_clusters:
            p.clusters_list = labels[index: index + p.cluster_count_param]
            index += p.cluster_count_param

        shared_clusters = labels[index: index + shared_clusters_count]
        for p in partners_with_shared_clusters:
            p.clusters_list = random.sample(shared_clusters, k=p.cluster_count_param)

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
            nb_samples_requested = int(self.amounts_per_partner[p.id] * len(y))
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
                self.amounts_per_partner[p.id] * len(y) * resize_factor_specific
            )
            initial_amount_resized_per_cluster = int(
                initial_amount_resized / p.cluster_count_param
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
        final_nb_samples_per_partner = [int(amount * len(y) * final_resize_factor)
                                        for amount in self.amounts_per_partner]
        final_nb_samples_p_cluster = [int(nb_samples / p.cluster_count_param)
                                      for nb_samples, p in zip(final_nb_samples_per_partner, self.partners_list)]

        total_nb_samples = sum(final_nb_samples_per_partner)
        relative_nb_sample = [nb / total_nb_samples for nb in final_nb_samples_per_partner]

        # Partners receive their subsets
        shared_clusters_index = dict.fromkeys(shared_clusters, 0)
        res = []
        for p in self.partners_list:

            list_arrays_x, list_arrays_y = [], []

            if p in partners_with_shared_clusters:
                for cl in p.clusters_list:
                    idx = shared_clusters_index[cl]
                    list_arrays_x.append(
                        x_for_cluster[cl][idx: idx + final_nb_samples_p_cluster[p.id]]
                    )
                    list_arrays_y.append(
                        y_for_cluster[cl][idx: idx + final_nb_samples_p_cluster[p.id]]
                    )
                    shared_clusters_index[cl] += final_nb_samples_p_cluster[p.id]
            elif p in partners_with_specific_clusters:
                for cl in p.clusters_list:
                    list_arrays_x.append(
                        x_for_cluster[cl][: final_nb_samples_p_cluster[p.id]]
                    )
                    list_arrays_y.append(
                        y_for_cluster[cl][: final_nb_samples_p_cluster[p.id]]
                    )
            res.append((np.concatenate(list_arrays_x), np.concatenate(list_arrays_y)))

        logger.info(
            f"Partners' relative number of samples: {[round(nb, 2) for nb in relative_nb_sample]} "
            f"(versus initially configured: {self.amounts_per_partner})"
        )

        return res


IMPLEMENTED_SPLITTERS = {
    'flexible': FlexibleSplitter,
    'random': RandomSplitter,
    'stratified': StratifiedSplitter,
    'advanced': AdvancedSplitter
}
