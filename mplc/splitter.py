# -*- coding: utf-8 -*-
"""
This enables to split the original dataset between the partners.
"""
import operator
from abc import ABC, abstractmethod

import numpy as np
from sklearn.preprocessing import LabelEncoder


class Splitter(ABC):
    def __init__(self, dataset, partners_list, amounts_per_partner, val_set='global', test_set='global'):

        self.amounts_per_partner = amounts_per_partner
        self.test_set = test_set
        self.val_set = val_set
        self.dataset = dataset
        self.partners_list = partners_list

        # Check the percentages of samples per partner and control its coherence
        assert (
                len(self.amounts_per_partner) == self.partners_count
        ), "Error: in the provided config file, \
                    amounts_per_partner list should have a size equals to partners_count"
        assert (
                np.sum(self.amounts_per_partner) == 1
        ), "Error: in the provided config file, \
                    amounts_per_partner argument: the sum of the proportions you provided isn't equal to 1"

    @property
    def partners_count(self):
        return len(self.partners_list)

    def split(self):
        self._split_train()
        if self.val_set == 'local':
            self._split_val()
        if self.test_set == 'local':
            self._split_test()

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


class RandomSplitter(Splitter):
    def _generate_subset(self, x, y):
        if self.partners_count == 1:
            return [(x, y)]
        else:
            y = LabelEncoder().fit_transform([str(label) for label in y])
            splitting_indices = (np.cumsum(self.amounts_per_partner)[:-1] * len(y)).astype(int)
            idxs = np.arange(len(y))
            np.random.shuffle(idxs)
            idx_list = np.split(idxs, splitting_indices)
            res = []
            for slice_idx in idx_list:
                res.append((x[slice_idx], y[slice_idx]))
            return res


class StratifiedSplitter(Splitter):
    def _generate_subset(self, x, y):
        if self.partners_count == 1:
            return [(x, y)]
        else:
            y = LabelEncoder().fit_transform([str(label) for label in y])
            splitting_indices = (np.cumsum(self.amounts_per_partner)[:-1] * len(y)).astype(int)
            idxs = y.argsort()
            idx_list = np.split(idxs, splitting_indices)
            res = []
            for slice_idx in idx_list:
                res.append((x[slice_idx], y[slice_idx]))
            return res


class AdvancedSplitter(Splitter):
    def __init__(self, dataset, partners_list, amounts_per_partner, samples_split_description):
        self.num_clusters, self.specific_shared = list(zip(*samples_split_description))
        super().__init__(dataset, partners_list, amounts_per_partner)

    def _generate_subset(self, x, y):
        lb = LabelEncoder()
        y = lb.fit_transform([str(label) for label in y])
        nb_diff_labels = len(lb.classes_)

        for p_id, p in enumerate(self.partners_list):
            p.cluster_count_param = self.specific_shared[p_id]
            p.cluster_split_option = self.num_clusters[p_id]

        partners_with_specific_clusters = [p for p, option in zip(self.partners_list, self.specific_shared) if
                                           option == 'specific']
        partners_with_specific_clusters.sort(key=operator.attrgetter("cluster_count"), reverse=True)
        partners_with_shared_clusters = [p for p, option in zip(self.partners_list, self.specific_shared) if
                                         option == 'shared']
        partners_with_shared_clusters.sort(key=operator.attrgetter("cluster_count"), reverse=True)

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

        x_for_cluster, y_for_cluster, nb_samples_per_cluster = {}, {}, {}
        for label in lb.classes_:
            idx_in_full_set = np.where(y == label)
            x_for_cluster[label] = x[idx_in_full_set]
            y_for_cluster[label] = y[idx_in_full_set]
            nb_samples_per_cluster[label] = len(y_for_cluster[label])
