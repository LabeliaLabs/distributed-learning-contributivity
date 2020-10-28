# -*- coding: utf-8 -*-
"""
This enables to parameterize the partners that participate to the simulated federated learning scenario.
"""

from random import sample

import numpy as np

from . import constants


class Partner:
    def __init__(self, partner_id):

        self.id = partner_id

        self.batch_size = constants.DEFAULT_BATCH_SIZE

        self.cluster_count = int
        self.cluster_split_option = str
        self.clusters_list = []
        self.final_nb_samples = int
        self.final_nb_samples_p_cluster = int

        self.x_train = None
        self.x_val = None
        self.x_test = None

        self.y_train = None
        self.y_val = None
        self.y_test = None

    def corrupt_labels(self, proportion_corrupted):
        if not 0 <= proportion_corrupted <= 1:
            raise ValueError(
                f"The proportion of labels to corrupted was {proportion_corrupted} but it must be between 0 and 1."
            )

        # Select the indices where the label will be off-set
        n = int(len(self.y_train) * proportion_corrupted)
        idx = sample(list(range(len(self.y_train))), n)

        # Off-set  the labels
        for i in idx:
            new_label = self.y_train[i]
            idx_max = np.argmax(new_label)
            new_label[idx_max] = 0.0
            new_label[idx_max - 1] = 1.0
            self.y_train[i] = new_label

    def shuffle_labels(self, proportion_shuffled):
        if not 0 <= proportion_shuffled <= 1:
            raise ValueError(
                f"The proportion of labels to corrupted was {proportion_shuffled} but it must be between 0 and 1."
            )
        n = int(len(self.y_train) * proportion_shuffled)
        idx = np.random.choice(len(self.y_train), size=n, replace=False)
        for i in idx:
            np.random.shuffle(self.y_train[i])
