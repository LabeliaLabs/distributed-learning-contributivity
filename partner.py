# -*- coding: utf-8 -*-
"""
This enables to parameterize the partners that participate to the simulated federated learning scenario.
"""

import keras
import numpy as np
from random import shuffle

import utils
import constants


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

    def corrupt_labels(self):
        for label in self.y_train:
            idx_max = np.argmax(label)
            label[idx_max] = 0.0
            label[idx_max - 1] = 1.0

    def shuffle_labels(self):
        for label in self.y_train:
            label = shuffle(label)
