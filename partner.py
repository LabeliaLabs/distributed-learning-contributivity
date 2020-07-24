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

    def get_x_train_len(self):  # not used
        return len(self.x_train)


    def corrupt_labels(self):
        if type(self.y_train) != np.ndarray:
            raise TypeError("partner.y_train should be a numpy.ndarray")
        if self.y_train.shape == (0,):
            raise ValueError("partner.y_train shouldn't be an empty array")
        if self.y_train.dtype != "float32":
            raise TypeError("partner.y_train.dtype should be float32")
        for label in self.y_train:
            idx_max = np.argmax(label)
            label[idx_max] = 0.0
            label[idx_max - 1] = 1.0

    def shuffle_labels(self):
        if type(self.y_train) != np.ndarray:
            raise TypeError("partner.y_train should be a numpy.ndarray")
        if self.y_train.shape == (0,):
            raise ValueError("partner.y_train shouldn't be an empty array")
        if self.y_train.dtype != "float32":
            raise TypeError("partner.y_train.dtype should be float32")
        for label in self.y_train:
            label = shuffle(label)
