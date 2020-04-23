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
    def __init__(self, x_train, x_test, y_train, y_test, batch_size_single, batch_size_multi, partner_id):
        self.x_train = x_train
        self.x_val = []
        self.x_test = x_test

        self.y_train = y_train
        self.y_val = []
        self.y_test = y_test

        self.batch_size_single = batch_size_single
        self.batch_size_multi = batch_size_multi

        self.partner_id = partner_id

    def get_x_train_len(self):
        return len(self.x_train)

    def corrupt_labels(self):
        for label in self.y_train:
            idx_max = np.argmax(label)
            label[idx_max] = 0.0
            label[idx_max - 1] = 1.0

    def shuffle_labels(self):
        for label in self.y_train:
            label = shuffle(label)
