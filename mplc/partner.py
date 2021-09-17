# -*- coding: utf-8 -*-
"""
This enables to parameterize the partners that participate to the simulated federated learning scenario.
"""

import numpy as np
from loguru import logger
from tensorflow.keras.utils import to_categorical
import copy

from . import constants
from .corruption import NoCorruption


class Partner:

    def __init__(self, partner_id, **kwargs):

        #  Corruption related attributes
        self.corruption = kwargs.get('corruption', NoCorruption())
        self.corruption.set_partner(self)
        self.y_train_true = []  # used only if y_train is corrupted

        self.id = partner_id

        self.batch_size = constants.DEFAULT_BATCH_SIZE

        self.cluster_count: int
        self.cluster_split_option: str
        self.clusters_list = []
        self.final_nb_samples_p_cluster: int

        self.x_train = []
        self.x_val = []
        self.x_test = []

        self.y_train = []
        self.y_val = []
        self.y_test = []

    @property
    def num_labels(self):
        return self.y_train.shape[1]

    @property
    def data_volume(self):
        return len(self.y_train)

    @property
    def final_nb_samples(self):
        return len(self.y_train)

    @property
    def labels(self):
        if self.y_train.ndim == 1:
            return np.unique(self.y_train)
        else:
            return np.unique(np.argmax(self.y_train, axis=1))

    def corrupt(self):
        self.y_train_true = copy.deepcopy(self.y_train)

        # Check if the labels are encoded into categorical. If not, convert them
        if self.y_train.ndim == 1:
            self.y_train = to_categorical(self.y_train.reshape(-1, 1))
            one_label = True
        else:
            one_label = False

        self.corruption.apply()

        if one_label:
            self.y_train = np.argmax(self.y_train, axis=1).astype('float32')
        logger.debug(f"   Partner #{self.id}: Done.")


class PartnerMpl:
    def __init__(self, partner_parent, mpl):
        """
        :type partner_parent: Partner
        :type mpl: MultiPartnerLearning
        """
        self.grads = None
        self.mpl = mpl
        self.id = partner_parent.id
        self.batch_size = partner_parent.batch_size
        self.minibatch_count = mpl.minibatch_count
        self.partner_parent = partner_parent
        self.model_weights = None

        self.minibatched_x_train = np.nan * np.zeros(self.minibatch_count)
        self.minibatched_y_train = np.nan * np.zeros(self.minibatch_count)

    @property
    def y_train(self):
        return self.partner_parent.y_train

    @property
    def y_test(self):
        return self.partner_parent.y_test

    @property
    def y_val(self):
        return self.partner_parent.y_val

    @property
    def x_train(self):
        return self.partner_parent.x_train

    @property
    def x_test(self):
        return self.partner_parent.x_test

    @property
    def x_val(self):
        return self.partner_parent.x_val

    @property
    def data_volume(self):
        return len(self.partner_parent.y_train)

    @property
    def last_round_score(self):
        return self.mpl.history.history[self.id]['val_accuracy'][self.mpl.epoch_index, self.mpl.minibatch_index]

    @property
    def history(self):
        return self.mpl.history.history[self.id]

    def split_minibatches(self):
        """Split the dataset of the partner parent in mini-batches"""

        # Create the indices where to split
        split_indices = np.arange(1, self.minibatch_count + 1) / self.minibatch_count

        # Shuffle the dataset
        idx = np.random.permutation(len(self.partner_parent.x_train))
        x_train, y_train = self.partner_parent.x_train[idx], self.partner_parent.y_train[idx]

        # Split the samples and labels
        self.minibatched_x_train = np.split(x_train, (split_indices[:-1] * len(x_train)).astype(int))
        self.minibatched_y_train = np.split(y_train, (split_indices[:-1] * len(y_train)).astype(int))

    def build_model(self):
        return self.mpl.build_model_from_weights(self.model_weights)
