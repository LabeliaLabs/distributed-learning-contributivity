# -*- coding: utf-8 -*-
"""
This enables to parameterize the partners that participate to the simulated federated learning scenario.
"""

from random import sample

import numpy as np
from keras.utils import to_categorical

from . import constants


class Partner:
    def __init__(self, partner_id):

        self.id = partner_id

        self.batch_size = constants.DEFAULT_BATCH_SIZE

        self.cluster_count: int
        self.cluster_split_option: str
        self.clusters_list = []
        self.final_nb_samples: int
        self.final_nb_samples_p_cluster: int

        self.x_train = None
        self.x_val = None
        self.x_test = None

        self.y_train = None
        self.y_val = None
        self.y_test = None

        self.corruption_matrix = None

    class _Decorator:  # This class allows use to create private decorators for the partner class.
        @classmethod
        def categorical_needed(cls, func):
            def _decorator(self, *args,
                           **kwargs):  # It is not very clear to me why self variable is accessible, but it's working
                # Check if the labels are encoded into categorical. If not, convert it
                if self.y_train.ndim == 1:
                    self.y_train = to_categorical(self.y_train.reshape(-1, 1))
                    one_label = True
                else:
                    one_label = False
                # Call the function
                res = func(self, *args, **kwargs)
                # If needed, convert it back
                if one_label:
                    self.y_train = np.argmax(self.y_train, axis=1)
                return res

            return _decorator

    @property
    def num_labels(self):
        return self.y_train.shape[1]

    @_Decorator.categorical_needed
    def corrupt_labels(self, proportion_corrupted):
        if not 0 <= proportion_corrupted <= 1:
            raise ValueError(
                f"The proportion of labels to corrupted was {proportion_corrupted} but it must be between 0 and 1."
            )

        # Select the indices where the label will be off-set
        n = int(len(self.y_train) * proportion_corrupted)
        idx = sample(list(range(len(self.y_train))), n)

        # Offset the labels
        for i in idx:
            new_label = self.y_train[i]
            idx_max = np.argmax(new_label)
            new_label[idx_max] = 0.0
            new_label[idx_max - 1] = 1.0
            self.y_train[i] = new_label

    @_Decorator.categorical_needed
    def permute_labels(self, proportion_corrupted=1):
        if not 0 <= proportion_corrupted <= 1:
            raise ValueError(
                f"The proportion of labels to corrupted was {proportion_corrupted} but it must be between 0 and 1."
            )

        # Select the indices where the label will be permuted
        n = int(len(self.y_train) * proportion_corrupted)
        idx = sample(list(range(len(self.y_train))), n)
        # Generate the permutation matrix to use
        self.corruption_matrix = np.zeros((self.num_labels, self.num_labels))
        idx_permutation = np.random.permutation(self.num_labels)
        self.corruption_matrix[np.arange(self.num_labels), idx_permutation] = 1
        # Permute the labels
        self.y_train[idx] = np.dot(self.y_train[idx], self.corruption_matrix.T)

    @_Decorator.categorical_needed
    def random_labels(self, proportion_corrupted=1):
        if not 0 <= proportion_corrupted <= 1:
            raise ValueError(
                f"The proportion of labels to corrupted was {proportion_corrupted} but it must be between 0 and 1."
            )
        # Select the indices where the label will be off-set
        n = int(len(self.y_train) * proportion_corrupted)
        idx = sample(list(range(len(self.y_train))), n)
        # Generate the random matrix to use
        alpha = np.ones(self.num_labels)
        self.corruption_matrix = np.random.dirichlet(alpha, self.num_labels)
        # Randomize the labels
        for i in idx:  # TODO vectorize
            temp = np.zeros((self.num_labels,))
            temp[np.random.choice(self.num_labels, p=self.corruption_matrix[np.argmax(self.y_train[i])])] = 1
            self.y_train[i] = temp

    @_Decorator.categorical_needed
    def shuffle_labels(self, proportion_shuffled):
        if not 0 <= proportion_shuffled <= 1:
            raise ValueError(
                f"The proportion of labels to corrupted was {proportion_shuffled} but it must be between 0 and 1."
            )
        n = int(len(self.y_train) * proportion_shuffled)
        idx = np.random.choice(len(self.y_train), size=n, replace=False)
        for i in idx:
            np.random.shuffle(self.y_train[i])


class PartnerMpl:
    def __init__(self, partner_parent, mpl):
        """
        :type partner_parent: Partner
        :type mpl: MultiPartnerLearning
        """
        self.mpl = mpl
        self.id = partner_parent.id
        self.batch_size = partner_parent.batch_size
        self.minibatch_count = mpl.minibatch_count
        self.partner_parent = partner_parent
        self.model_weights = None

        self.minibatched_x_train = np.nan * np.zeros(self.minibatch_count)
        self.minibatched_y_train = np.nan * np.zeros(self.minibatch_count)

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
