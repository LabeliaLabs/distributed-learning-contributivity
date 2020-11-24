# -*- coding: utf-8 -*-
"""
This enables to parameterize the partners that participate to the simulated federated learning scenario.
"""

import numpy as np
from keras.utils import to_categorical
from loguru import logger

from . import constants


class Partner:
    IMPLEMENTED_CORRUPTION = ['not_corrupted', 'duplication', 'permutation', 'random', 'random-uniform',
                              'permutation-circular',
                              'redundancy']

    def __init__(self, partner_id, **kwargs):

        #  Corruption related attributes
        self.corruption_method = kwargs.get('corruption_method', 'not_corrupted')
        self.proportion_corrupted = kwargs.get('proportion_corrupted', 1.)
        if not 0 <= self.proportion_corrupted <= 1:
            raise ValueError(f"The proportion of labels to corrupted was {self.proportion_corrupted} "
                             f"but it must be between 0 and 1.")
        if self.corruption_method not in self.IMPLEMENTED_CORRUPTION:
            raise ValueError(f'Unrecognized corruption method {self.corruption_method}')
        self.duplicated_partner_id = kwargs.get('duplicated_partner', None)
        self.duplicated_partner = None

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

    class _Decorator:  # This class allows creation of private decorators for the partner class.
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

    @property
    def data_volume(self):
        return len(self.y_train)

    def corrupt(self):
        if self.corruption_method == 'not_corrupted':
            logger.debug(f"   Partner #{self.id}: not corrupted.")
        else:
            logger.debug(
                f"... Corrupting ({self.corruption_method}) "
                f" {self.proportion_corrupted * 100} "
                f"percent of the data of partner #{self.id}")
            if self.corruption_method == 'duplication':
                if not self.duplicated_partner:
                    raise AttributeError(f'The partner for duplication is missing, please set the '
                                         f'.duplicated_partner attribute to the partner '
                                         f'with id {self.duplicated_partner_id}')
                self.duplicate_data()
            elif self.corruption_method == 'permutation':
                self.permute_labels()
            elif self.corruption_method == 'permutation-circular':
                self.permute_labels_circular()
            elif self.corruption_method == 'random':
                self.random_labels()
            elif self.corruption_method == 'random-uniform':
                self.random_labels_uniform()
            elif self.corruption_method == 'redundancy':
                self.redundant_data()
            logger.debug(f"   Partner #{self.id}: Done.")

    @_Decorator.categorical_needed
    def permute_labels_circular(self):
        """
        Offset the labels of the partner's dataset. It is equivalent to perform a circular permutation.

        """
        # Select the indices where the label will be off-set
        n = int(len(self.y_train) * self.proportion_corrupted)
        idx = np.random.choice(len(self.y_train), size=n, replace=False)

        # Define the corruption matrix
        self.corruption_matrix = np.roll(np.identity(self.num_labels), -1, axis=1)
        # offset the labels
        self.y_train[idx] = np.dot(self.y_train[idx], self.corruption_matrix.T)

    @_Decorator.categorical_needed
    def permute_labels(self):
        # Select the indices where the label will be permuted
        n = int(len(self.y_train) * self.proportion_corrupted)
        idx = np.random.choice(len(self.y_train), size=n, replace=False)
        # Generate the permutation matrix to use
        self.corruption_matrix = np.zeros((self.num_labels, self.num_labels))
        idx_permutation = np.random.permutation(self.num_labels)
        self.corruption_matrix[np.arange(self.num_labels), idx_permutation] = 1
        # Permute the labels
        self.y_train[idx] = np.dot(self.y_train[idx], self.corruption_matrix.T)

    @_Decorator.categorical_needed
    def random_labels(self):
        """
        Draw new labels from a dirichlet distribution. The corruption matrix is a stochastic one, where
        the i, j number stands for the probability of the labels to be set to i, knowing that the true label is j.

        """

        # Select the indices where the label will be off-set
        n = int(len(self.y_train) * self.proportion_corrupted)
        idx = np.random.choice(len(self.y_train), size=n, replace=False)
        # Generate the random matrix to use
        alpha = np.ones(self.num_labels)
        self.corruption_matrix = np.random.dirichlet(alpha, self.num_labels)
        # Randomize the labels
        for i in idx:
            temp = np.zeros((self.num_labels,))
            temp[np.random.choice(self.num_labels, p=self.corruption_matrix[np.argmax(self.y_train[i])])] = 1
            self.y_train[i] = temp

    @_Decorator.categorical_needed
    def random_labels_uniform(self):
        """
        Shuffle the labels of the partner dataset, with uniform distribution.

        :return:
        """
        if not 0 <= self.proportion_corrupted <= 1:
            raise ValueError(
                f"The proportion of labels to corrupted was {self.proportion_corrupted} but it must be between 0 and 1."
            )
        n = int(len(self.y_train) * self.proportion_corrupted)
        idx = np.random.choice(len(self.y_train), size=n, replace=False)
        for i in idx:
            np.random.shuffle(self.y_train[i])

    def duplicate_data(self):
        """
        The partner's dataset is replaced by another partner's dataset
        :return:
        """
        # Select the indices where the label will be off-set
        n = int(len(self.y_train) * self.proportion_corrupted)

        if len(self.duplicated_partner.y_train) > n:
            idx = np.random.choice(len(self.y_train), size=n, replace=False)
        else:
            n = int(len(self.duplicated_partner.y_train))
            idx = np.random.choice(len(self.y_train), size=n, replace=False)
            logger.warning(f"The partner which dataset would have been copied does not have enough data"
                           f" to corrupt {self.proportion_corrupted * 100}% of this partner's dataset."
                           f" Only {np.round((n / len(self.y_train)), 2) * 100} percent will be corrupted")
            self.proportion_corrupted = np.round((n / len(self.y_train)), 2)
        self.y_train[idx] = self.duplicated_partner.y_train[:len(idx)]
        self.x_train[idx] = self.duplicated_partner.x_train[:len(idx)]

    def redundant_data(self):
        """
        The partner's data are replaced by a copy of one of its sample
        """
        # Select the indices where the label will be off-set
        n = int(len(self.y_train) * self.proportion_corrupted)
        idx = np.random.choice(len(self.y_train), size=n, replace=False)
        self.y_train[idx] = np.tile(self.y_train[idx[0]], (n,) + (1,) * self.y_train[idx[0]].ndim)
        self.x_train[idx] = np.tile(self.x_train[idx[0]], (n,) + (1,) * self.x_train[idx[0]].ndim)


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
