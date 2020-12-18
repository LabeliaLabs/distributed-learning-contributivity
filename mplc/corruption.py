# -*- coding: utf-8 -*-
"""
This enables to corrupt partner's datasets.
"""

from abc import ABC, abstractmethod

import numpy as np
from loguru import logger
from tensorflow.keras.utils import to_categorical


class Corruption(ABC):
    def __init__(self, proportion=1, partner=None):
        if not 0 <= proportion <= 1:
            raise ValueError(f"The proportion of labels to corrupted was {proportion} "
                             f"but it must be between 0 and 1.")
        else:
            self.proportion = proportion

        self.partner = None
        self.matrix = None

        self._corrupted_train_idx = None
        self._corrupted_val_idx = None
        self._corrupted_test_idx = None

        if partner:
            self.set_partner(partner)

    def set_partner(self, partner):
        self.partner = partner

    @property
    def corrupted_train_index(self):
        if self._corrupted_train_idx is None:
            n = int(len(self.partner.y_train) * self.proportion)
            self._corrupted_train_idx = np.random.choice(len(self.partner.y_train), size=n, replace=False)
        return self._corrupted_train_idx

    @property
    def corrupted_val_index(self):
        if self._corrupted_val_idx is None:
            n = int(len(self.partner.y_val) * self.proportion)
            self._corrupted_val_idx = np.random.choice(len(self.partner.y_val), size=n, replace=False)
        return self._corrupted_val_idx

    @property
    def corrupted_test_index(self):
        if self._corrupted_test_idx is None:
            n = int(len(self.partner.y_test) * self.proportion)
            self._corrupted_test_idx = np.random.choice(len(self.partner.y_test), size=n, replace=False)
        return self._corrupted_test_idx

    @abstractmethod
    def apply(self):
        self.generate_matrix()

    def generate_matrix(self):
        self.matrix = np.identity(self.partner.num_labels)

    def error_on_corruption_matrix(self, matrix):
        return np.linalg.norm(self.matrix - matrix) / np.linalg.norm(self.matrix)


class NoCorruption(Corruption):
    name = 'not-corrupted'

    def apply(self):
        self.generate_matrix()
        logger.debug(f"   Partner #{self.partner.id}: not corrupted.")


class Permutation(Corruption):
    name = 'permutation'

    def generate_matrix(self):
        self.matrix = np.zeros((self.partner.num_labels, self.partner.num_labels))
        idx_permutation = np.random.permutation(self.partner.num_labels)
        self.matrix[np.arange(self.partner.num_labels), idx_permutation] = 1

    def apply(self):
        # Generate the permutation matrix to use
        self.generate_matrix()
        # Permute the labels
        self.partner.y_train[self.corrupted_train_index] = np.dot(self.partner.y_train[self.corrupted_train_index],
                                                                  self.matrix)
        if len(self.corrupted_val_index) > 0:
            self.partner.y_val[self.corrupted_val_index] = np.dot(self.partner.y_val[self.corrupted_val_index],
                                                                  self.matrix)
        if len(self.corrupted_test_index) > 0:
            self.partner.y_test[self.corrupted_test_index] = np.dot(self.partner.y_test[self.corrupted_test_index],
                                                                    self.matrix)

        logger.debug(f"   Partner #{self.partner.id}: Done.")


class PermutationCircular(Permutation):
    name = 'permutation-circular'

    def generate_matrix(self):
        self.matrix = np.roll(np.identity(self.partner.num_labels), -1, axis=1)


class Randomize(Corruption):
    name = 'random'

    def generate_matrix(self):
        alpha = np.ones(self.partner.num_labels)
        self.matrix = np.random.dirichlet(alpha, self.partner.num_labels)

    def apply(self):
        # Generate the permutation matrix to use
        self.generate_matrix()
        # Randomize the labels
        idx_train = self.corrupted_train_index
        for i in idx_train:
            temp = np.zeros((self.partner.num_labels,))
            temp[np.random.choice(self.partner.num_labels, p=self.matrix[np.argmax(self.partner.y_train[i])])] = 1
            self.partner.y_train[i] = temp
        logger.debug(f"   Partner #{self.partner.id}: Done.")
        idx_val = self.corrupted_val_index
        for i in idx_val:
            temp = np.zeros((self.partner.num_labels,))
            temp[np.random.choice(self.partner.num_labels, p=self.matrix[np.argmax(self.partner.y_val[i])])] = 1
            self.partner.y_val[i] = temp
        logger.debug(f"   Partner #{self.partner.id}: Done.")
        idx_test = self.corrupted_test_index
        for i in idx_test:
            temp = np.zeros((self.partner.num_labels,))
            temp[np.random.choice(self.partner.num_labels, p=self.matrix[np.argmax(self.partner.y_test[i])])] = 1
            self.partner.y_test[i] = temp
        logger.debug(f"   Partner #{self.partner.id}: Done.")


class RandomizeUniform(Randomize):
    name = 'random-uniform'

    def generate_matrix(self):
        self.matrix = np.ones((self.partner.num_labels,) * 2) / self.partner.num_labels

    def apply(self):
        # Generate the permutation matrix to use
        self.generate_matrix()
        # Randomize the labels
        idx_train = self.corrupted_train_index
        idx_val = self.corrupted_val_index
        idx_test = self.corrupted_test_index
        for i in idx_train:
            np.random.shuffle(self.partner.y_train[i])
        for i in idx_val:
            np.random.shuffle(self.partner.y_val[i])
        for i in idx_test:
            np.random.shuffle(self.partner.y_test[i])

        logger.debug(f"   Partner #{self.partner.id}: Done.")


class Redundancy(Corruption):
    name = 'redundancy'

    def apply(self):
        idx_train = self.corrupted_train_index
        self.generate_matrix()
        self.partner.y_train[idx_train] = np.tile(self.partner.y_train[idx_train[0]],
                                                  (len(idx_train),) + (1,) * self.partner.y_train[idx_train[0]].ndim)
        self.partner.x_train[idx_train] = np.tile(self.partner.x_train[idx_train[0]],
                                                  (len(idx_train),) + (1,) * self.partner.x_train[idx_train[0]].ndim)
        idx_val = self.corrupted_val_index
        if len(idx_val) > 0:
            self.partner.y_val[idx_val] = np.tile(self.partner.y_val[idx_val[0]],
                                                  (len(idx_val),) + (1,) * self.partner.y_val[idx_val[0]].ndim)
            self.partner.x_val[idx_val] = np.tile(self.partner.x_val[idx_val[0]],
                                                  (len(idx_val),) + (1,) * self.partner.x_val[idx_val[0]].ndim)
        idx_test = self.corrupted_test_index
        if len(idx_test) > 0:
            self.partner.y_test[idx_test] = np.tile(self.partner.y_test[idx_test[0]],
                                                    (len(idx_test),) + (1,) * self.partner.y_test[idx_test[0]].ndim)
            self.partner.x_test[idx_test] = np.tile(self.partner.x_test[idx_test[0]],
                                                    (len(idx_test),) + (1,) * self.partner.x_test[idx_test[0]].ndim)
        logger.debug(f"   Partner #{self.partner.id}: Done.")


class Duplication(Corruption):
    name = 'duplication'

    def __init__(self, proportion=1, partner=None, duplicated_partner_id=None, duplicated_partner=None):
        super(Duplication, self).__init__(proportion=proportion, partner=partner)
        self.duplicated_partner = duplicated_partner
        self.duplicated_partner_id = duplicated_partner_id
        if duplicated_partner_id is None and not duplicated_partner:
            raise Exception('Please provide either a Partner to duplicate, or its id')

    @property
    def corrupted_train_index(self):
        if self._corrupted_train_idx is None:
            n = int(len(self.partner.y_train) * self.proportion)
            if len(self.duplicated_partner.y_train) > n:
                self._corrupted_train_idx = np.random.choice(len(self.partner.y_train), size=n, replace=False)
            else:
                n = int(len(self.duplicated_partner.y_train))
                self._corrupted_train_idx = np.random.choice(len(self.partner.y_train), size=n, replace=False)
                logger.warning(f"The partner which dataset would have been copied does not have enough data"
                               f" to corrupt {self.proportion * 100}% of this partner's dataset."
                               f" Only {np.round((n / len(self.partner.y_train)), 2) * 100} percent will be corrupted")
                self.proportion = np.round((n / len(self.partner.y_train)), 2)
        return self._corrupted_train_idx

    @property
    def corrupted_val_index(self):
        if self._corrupted_val_idx is None:
            n = int(len(self.partner.y_val) * self.proportion)
            if len(self.duplicated_partner.y_val) > n:
                self._corrupted_val_idx = np.random.choice(len(self.partner.y_val), size=n, replace=False)
            elif n == 0:
                self._corrupted_val_idx = []
            else:
                n = int(len(self.duplicated_partner.y_val))
                self._corrupted_val_idx = np.random.choice(len(self.partner.y_val), size=n, replace=False)
                logger.warning(f"The partner which dataset would have been copied does not have enough data"
                               f" to corrupt {self.proportion * 100}% of this partner's dataset."
                               f" Only {np.round((n / len(self.partner.y_val)), 2) * 100} percent will be corrupted")
                self.proportion = np.round((n / len(self.partner.y_val)), 2)
        return self._corrupted_val_idx

    @property
    def corrupted_test_index(self):
        if self._corrupted_test_idx is None:
            n = int(len(self.partner.y_test) * self.proportion)
            if len(self.duplicated_partner.y_test) > n:
                self._corrupted_test_idx = np.random.choice(len(self.partner.y_test), size=n, replace=False)
            elif n == 0:
                self._corrupted_test_idx = []
            else:
                n = int(len(self.duplicated_partner.y_test))
                self._corrupted_test_idx = np.random.choice(len(self.partner.y_test), size=n, replace=False)
                logger.warning(f"The partner which dataset would have been copied does not have enough data"
                               f" to corrupt {self.proportion * 100}% of this partner's dataset."
                               f" Only {np.round((n / len(self.partner.y_test)), 2) * 100} percent will be corrupted")
                self.proportion = np.round((n / len(self.partner.y_test)), 2)
        return self._corrupted_test_idx

    def set_duplicated_partner(self, partner_list):
        if self.duplicated_partner_id is None:  # We cannot use the `if not self.dupl... ` because if not 0 return True
            raise Exception('Missing the Partner-to-duplicate\'s id.')
        self.duplicated_partner = [partner for partner in partner_list if partner.id == self.duplicated_partner_id][0]

    def apply(self):
        if not self.duplicated_partner:
            raise Exception('Missing the Partner to duplicate')
        self.generate_matrix()
        idx = self.corrupted_train_index
        if self.duplicated_partner.y_train.ndim == 1:
            self.duplicated_partner.y_train = to_categorical(self.duplicated_partner.y_train.reshape(-1, 1))
            one_label = True
        else:
            one_label = False
        self.partner.y_train[idx] = self.duplicated_partner.y_train[:len(idx)]
        self.partner.x_train[idx] = self.duplicated_partner.x_train[:len(idx)]
        if one_label:
            self.duplicated_partner.y_train = np.argmax(self.duplicated_partner.y_train, axis=1).astype('float32')

        if len(self.corrupted_val_index) > 0:
            self.partner.y_val[self.corrupted_val_index] = self.duplicated_partner.y_val[
                                                           :len(self.corrupted_val_index)]
            self.partner.x_val[self.corrupted_val_index] = self.duplicated_partner.x_val[
                                                           :len(self.corrupted_val_index)]
        if len(self.corrupted_test_index) > 0:
            self.partner.y_test[self.corrupted_test_index] = self.duplicated_partner.y_test[
                                                             :len(self.corrupted_test_index)]
            self.partner.x_test[self.corrupted_test_index] = self.duplicated_partner.x_test[
                                                             :len(self.corrupted_test_index)]
        logger.debug(f"   Partner #{self.partner.id}: Done.")


IMPLEMENTED_CORRUPTION = {'not-corrupted': NoCorruption,
                          'duplication': Duplication,
                          'permutation': Permutation,
                          'random': Randomize,
                          'random-uniform': RandomizeUniform,
                          'permutation-circular': PermutationCircular,
                          'redundancy': Redundancy}
