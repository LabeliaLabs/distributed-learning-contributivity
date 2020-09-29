# -*- coding: utf-8 -*-
"""
This enables to parameterize the partners that participate to the simulated federated learning scenario.
"""

from random import shuffle, sample

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
        if proportion_corrupted>1 or proportion_corrupted<0:
            raise ValueError(
                "The proportion of labells to corrupted ["
                + str(proportion_corrupted)
                + "] must be between 0 and 1."
            )
        
        # Select the indices where the label will be off-set 
        n = int(len(self.y_train)*proportion_corrupted)
        print()
        idx = sample(list(range(len(self.y_train))), n)
        
        # Off-set  the labels
        for label in self.y_train[idx]:
            idx_max = np.argmax(label)
            label[idx_max] = 0.0
            label[idx_max - 1] = 1.0

    def shuffle_labels(self, proportion_shuffled):
        if proportion_shuffled>1 or proportion_shuffled<0:
            raise ValueError(
                "The proportion of labells to corrupted ["
                + str(proportion_shuffled)
                + "] must be between 0 and 1."
            )
            
        # Select the indices where the label will be shuffled
        n = int(len(self.y_train)*proportion_shuffled)
        idx = sample(list(range(len(self.y_train))), n)
        
        # Suffle the labels
        for label in self.y_train[idx]:
            # Suffle the label
            new_label = shuffle(label)
            
            # Force the label to be different
            while np.all(new_label == label) :
                new_label = shuffle(label)
                
            label = new_label 
