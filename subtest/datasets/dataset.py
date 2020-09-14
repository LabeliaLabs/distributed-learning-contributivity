# -*- coding: utf-8 -*-
"""
The dataset object used in the multi-partner learning and contributivity measurement experiments.
"""
import numpy as np
from sklearn.model_selection import train_test_split


class Dataset:

    def __init__(self,
                 dataset_name,
                 x_train,
                 x_test,
                 y_train,
                 y_test,
                 input_shape,
                 num_classes,
                 preprocess_dataset_labels,
                 generate_new_model_for_dataset,
                 train_val_split_global=None,
                 train_test_split_local=None,
                 train_val_split_local=None
                 ):
        self.name = dataset_name

        self.input_shape = input_shape
        self.num_classes = num_classes

        self.x_train = x_train
        self.x_val = None
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = None
        self.y_test = y_test

        self.preprocess_dataset_labels = preprocess_dataset_labels
        self.generate_new_model_for_dataset = generate_new_model_for_dataset

        if train_val_split_local:
            self.train_val_split_local = train_val_split_local
        else:
            self.train_test_split_local = self.__default_train_split_local

        if train_test_split_local:
            self.train_test_split_local = train_test_split_local
        else:
            self.train_test_split_local = self.__default_train_split_local

        self.train_val_split_global = train_val_split_global
        self.train_val_split()

    def train_val_split(self):
        """Called once, at the end of Dataset's constructor"""
        if self.x_val or self.y_val:
            raise Exception("x_val and y_val should be of NoneType")
        if self.train_val_split_global:
            self.x_train, self.x_val, self.y_train, self.y_val = self.train_val_split_global(self.x_train, self.y_train)
        else:
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train,
                                                                                  self.y_train,
                                                                                  test_size=0.1,
                                                                                  random_state=42)

    @staticmethod
    def __default_train_split_local(x, y):
        return x, np.array([]), y, np.array([])

    def generate_new_model(self):
        return self.generate_new_model_for_dataset()
