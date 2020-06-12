# -*- coding: utf-8 -*-
"""
The dataset object used in the multi-partner learning and contributivity measurement experiments.
"""

from sklearn.model_selection import train_test_split


class DatasetAndModel:

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

    def train_val_split(self):

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train, self.y_train, test_size=0.2, random_state=42
        )

    def preprocess_labels(self):

        self.y_train = self.preprocess_dataset_labels(self.y_train)
        self.y_val = self.preprocess_dataset_labels(self.y_val)
        self.y_test = self.preprocess_dataset_labels(self.y_test)

    def generate_new_model(self):

        return self.generate_new_model_for_dataset()
