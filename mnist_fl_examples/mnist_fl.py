# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:33:15 2019

@author: RGOUSSAULT
inspired from: https://keras.io/examples/mnist_cnn/
"""

from __future__ import print_function
import keras


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

import sys

sys.path.append("..")
import utils
import constants
import my_scenario
import data_splitting

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


#%% Fetch data splitting scenario

partners_count = my_scenario.partners_count
partners_list = data_splitting.process_data_splitting_scenario()


#%% Pre-process data for ML training

for partner_index, partner in enumerate(partners_list):

    # Preprocess input (x) data
    partner.preprocess_data()

    # Crete validation dataset
    x_partner_train, x_partner_val, y_partner_train, y_partner_val = train_test_split(
        partner.x_train, partner.y_train, test_size=0.1, random_state=42
    )
    partner.x_train = x_partner_train
    partner.x_val = x_partner_val
    partner.y_train = y_partner_train
    partner.y_val = y_partner_val

    # Align variable names
    x_partner_test = partner.x_test
    y_partner_test = partner.y_test

    print(str(len(x_partner_train)) + " train data for partner " + str(partner_index))
    print(str(len(x_partner_val)) + " val data for partner " + str(partner_index))
    print(str(len(x_partner_test)) + " test data for partner " + str(partner_index))


#%% Federated training

model_list = [None] * partners_count
epochs = 2
score_matrix = np.zeros(shape=(epochs, partners_count))
val_acc_epoch = []
acc_epoch = []

for epoch in range(epochs):

    print("\n=============================================")
    print("Epoch " + str(epoch))
    is_first_epoch = epoch == 0

    # Aggregation phase
    if is_first_epoch:
        # First epoch
        print("First epoch, generate model from scratch")

    else:
        print("Aggregating models weights to build a new model")
        # Aggregating phase : averaging the weights
        weights = [model.get_weights() for model in model_list]
        new_weights = list()

        # TODO : make this clearer
        for weights_list_tuple in zip(*weights):
            new_weights.append(
                [
                    np.array(weights_).mean(axis=0)
                    for weights_ in zip(*weights_list_tuple)
                ]
            )

    # Training phase
    val_acc_list = []
    acc_list = []
    for partner_index, partner in enumerate(partners_list):

        print("\nTraining on partner " + str(partner_index))
        partner_model = utils.generate_new_cnn_model()

        # Model weights are the averaged weights
        if not is_first_epoch:
            partner_model.set_weights(new_weights)
            partner_model.compile(
                loss=keras.losses.categorical_crossentropy,
                optimizer="adam",
                metrics=["accuracy"],
            )

        # Train on whole partner local data set
        history = partner_model.fit(
            partner.x_train,
            partner.y_train,
            batch_size=constants.BATCH_SIZE,
            epochs=1,
            verbose=1,
            validation_data=(partner.x_val, partner.y_val),
        )

        val_acc_list.append(history.history["val_accuracy"])
        acc_list.append(history.history["accuracy"])

        model_list[partner_index] = partner_model

    val_acc_epoch.append(np.median(val_acc_list))
    acc_epoch.append(np.median(acc_list))

    # TODO Evaluation phase: evaluate data on every partner test set

# TODO Compute contributivity score


#%% Plot history

plt.figure()
plt.plot(acc_epoch, "+-")
plt.plot(val_acc_epoch, "+-")
plt.title("Accuracy")
plt.ylabel("Acc")
plt.xlabel("Epoch")
plt.legend(("Train", "Val"))
plt.show()
