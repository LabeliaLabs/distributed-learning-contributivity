# -*- coding: utf-8 -*-
"""
Functions for model training and evaluation (single-partner and multi-partner cases)
(inspired from: https://keras.io/examples/mnist_cnn/)
"""

from __future__ import print_function

import keras
from keras.backend.tensorflow_backend import clear_session
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

import utils
import constants

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


#%% Pre-process data for ML training


def preprocess_scenarios_data(scenario):
    """Return scenario with central datasets (val, test) and distributed datasets (partners) pre-processed"""

    print("\n## Pre-processing datasets of the scenario for keras CNN:")

    # First, datasets of each partner
    for partner_index, partner in enumerate(scenario.partners_list):

        # Preprocess inputs (x) data
        partner.x_train = utils.preprocess_input(partner.x_train)
        partner.x_test = utils.preprocess_input(partner.x_test)

        # Preprocess labels (y) data
        partner.y_train = keras.utils.to_categorical(partner.y_train, constants.NUM_CLASSES)
        partner.y_test = keras.utils.to_categorical(partner.y_test, constants.NUM_CLASSES)

        # Create validation dataset
        partner.x_train, partner.x_val, partner.y_train, partner.y_val = train_test_split(
            partner.x_train, partner.y_train, test_size=0.1, random_state=42
        )

        if scenario.corrupted_partners[partner_index] == "corrupted":
            print("   ... Corrupting data (offsetting labels) of partner " + str(partner_index))
            partner.corrupt_labels()
        elif scenario.corrupted_partners[partner_index] == "shuffled":
            print("   ... Corrupting data (shuffling labels) of partner " + str(partner_index))
            partner.shuffle_labels()
        elif scenario.corrupted_partners[partner_index] == "not_corrupted":
            pass
        else:
            print("Unexpected label of corruption, not corruption performed!")

        print("   Partner #" + str(partner_index) + ": done.")

    # Then the scenario central dataset of the scenario
    scenario.x_val = utils.preprocess_input(scenario.x_val)
    scenario.y_val = keras.utils.to_categorical(scenario.y_val, constants.NUM_CLASSES)
    print("   Central early stopping validation set: done.")
    scenario.x_test = utils.preprocess_input(scenario.x_test)
    scenario.y_test = keras.utils.to_categorical(scenario.y_test, constants.NUM_CLASSES)
    print("   Central testset: done.")

    return scenario


#%% Single partner training


def compute_test_score_for_single_partner(
    partner, epoch_count, single_partner_test_mode, global_x_test, global_y_test
):
    """Return the score on test data of a model trained on a single partner"""

    print("\n## Training and evaluating model on one single partner.")

    # Initialize model
    model = utils.generate_new_cnn_model()

    # Train model
    print("\n### Training model on one single partner: " + str(partner.partner_id))
    history = model.fit(
        partner.x_train,
        partner.y_train,
        batch_size=constants.BATCH_SIZE,
        epochs=epoch_count,
        verbose=0,
        validation_data=(partner.x_val, partner.y_val),
    )

    # Reference a testset according to the scenario configuration
    if single_partner_test_mode == "global":
        x_test = global_x_test
        y_test = global_y_test
    elif single_partner_test_mode == "local":
        x_test = partner.x_test
        y_test = partner.y_test
    else:
        raise NameError(
            "This single_partner_test_mode option ["
            + single_partner_test_mode
            + "] is not recognized."
        )

    # Evaluate trained model
    print("\n### Evaluating model on test data of the partner:")
    model_evaluation = model.evaluate(
        x_test, y_test, batch_size=constants.BATCH_SIZE, verbose=0
    )
    print("   Model metrics names: ", model.metrics_names)
    print("   Model metrics values: ", ["%.3f" % elem for elem in model_evaluation])

    model_eval_score = model_evaluation[1]  # 0 is for the loss

    # Return model score on test data
    print("\nTraining and evaluation on one single partner: done.")
    return model_eval_score


#%% TODO no methods overloading


def compute_test_score_with_scenario(scenario, is_save_fig=False):
    return compute_test_score(
        scenario.partners_list,
        scenario.epoch_count,
        scenario.x_val,
        scenario.y_val,
        scenario.x_test,
        scenario.y_test,
        scenario.aggregation_weighting,
        scenario.minibatch_count,
        scenario.is_early_stopping,
        scenario.single_partner_test_mode,
        is_save_fig,
        save_folder=scenario.save_folder,
    )


def split_in_minibatches(minibatch_count, x_train, y_train):
    """Returns the list of mini-batches into which the dataset has been split"""

    # Shuffle the dataset
    idx = np.random.permutation(len(x_train))
    x_train, y_train = x_train[idx], y_train[idx]

    # Create the indices where to split
    split_indices = np.arange(1, minibatch_count + 1) / minibatch_count

    # Split the samples and labels
    minibatched_x_train = np.split(x_train, (split_indices[:-1] * len(x_train)).astype(int))
    minibatched_y_train = np.split(y_train, (split_indices[:-1] * len(y_train)).astype(int))

    return minibatched_x_train, minibatched_y_train


def prepare_aggregation_weights(
    aggregation_weighting, partners_count, partners_list, input_weights
):
    """Returns a list of weights for the weighted average aggregation of model weights"""

    aggregation_weights = []
    if aggregation_weighting == "uniform":
        aggregation_weights = [1 / partners_count] * partners_count
    elif aggregation_weighting == "data_volume":
        partners_sizes = [len(partner.x_train) for partner in partners_list]
        aggregation_weights = partners_sizes / np.sum(partners_sizes)
    elif aggregation_weighting == "local_score":
        aggregation_weights = input_weights / np.sum(input_weights)
    else:
        raise NameError(
            "This aggregation_weighting scenario ["
            + aggregation_weighting
            + "] is not recognized."
        )

    return aggregation_weights


def aggregate_model_weights(model_list, aggregation_weights):
    """Aggregate model weights from a list of models, with a weighted average"""

    weights_per_model = [model.get_weights() for model in model_list]
    weights_per_layer = list(zip(*weights_per_model))
    new_weights = list()

    for weights_for_layer in weights_per_layer:
        avg_weights_for_layer = np.average(
            np.array(weights_for_layer), axis=0, weights=aggregation_weights
        )
        new_weights.append(list(avg_weights_for_layer))

    return new_weights


def build_aggregated_model(new_weights):
    """Generate a new model initialized with weights passed as arguments"""

    aggregated_model = utils.generate_new_cnn_model()
    aggregated_model.set_weights(new_weights)
    aggregated_model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer="adam",
        metrics=["accuracy"],
    )

    return aggregated_model


def compute_test_score(
    partners_list,
    epoch_count,
    x_val_global,
    y_val_global,
    x_test,
    y_test,
    aggregation_weighting,
    minibatch_count,
    is_early_stopping=True,
    single_partner_test_mode="global",
    is_save_fig=False,
    save_folder="",
):
    """Return the score on test data of a final aggregated model trained in a federated way on each partner"""

    # First, if only one partner, fall back to dedicated single partner function
    partners_count = len(partners_list)
    if partners_count == 1:
        return compute_test_score_for_single_partner(
            partners_list[0], epoch_count, single_partner_test_mode, x_test, y_test
        )

    # Else, continue onto a federated learning procedure
    print("\n## Training and evaluating model on multiple partners: " + str(partners_list))

    # Initialize variables
    model_list, local_score_list = [None] * partners_count, [None] * partners_count
    score_matrix = np.zeros(shape=(epoch_count, partners_count))
    score_matrix_extended = np.zeros(shape=(epoch_count, minibatch_count, partners_count))
    global_val_acc, global_val_loss = [], []

    # Train model (iterate for each epoch and mini-batch)
    print("\n### Training model:")
    for epoch_index in range(epoch_count):

        print("\n   Epoch " + str(epoch_index) + " out of " + str(epoch_count-1) + " total epochs")
        is_first_epoch = epoch_index == 0
        clear_session()

        # Split the train dataset in mini-batches
        minibatched_x_train, minibatched_y_train = [None] * partners_count, [None] * partners_count
        for partner_index, partner in enumerate(partners_list):
            (
                minibatched_x_train[partner_index],
                minibatched_y_train[partner_index],
            ) = split_in_minibatches(minibatch_count, partner.x_train, partner.y_train)

        # Iterate over mini-batches for training, starting each new iteration with an aggregation of the previous one
        for minibatch_index in range(minibatch_count):

            print(
                "\n      Mini-batch "
                + str(minibatch_index)
                + " out of "
                + str(minibatch_count - 1)
                + " total mini-batches"
            )
            is_first_minibatch = minibatch_index == 0

            # Starting model for each partner is the aggregated model from the previous mini-batch iteration
            agg_model_for_iteration = [None] * partners_count
            if not is_first_epoch or not is_first_minibatch:
                aggregation_weights = prepare_aggregation_weights(
                    aggregation_weighting, partners_count, partners_list, local_score_list
                )
            for partner_index, partner in enumerate(partners_list):
                if is_first_epoch and is_first_minibatch:
                    agg_model_for_iteration[partner_index] = utils.generate_new_cnn_model()
                else:
                    agg_model_for_iteration[partner_index] = build_aggregated_model(
                        aggregate_model_weights(model_list, aggregation_weights)
                    )

            # Iterate over partners for training each individual model
            for partner_index, partner in enumerate(partners_list):

                partner_model = agg_model_for_iteration[partner_index]

                # Train on partner local data set
                print(
                    "         Training on partner "
                    + str(partner_index)
                    + " out of "
                    + str(partners_count)
                    + " total partners"
                )
                history = partner_model.fit(
                    minibatched_x_train[partner_index][minibatch_index],
                    minibatched_y_train[partner_index][minibatch_index],
                    batch_size=constants.BATCH_SIZE,
                    epochs=1,
                    verbose=0,
                    validation_data=(x_val_global, y_val_global),
                )
                print(
                    "            val_accuracy: "
                    + str(history.history["val_accuracy"][0])
                )  # DEBUG

                # Update the partner's model in the models' list
                model_list[partner_index] = partner_model
                local_score_list[partner_index] = history.history["val_accuracy"][0]

                # At the end of each mini-batch, for each partner, populate the extended score matrix
                score_matrix_extended[
                    epoch_index, minibatch_index, partner_index
                ] = history.history["val_accuracy"][0]

                # At the end of each epoch (on the last mini-batch), for each partner, populate the score matrix
                if minibatch_index == (minibatch_count - 1):
                    score_matrix[epoch_index, partner_index] = history.history[
                        "val_accuracy"
                    ][0]

        # At the end of each epoch, evaluate the aggregated model for early stopping on a global validation set
        aggregation_weights = prepare_aggregation_weights(
            aggregation_weighting, partners_count, partners_list, local_score_list
        )
        aggregated_model = build_aggregated_model(
            aggregate_model_weights(model_list, aggregation_weights)
        )
        model_evaluation = aggregated_model.evaluate(
            x_val_global, y_val_global, batch_size=constants.BATCH_SIZE, verbose=0,
        )
        current_val_loss = model_evaluation[0]
        global_val_acc.append(model_evaluation[1])
        global_val_loss.append(current_val_loss)
        print(
            "\n   Aggregated model evaluation at the end of the epoch:",
            model_evaluation,
        )

        print("      Checking if early stopping critera are met:")
        if is_early_stopping:
            # Early stopping parameters
            if (
                epoch_index >= constants.PATIENCE
                and current_val_loss > global_val_loss[-constants.PATIENCE]
            ):
                print("         -> Early stopping critera are met, stopping here.")
                break
            else:
                print("         -> Early stopping critera are not met, continuing with training.")

    # After last epoch or if early stopping was triggered, evaluate model on the global testset
    print("\n### Evaluating model on test data:")
    model_evaluation = aggregated_model.evaluate(
        x_test, y_test, batch_size=constants.BATCH_SIZE, verbose=0,
    )
    print("   Model metrics names: ", aggregated_model.metrics_names)
    print("   Model metrics values: ", ["%.3f" % elem for elem in model_evaluation])
    test_score = model_evaluation[1]  # 0 is for the loss

    # Plot training history
    if is_save_fig:

        # Save data
        np.save(save_folder / "score_matrix", score_matrix)
        np.save(save_folder / "global_val_acc", global_val_acc)
        np.save(save_folder / "global_val_loss", global_val_loss)
        np.save(save_folder / "score_matrix_extended", score_matrix_extended)

        plt.figure()
        plt.plot(global_val_loss)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig(save_folder / "federated_training_loss.png")

        plt.figure()
        plt.plot(global_val_acc)
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        # plt.yscale('log')
        plt.ylim([0, 1])
        plt.savefig(save_folder / "federated_training_acc.png")

        plt.figure()
        plt.plot(
            score_matrix[: epoch_index + 1,]
        )  # Cut the matrix
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["partner " + str(i) for i in range(partners_count)])
        # plt.yscale('log')
        plt.ylim([0, 1])
        plt.savefig(save_folder / "all_partners.png")

    print("\nTraining and evaluation on multiple partners: done.")
    return test_score
