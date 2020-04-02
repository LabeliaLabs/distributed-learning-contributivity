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
    """Return scenario with central datasets (val, test) and distributed datasets (nodes) pre-processed"""

    print("\n## Pre-processing datasets of the scenario for keras CNN:")

    # First, datasets of each node
    for node_index, node in enumerate(scenario.node_list):

        # Preprocess inputs (x) data
        node.x_train = utils.preprocess_input(node.x_train)
        node.x_test = utils.preprocess_input(node.x_test)

        # Preprocess labels (y) data
        node.y_train = keras.utils.to_categorical(node.y_train, constants.NUM_CLASSES)
        node.y_test = keras.utils.to_categorical(node.y_test, constants.NUM_CLASSES)

        # Create validation dataset
        node.x_train, node.x_val, node.y_train, node.y_val = train_test_split(
            node.x_train, node.y_train, test_size=0.1, random_state=42
        )

        if scenario.corrupted_nodes[node_index] == "corrupted":
            print("corruption of node " + str(node_index) + "\n")
            node.corrupt_labels()
        elif scenario.corrupted_nodes[node_index] == "shuffled":
            print("shuffleling of node " + str(node_index) + "\n")
            node.shuffle_labels()
        elif scenario.corrupted_nodes[node_index] == "not_corrupted":
            pass
        else:
            print("Unexpected label of corruption")

        print("   Node #" + str(node_index) + ": done.")

    # Then the scenario central dataset of the scenario
    scenario.x_val = utils.preprocess_input(scenario.x_val)
    scenario.y_val = keras.utils.to_categorical(scenario.y_val, constants.NUM_CLASSES)
    print("   Central early stopping validation set: done.")
    scenario.x_test = utils.preprocess_input(scenario.x_test)
    scenario.y_test = keras.utils.to_categorical(scenario.y_test, constants.NUM_CLASSES)
    print("   Central testset: done.")

    return scenario


#%% Single partner training


def compute_test_score_for_single_node(
    node, epoch_count, single_partner_test_mode, global_x_test, global_y_test
):
    """Return the score on test data of a model trained on a single node"""

    print("\n## Training and evaluating model on one single node.")

    # Initialize model
    model = utils.generate_new_cnn_model()

    # Train model
    print("\n### Training model on one single node: " + str(node))
    history = model.fit(
        node.x_train,
        node.y_train,
        batch_size=constants.BATCH_SIZE,
        epochs=epoch_count,
        verbose=0,
        validation_data=(node.x_val, node.y_val),
    )

    # Reference a testset according to the scenario configuration
    if single_partner_test_mode == "global":
        x_test = global_x_test
        y_test = global_y_test
    elif single_partner_test_mode == "local":
        x_test = node.x_test
        y_test = node.y_test
    else:
        raise NameError(
            "This single_partner_test_mode option ["
            + single_partner_test_mode
            + "] is not recognized."
        )

    # Evaluate trained model
    print("\n### Evaluating model on test data of the node:")
    model_evaluation = model.evaluate(
        x_test, y_test, batch_size=constants.BATCH_SIZE, verbose=0
    )
    print("   Model metrics names: ", model.metrics_names)
    print("   Model metrics values: ", ["%.3f" % elem for elem in model_evaluation])

    model_eval_score = model_evaluation[1]  # 0 is for the loss

    # Return model score on test data
    print("\nTraining and evaluation on one single node: done.")
    return model_eval_score


#%% TODO no methods overloading


def compute_test_score_with_scenario(scenario, is_save_fig=False):
    return compute_test_score(
        scenario.node_list,
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


#%% Distributed learning training

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

def prepare_aggregation_weights(aggregation_weighting, nodes_count, node_list, input_weights):
    """Returns a list of weights for the weighted average aggregation of model weights"""

    aggregation_weights = []
    if aggregation_weighting == "uniform":
        aggregation_weights = [1/nodes_count] * nodes_count
    elif aggregation_weighting == "data_volume":
        node_sizes = [len(node.x_train) for node in node_list]
        aggregation_weights = node_sizes / np.sum(node_sizes)
    elif aggregation_weighting == "local_perf":
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
        avg_weights_for_layer = np.average(np.array(weights_for_layer), axis=0, weights=aggregation_weights)
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
    node_list,
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
    """Return the score on test data of a final aggregated model trained in a federated way on each node"""

    # First, if only one node, fall back to dedicated single node function
    nodes_count = len(node_list)
    if nodes_count == 1:
        return compute_test_score_for_single_node(
            node_list[0], epoch_count, single_partner_test_mode, x_test, y_test
        )

    # Else, continue onto a federated learning procedure
    print("\n## Training and evaluating model on multiple nodes: " + str(node_list))

    # Initialize variables
    model_list, local_perf_list = [None] * nodes_count, [None] * nodes_count
    score_matrix = np.zeros(shape=(epoch_count, nodes_count))
    global_val_acc, global_val_loss = [], []

    # Train model (iterate for each epoch and mini-batch)
    print("\n### Training model:")
    for epoch in range(epoch_count):

        print("\n   Epoch " + str(epoch) + " out of " + str(epoch_count-1) + " total epochs")
        is_first_epoch = epoch == 0
        clear_session()

        # Split the train dataset in mini-batches
        minibatched_x_train, minibatched_y_train = [None] * nodes_count, [None] * nodes_count
        for node_index, node in enumerate(node_list):
            minibatched_x_train[node_index], minibatched_y_train[node_index] = split_in_minibatches(minibatch_count, node.x_train, node.y_train)

        # Iterate over mini-batches for training, starting each new iteration with an aggregation of the previous one
        for minibatch_index in range(minibatch_count):

            print("\n      Mini-batch " + str(minibatch_index) + " out of " + str(minibatch_count-1) + " total mini-batches")
            is_first_minibatch = minibatch_index == 0

            # Starting model for each node is the aggregated model from the previous mini-batch iteration
            agg_model_for_iteration = [None] * nodes_count
            aggregation_weights = prepare_aggregation_weights(aggregation_weighting, nodes_count, node_list, local_perf_list)
            for node_index, node in enumerate(node_list):
                if is_first_epoch and is_first_minibatch:
                    agg_model_for_iteration[node_index] = utils.generate_new_cnn_model()
                else:
                    agg_model_for_iteration[node_index] = build_aggregated_model(aggregate_model_weights(model_list, aggregation_weights))

            # Iterate over nodes for training each individual model
            for node_index, node in enumerate(node_list):

                node_model = agg_model_for_iteration[node_index]

                # Train on node local data set
                print("         Training on node " + str(node_index) + " - " + str(node))
                history = node_model.fit(
                    minibatched_x_train[node_index][minibatch_index],
                    minibatched_y_train[node_index][minibatch_index],
                    batch_size=constants.BATCH_SIZE,
                    epochs=1,
                    verbose=0,
                    validation_data=(x_val_global, y_val_global),
                )
                print("            val_accuracy: " + str(history.history["val_accuracy"][0])) # DEBUG

                # Update the node's model in the models' list
                model_list[node_index] = node_model
                local_perf_list[node_index] = history.history["val_accuracy"][0]

                # At the end of each epoch (on the last mini-batch), for each node, populate the score matrix
                if minibatch_index == (minibatch_count - 1):
                    score_matrix[epoch, node_index] = history.history["val_accuracy"][0]

        # At the end of each epoch, evaluate the aggregated model for early stopping on a global validation set
        aggregation_weights = prepare_aggregation_weights(aggregation_weighting, nodes_count, node_list, local_perf_list)
        aggregated_model = build_aggregated_model(aggregate_model_weights(model_list, aggregation_weights))
        model_evaluation = aggregated_model.evaluate(
            x_val_global,
            y_val_global,
            batch_size=constants.BATCH_SIZE,
            verbose=0,
        )
        current_val_loss = model_evaluation[0]
        global_val_acc.append(model_evaluation[1])
        global_val_loss.append(current_val_loss)
        print("\n   Aggregated model evaluation at the end of the epoch:", model_evaluation)

        print("      Checking if early stopping critera are met:")
        if is_early_stopping:
            # Early stopping parameters
            if (
                epoch >= constants.PATIENCE
                and current_val_loss > global_val_loss[-constants.PATIENCE]
            ):
                print("         -> Early stopping critera are met, stopping here.")
                break
            else:
                print("         -> Early stopping critera are not met, continuing with training.")

    # After last epoch or if early stopping was triggered, evaluate model on the global testset
    print("\n### Evaluating model on test data:")
    model_evaluation = aggregated_model.evaluate(
        x_test,
        y_test,
        batch_size=constants.BATCH_SIZE,
        verbose=0,
    )
    print("   Model metrics names: ", aggregated_model.metrics_names)
    print("   Model metrics values: ", ["%.3f" % elem for elem in model_evaluation])
    test_score = model_evaluation[1] # 0 is for the loss

    # Plot training history
    if is_save_fig:

        # Save data
        np.save(save_folder / "score_matrix", score_matrix)
        np.save(save_folder / "global_val_acc", global_val_acc)
        np.save(save_folder / "global_val_loss", global_val_loss)

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
            score_matrix[: epoch + 1,]
        )  # Cut the matrix
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Node " + str(i) for i in range(nodes_count)])
        # plt.yscale('log')
        plt.ylim([0, 1])
        plt.savefig(save_folder / "all_nodes.png")

    print("\nTraining and evaluation on multiple nodes: done.")
    return test_score
