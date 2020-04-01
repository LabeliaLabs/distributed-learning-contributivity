# -*- coding: utf-8 -*-
"""
Train a model across multiple nodes
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
        elif scenario.corrupted_nodes[node_index] == "not-corrupted":
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
    # print(model.summary())

    # Train model
    print("\n### Training model on one single node: " + str(node.node_id))
    history = model.fit(
        node.x_train,
        node.y_train,
        batch_size=constants.BATCH_SIZE,
        epochs=epoch_count,
        verbose=0,
        validation_data=(node.x_val, node.y_val),
    )

    # Reference testset according to scenario
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
        scenario.is_early_stopping,
        scenario.single_partner_test_mode,
        is_save_fig,
        save_folder=scenario.save_folder,
    )


#%% Distributed learning training


def compute_test_score(
    node_list,
    epoch_count,
    x_val_global,
    y_val_global,
    x_test,
    y_test,
    aggregation_weighting="uniform",
    is_early_stopping=True,
    single_partner_test_mode="global",
    is_save_fig=False,
    save_folder="",
):
    """Return the score on test data of a final aggregated model trained in a federated way on each node"""

    nodes_count = len(node_list)

    # If only one node, fall back to dedicated single node function
    if nodes_count == 1:
        return compute_test_score_for_single_node(
            node_list[0], epoch_count, single_partner_test_mode, x_test, y_test
        )

    # Else, follow a federated learning procedure
    else:
        print("\n## Training and evaluating model on multiple nodes: " + str(node_list))

        model_list = [None] * nodes_count
        epochs = epoch_count
        score_matrix = np.zeros(shape=(epochs, nodes_count))
        global_val_acc = []
        global_val_loss = []

        # Create list of weights for aggregation steps
        aggregation_weights = []
        if aggregation_weighting == "uniform":
            aggregation_weights = [1/nodes_count] * nodes_count
        elif aggregation_weighting == "data-volume":
            node_sizes = [len(node.x_train) for node in node_list]
            aggregation_weights = node_sizes / np.sum(node_sizes)
        assert (np.sum(aggregation_weights) == 1)

        print("\n### Training model:")
        for epoch in range(epochs):

            print(
                "\nEpoch #"
                + str(epoch + 1)
                + " out of "
                + str(epochs)
                + " total epochs"
            )
            is_first_epoch = epoch == 0
            clear_session()

            # Aggregation, intermediate evaluation and early stopping phase
            if is_first_epoch:
                # First epoch, no aggregation
                print("   First epoch, generate model from scratch")

            else:
                # Aggregating phase: averaging the weights
                print("   Aggregating models weights to build a new model")
                weights_per_model = [model.get_weights() for model in model_list]
                weights_per_layer = list(zip(*weights_per_model))
                new_weights = list()

                for weights_for_layer in weights_per_layer:
                    avg_weights_for_layer = np.average(np.array(weights_for_layer), axis=0, weights=aggregation_weights)
                    new_weights.append(list(avg_weights_for_layer))

                aggregated_model = utils.generate_new_cnn_model()
                aggregated_model.set_weights(new_weights)
                aggregated_model.compile(
                    loss=keras.losses.categorical_crossentropy,
                    optimizer="adam",
                    metrics=["accuracy"],
                )

                # Evaluate model for early stopping, on a central and dedicated 'early stopping validation' set
                model_evaluation = aggregated_model.evaluate(
                    x_val_global,
                    y_val_global,
                    batch_size=constants.BATCH_SIZE,
                    verbose=0,
                )
                current_val_loss = model_evaluation[0]
                global_val_acc.append(model_evaluation[1])
                global_val_loss.append(current_val_loss)

                # Early stopping
                print("   Checking if early stopping critera are met:")
                if is_early_stopping:
                    # Early stopping parameters
                    if (
                        epoch >= constants.PATIENCE
                        and current_val_loss > global_val_loss[-constants.PATIENCE]
                    ):
                        print("      -> Early stopping critera are met, stopping here.")
                        break
                    else:
                        print(
                            "      -> Early stopping critera are not met, continuing with training."
                        )

            # Training phase
            val_acc_list = []
            acc_list = []
            for node_index, node in enumerate(node_list):

                print("   Training on node " + str(node.node_id))
                node_model = utils.generate_new_cnn_model()

                # Model weights are the averaged weights
                if not is_first_epoch:
                    node_model.set_weights(new_weights)
                    node_model.compile(
                        loss=keras.losses.categorical_crossentropy,
                        optimizer="adam",
                        metrics=["accuracy"],
                    )

                # Train on whole node local data set
                history = node_model.fit(
                    node.x_train,
                    node.y_train,
                    batch_size=constants.BATCH_SIZE,
                    epochs=1,
                    verbose=0,
                    validation_data=(node.x_val, node.y_val),
                )

                val_acc_list.append(history.history["val_accuracy"])
                acc_list.append(history.history["accuracy"])
                score_matrix[epoch, node_index] = history.history["val_accuracy"][0]
                model_list[node_index] = node_model

        # Final aggregation: averaging the weights
        weights = [model.get_weights() for model in model_list]
        new_weights = list()
        for weights_list_tuple in zip(*weights):
            new_weights.append(
                [
                    np.array(weights_).mean(axis=0)
                    for weights_ in zip(*weights_list_tuple)
                ]
            )

        final_model = utils.generate_new_cnn_model()
        final_model.set_weights(new_weights)
        final_model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer="adam",
            metrics=["accuracy"],
        )

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

        # Evaluate model on a central and dedicated testset
        print("\n### Evaluating model on test data:")
        model_evaluation = final_model.evaluate(
            x_test, y_test, batch_size=constants.BATCH_SIZE, verbose=0
        )
        print("   Model metrics names: ", final_model.metrics_names)
        print("   Model metrics values: ", ["%.3f" % elem for elem in model_evaluation])

        test_score = model_evaluation[1]  # 0 is for the loss

        print("\nTraining and evaluation on multiple nodes: done.")
        return test_score
