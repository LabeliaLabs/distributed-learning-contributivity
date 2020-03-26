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


def preprocess_node_list(node_list):
    """Return node_list preprocessed for keras CNN"""

    print("\n## Pre-processing train data of each node for keras CNN:")
    for node_index, node in enumerate(node_list):

        # Preprocess input (x) data
        node.preprocess_data()

        # Crete validation dataset
        x_node_train, x_node_val, y_node_train, y_node_val = train_test_split(
            node.x_train, node.y_train, test_size=0.1, random_state=42
        )
        node.x_train = x_node_train
        node.x_val = x_node_val
        node.y_train = y_node_train
        node.y_val = y_node_val

        print("   Node #" + str(node_index) + ": done.")

    print("   Done.")
    return node_list


#%% Pre-process test data for model evaluation


def preprocess_test_data(x_test, y_test):
    """ Return x_test and y_test preprocessed for keras CNN"""

    print("\n## Pre-processing test data for keras CNN:")
    x_test = utils.preprocess_input(x_test)
    y_test = keras.utils.to_categorical(y_test, constants.NUM_CLASSES)

    print("   Done.")
    return x_test, y_test


#%% Single partner training


def compute_test_score_for_single_node(node, epoch_count):
    """Return the score on test data of a model trained on a single node"""

    print("\n## Training and evaluating model on one single node.")
    # Initialize model
    model = utils.generate_new_cnn_model()
    # print(model.summary())

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

    # Evaluate trained model
    print("\n### Evaluating model on test data of the node:")
    model_evaluation = model.evaluate(
        node.x_test, node.y_test, batch_size=constants.BATCH_SIZE, verbose=0
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
        scenario.x_esval,
        scenario.y_esval,
        scenario.x_test,
        scenario.y_test,
        scenario.is_early_stopping,
        scenario.amounts_per_node,
        is_save_fig,
        save_folder=scenario.save_folder,
    )


#%% Distributed learning training
def compute_test_score(
    node_list,
    epoch_count,
    x_esval,
    y_esval,
    x_test,
    y_test,
    is_early_stopping=True,
    aggregation_weights=None,
    is_save_fig=False,
    save_folder="",
):
    """Return the score on test data of a final aggregated model trained in a federated way on each node"""

    nodes_count = len(node_list)

    # If only one node, fall back to dedicated single node function
    if nodes_count == 1:
        return compute_test_score_for_single_node(node_list[0], epoch_count)

    # Else, follow a federated learning procedure
    else:
        print("\n## Training and evaluating model on multiple nodes: " + str(node_list))

        model_list = [None] * nodes_count
        epochs = epoch_count
        score_matrix = np.zeros(shape=(epochs, nodes_count))
        global_val_acc = []
        global_val_loss = []

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

                # OLD:
                # for weights_list_tuple in zip(*weights):  # TODO : make this clearer
                #     new_weights.append(
                #         [
                #             np.array(weights_).mean(axis=0)
                #             for weights_ in zip(*weights_list_tuple)
                #         ]
                #     )

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

                # Evaluate model for early stopping, on a dedicated 'early stopping validation' set
                model_evaluation = aggregated_model.evaluate(
                    x_esval, y_esval, batch_size=constants.BATCH_SIZE, verbose=0
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

                print("   Training on node " + str(node))
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

        # Evaluate model
        print("\n### Evaluating model on test data:")
        model_evaluation = final_model.evaluate(
            x_test, y_test, batch_size=constants.BATCH_SIZE, verbose=0
        )
        print("   Model metrics names: ", final_model.metrics_names)
        print("   Model metrics values: ", ["%.3f" % elem for elem in model_evaluation])

        test_score = model_evaluation[1]  # 0 is for the loss

        print("\nTraining and evaluation on multiple nodes: done.")
        return test_score
