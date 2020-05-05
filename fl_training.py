# -*- coding: utf-8 -*-
"""
Functions for model training and evaluation (single-partner and multi-partner cases)
(inspired from: https://keras.io/examples/mnist_cnn/)
"""

from __future__ import print_function

import keras
from keras.backend.tensorflow_backend import clear_session
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import operator
from loguru import logger

import utils
import constants

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def preprocess_scenarios_data(scenario):
    """Return scenario with central datasets (val, test) and distributed datasets (partners) pre-processed"""

    logger.info("## Pre-processing datasets of the scenario for keras CNN:")

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

        if scenario.corrupted_datasets[partner_index] == "corrupted":
            logger.info(f"   ... Corrupting data (offsetting labels) of partner #{partner.id}")
            partner.corrupt_labels()
        elif scenario.corrupted_datasets[partner_index] == "shuffled":
            logger.info(f"   ... Corrupting data (shuffling labels) of partner #{partner.id}")
            partner.shuffle_labels()
        elif scenario.corrupted_datasets[partner_index] == "not_corrupted":
            pass
        else:
            logger.info("Unexpected label of corruption, not corruption performed!")

        logger.info(f"   Partner #{partner.id}: done.")

    # Then the scenario central dataset of the scenario
    scenario.x_val = utils.preprocess_input(scenario.x_val)
    scenario.y_val = keras.utils.to_categorical(scenario.y_val, constants.NUM_CLASSES)
    logger.info("   Central early stopping validation set: done.")
    scenario.x_test = utils.preprocess_input(scenario.x_test)
    scenario.y_test = keras.utils.to_categorical(scenario.y_test, constants.NUM_CLASSES)
    logger.info("   Central testset: done.")

    return scenario


def compute_test_score_for_single_partner(
    partner, epoch_count, single_partner_test_mode, global_x_test, global_y_test, is_early_stopping
):
    """Return the score on test data of a model trained on a single partner"""

    logger.info(f"## Training and evaluating model on partner with id #{partner.id}")

    # Initialize model
    model = utils.generate_new_cnn_model()

    # Set if early stopping if needed
    cb = []
    if is_early_stopping:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=constants.PATIENCE)
        cb.append(es)

    # Train model
    logger.info("   Training model...")
    history = model.fit(
        partner.x_train,
        partner.y_train,
        batch_size=partner.batch_size,
        epochs=epoch_count,
        verbose=0,
        validation_data=(partner.x_val, partner.y_val),
        callbacks=cb,
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
    model_evaluation = model.evaluate(x_test, y_test, batch_size=constants.DEFAULT_BATCH_SIZE, verbose=0)
    logger.info(f"   Model evaluation on test data: "
                f"{list(zip(model.metrics_names,['%.3f' % elem for elem in model_evaluation]))}")

    model_eval_score = model_evaluation[1]  # 0 is for the loss

    # Return model score on test data
    return model_eval_score


def compute_test_score_with_scenario(scenario, is_save_fig=False):
    return compute_test_score(
        scenario.partners_list,
        scenario.epoch_count,
        scenario.x_val,
        scenario.y_val,
        scenario.x_test,
        scenario.y_test,
        scenario.multi_partner_learning_approach,
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


def prepare_aggregation_weights(aggregation_weighting, partners_list, input_weights):
    """Returns a list of weights for the weighted average aggregation of model weights"""

    partners_count = len(partners_list)
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


def init_with_new_models(partners_list):
    """Return a list of newly generated models, one per partner"""

    partners_model_list = [None] * len(partners_list)
    for partner_index, partner in enumerate(partners_list):
        partners_model_list[partner_index] = utils.generate_new_cnn_model()
    return partners_model_list


def init_with_agg_models(partners_list, aggregation_weighting_approach, local_score_list, model_list):
    """Return a list with the aggregated model duplicated for each partner"""

    aggregation_weights = prepare_aggregation_weights(aggregation_weighting_approach, partners_list, local_score_list)
    partners_model_list = [None] * len(partners_list)
    for partner_index, partner in enumerate(partners_list):
        partners_model_list[partner_index] = build_aggregated_model(
            aggregate_model_weights(model_list, aggregation_weights)
        )
    return partners_model_list


def init_with_agg_model(partners_list, aggregation_weighting_approach, local_score_list, model_list):
    """Return a new model aggregating models passed as argument"""

    aggregation_weights = prepare_aggregation_weights(aggregation_weighting_approach, partners_list, local_score_list)
    new_agg_model = build_aggregated_model(aggregate_model_weights(model_list, aggregation_weights))

    return new_agg_model


def build_from_previous_model(previous_model):
    """Return a new model initialized with weights of a given model"""

    new_model_from_previous_model = utils.generate_new_cnn_model()
    new_model_from_previous_model.set_weights(previous_model.get_weights())
    new_model_from_previous_model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer="adam",
        metrics=["accuracy"],
    )

    return new_model_from_previous_model


def collaborative_round_fit(model_to_fit, train_data, val_data, batch_size):
    """Fit the model with arguments passed as parameters and returns the history object"""

    x_train, y_train = train_data
    history = model_to_fit.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=1,
        verbose=0,
        validation_data=val_data,
    )

    return history


def print_collaborative_round_partner_result(partner, partners_count, collaborative_round_indexes, validation_score):
    """Print the validation accuracy of the collaborative round"""

    epoch_index, epoch_count, minibatch_index, minibatch_count = collaborative_round_indexes

    epoch_nb_str = f"Epoch {str(epoch_index).zfill(2)}/{str(epoch_count - 1).zfill(2)}"
    mb_nb_str = f"Minibatch {str(minibatch_index).zfill(2)}/{str(minibatch_count - 1).zfill(2)}"
    partner_id_str = f"Partner {partner.id}/{partners_count}"
    val_acc_str = f"{round(validation_score, 2)}"

    logger.info(f"{epoch_nb_str} > {mb_nb_str} > {partner_id_str} > val_acc: {val_acc_str}")


def update_iterative_results(partner_index, collaborative_round_indexes, fit_history, iterative_results):
    """Update the results arrays with results from the collaboration round"""

    epoch_index, epoch_count, minibatch_index, minibatch_count = collaborative_round_indexes
    local_score_list, model_list, score_matrix, score_matrix_extended = iterative_results
    validation_score = fit_history.history["val_accuracy"][0]

    local_score_list[partner_index] = validation_score

    # At the end of each mini-batch, for each partner, populate the extended score matrix
    score_matrix_extended[epoch_index, minibatch_index, partner_index] = validation_score

    # At the end of each epoch (at end of last mini-batch), for each partner, populate the score matrix
    if minibatch_index == (minibatch_count - 1):
        score_matrix[epoch_index, partner_index] = validation_score


def compute_collaborative_round_fedavg(
        partners_list,
        aggregation_weighting_approach,
        val_data,
        iterative_results,
        train_data,
        collaborative_round_indexes,
):
    """Proceed to a collaborative round with a federated averaging approach"""

    logger.debug("Start new fedavg collaborative round ...")

    # Initialize variables
    partners_count = len(partners_list)
    epoch_index, epoch_count, minibatch_index, minibatch_count = collaborative_round_indexes
    is_very_first_minibatch = (epoch_index == 0 and minibatch_index == 0)
    local_score_list, model_list, score_matrix, score_matrix_extended = iterative_results
    minibatched_x_train, minibatched_y_train = train_data

    # Starting model for each partner is the aggregated model from the previous mini-batch iteration
    if is_very_first_minibatch:  # Except for the very first mini-batch where it is a new model
        logger.debug(f"(fedavg) Very first minibatch of epoch n°{epoch_index}, init new models for each partner")
        partners_model_list_for_iteration = init_with_new_models(partners_list)
    else:
        logger.debug(f"(fedavg) Minibatch n°{minibatch_index} of epoch n°{epoch_index}, "
                     f"init aggregated model for each partner with models from previous round")
        partners_model_list_for_iteration = init_with_agg_models(
            partners_list, aggregation_weighting_approach, local_score_list, model_list)

    # Iterate over partners for training each individual model
    for partner_index, partner in enumerate(partners_list):

        logger.debug(f"(fedavg) Partner n°{partner_index}, reference partner's model, "
                     f"then train and validate on validation data")

        # Reference the partner's model
        partner_model = partners_model_list_for_iteration[partner_index]

        # Train on partner local data set
        train_data_for_fit_iteration = (
            minibatched_x_train[partner_index][minibatch_index],
            minibatched_y_train[partner_index][minibatch_index],
        )
        history = collaborative_round_fit(partner_model, train_data_for_fit_iteration, val_data, partner.batch_size)

        # Print results of the round
        print_collaborative_round_partner_result(
            partner, partners_count, collaborative_round_indexes, history.history["val_accuracy"][0])

        # Update the partner's model in the models' list
        model_list[partner_index] = partner_model

        # Update iterative results
        update_iterative_results(partner_index, collaborative_round_indexes, history, iterative_results)

    logger.debug("End of fedavg collaborative round.")


def compute_collaborative_round_sequential(
        partners_list,
        sequentially_trained_model,
        val_data,
        iterative_results,
        train_data,
        collaborative_round_indexes,
):
    """Proceed to a collaborative round with a sequential approach"""

    logger.debug("Start new sequential collaborative round ...")

    # Initialize variables
    partners_count = len(partners_list)
    epoch_index, epoch_count, minibatch_index, minibatch_count = collaborative_round_indexes
    minibatched_x_train, minibatched_y_train = train_data

    # Iterate over partners for training the model sequentially
    for partner_index, partner in enumerate(partners_list):

        logger.debug(f"(seq) Partner n°{partner_index}, train the model sequentially")

        # Train on partner local data set
        train_data_for_fit_iteration = (
            minibatched_x_train[partner_index][minibatch_index],
            minibatched_y_train[partner_index][minibatch_index],
        )
        history = collaborative_round_fit(
            sequentially_trained_model, train_data_for_fit_iteration, val_data, partner.batch_size)

        # Print results of the round
        print_collaborative_round_partner_result(
            partner, partners_count, collaborative_round_indexes, history.history["val_accuracy"][0])

        # Update iterative results
        update_iterative_results(partner_index, collaborative_round_indexes, history, iterative_results)

    logger.debug("End of sequential collaborative round.")


def compute_collaborative_round_seqavg(
        partners_list,
        aggregation_weighting_approach,
        val_data,
        iterative_results,
        train_data,
        collaborative_round_indexes,
):
    """Proceed to a collaborative round with a sequential averaging approach"""

    logger.debug("Start new seqavg collaborative round ...")

    # Initialize variables
    partners_count = len(partners_list)
    epoch_index, epoch_count, minibatch_index, minibatch_count = collaborative_round_indexes
    is_very_first_minibatch = (epoch_index == 0 and minibatch_index == 0)
    local_score_list, model_list, score_matrix, score_matrix_extended = iterative_results
    minibatched_x_train, minibatched_y_train = train_data

    # Starting model for each partner is the aggregated model from the previous mini-batch iteration
    if is_very_first_minibatch:  # Except for the very first mini-batch where it is a new model
        logger.debug(f"(seqavg) Very first minibatch of epoch n°{epoch_index}, init a new model for the round")
        model_for_round = utils.generate_new_cnn_model()
    else:
        logger.debug(f"(seqavg) Minibatch n°{minibatch_index} of epoch n°{epoch_index}, "
                     f"init aggregated model with models from previous round")
        model_for_round = init_with_agg_model(
            partners_list, aggregation_weighting_approach, local_score_list, model_list)

    # Iterate over partners for training each individual model
    for partner_index, partner in enumerate(partners_list):
        logger.debug(f"(seqavg) Partner n°{partner_index}, reference partner's model in iteration list, "
                     f"then train and validate on validation data")

        # Reference the partner's model
        if partner_index == 0:
            partner_model = model_for_round
        else:
            partner_model = build_from_previous_model(model_list[partner_index-1])

        # Train on partner local data set
        train_data_for_fit_iteration = (
            minibatched_x_train[partner_index][minibatch_index],
            minibatched_y_train[partner_index][minibatch_index],
        )
        history = collaborative_round_fit(partner_model, train_data_for_fit_iteration, val_data, partner.batch_size)

        # Print results of the round
        print_collaborative_round_partner_result(
            partner, partners_count, collaborative_round_indexes, history.history["val_accuracy"][0])

        # Update the partner's model in the models' list
        model_list[partner_index] = partner_model

        # Update iterative results
        update_iterative_results(partner_index, collaborative_round_indexes, history, iterative_results)

    logger.debug("End of seqavg collaborative round.")


def compute_test_score(
    partners_list,
    epoch_count,
    x_val_global,
    y_val_global,
    x_test,
    y_test,
    multi_partner_learning_approach,
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
            partners_list[0], epoch_count, single_partner_test_mode, x_test, y_test, is_early_stopping
        )

    # Else, continue onto a federated learning procedure
    partners_list = sorted(partners_list, key=operator.attrgetter("id"))
    logger.info(f"## Training and evaluating model on partners with ids: {['#'+str(p.id) for p in partners_list]}")

    # Initialize variables
    val_data = (x_val_global, y_val_global)
    model_list, local_score_list = [None] * partners_count, [None] * partners_count
    score_matrix = np.zeros(shape=(epoch_count, partners_count))
    score_matrix_extended = np.zeros(shape=(epoch_count, minibatch_count, partners_count))
    global_val_acc, global_val_loss = [], []
    iterative_results = (local_score_list, model_list, score_matrix, score_matrix_extended)
    if multi_partner_learning_approach == 'seq':
        sequentially_trained_model = utils.generate_new_cnn_model()

    # Train model (iterate for each epoch and mini-batch)
    for epoch_index in range(epoch_count):

        if multi_partner_learning_approach != 'seq':
            clear_session()

        # Split the train dataset in mini-batches
        minibatched_x_train, minibatched_y_train = [None] * partners_count, [None] * partners_count
        for partner_index, partner in enumerate(partners_list):
            (
                minibatched_x_train[partner_index],
                minibatched_y_train[partner_index],
            ) = split_in_minibatches(minibatch_count, partner.x_train, partner.y_train)
        current_epoch_train_data = (minibatched_x_train, minibatched_y_train)

        # Iterate over mini-batches for training
        for minibatch_index in range(minibatch_count):

            collaborative_round_indexes = (epoch_index, epoch_count, minibatch_index, minibatch_count)

            if multi_partner_learning_approach == 'fedavg':
                compute_collaborative_round_fedavg(
                    partners_list,
                    aggregation_weighting,
                    val_data,
                    iterative_results,
                    current_epoch_train_data,
                    collaborative_round_indexes,
                )

            elif multi_partner_learning_approach == 'seq':
                compute_collaborative_round_sequential(
                    partners_list,
                    sequentially_trained_model,
                    val_data,
                    iterative_results,
                    current_epoch_train_data,
                    collaborative_round_indexes,
                )

            elif multi_partner_learning_approach == 'seqavg':
                compute_collaborative_round_seqavg(
                    partners_list,
                    aggregation_weighting,
                    val_data,
                    iterative_results,
                    current_epoch_train_data,
                    collaborative_round_indexes,
                )

        # At the end of each epoch, evaluate the model for early stopping on a global validation set
        if multi_partner_learning_approach == 'fedavg' or multi_partner_learning_approach == 'seqavg':
            aggregation_weights = prepare_aggregation_weights(aggregation_weighting, partners_list, local_score_list)
            model_to_evaluate = build_aggregated_model(aggregate_model_weights(model_list, aggregation_weights))
        elif multi_partner_learning_approach == 'seq':
            model_to_evaluate = sequentially_trained_model
        model_evaluation = model_to_evaluate.evaluate(
            x_val_global, y_val_global, batch_size=constants.DEFAULT_BATCH_SIZE, verbose=0,
        )
        current_val_loss = model_evaluation[0]
        global_val_acc.append(model_evaluation[1])
        global_val_loss.append(current_val_loss)
        logger.info(f"   Model evaluation at the end of the epoch: "
                    f"{['%.3f' % elem for elem in model_evaluation]}")

        logger.info("      Checking if early stopping criteria are met:")
        if is_early_stopping:
            # Early stopping parameters
            if (
                epoch_index >= constants.PATIENCE
                and current_val_loss > global_val_loss[-constants.PATIENCE]
            ):
                logger.info("         -> Early stopping criteria are met, stopping here.")
                break
            else:
                logger.info("         -> Early stopping criteria are not met, continuing with training.")

    # After last epoch or if early stopping was triggered, evaluate model on the global testset
    logger.info("### Evaluating model on test data:")
    model_evaluation = model_to_evaluate.evaluate(x_test, y_test, batch_size=constants.DEFAULT_BATCH_SIZE, verbose=0)
    logger.info(f"   Model metrics names: {model_to_evaluate.metrics_names}")
    logger.info(f"   Model metrics values: {['%.3f' % elem for elem in model_evaluation]}")
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
        plt.close()

        plt.figure()
        plt.plot(global_val_acc)
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        # plt.yscale('log')
        plt.ylim([0, 1])
        plt.savefig(save_folder / "federated_training_acc.png")
        plt.close()

        plt.figure()
        plt.plot(
            score_matrix[: epoch_index + 1, ]
        )  # Cut the matrix
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["partner " + str(i) for i in range(partners_count)])
        # plt.yscale('log')
        plt.ylim([0, 1])
        plt.savefig(save_folder / "all_partners.png")
        plt.close()

    logger.info("Training and evaluation on multiple partners: done.")
    return test_score
