# -*- coding: utf-8 -*-
"""
Functions for model training and evaluation (single-partner and multi-partner cases)
"""

import os
from timeit import default_timer as timer
import pickle
import keras
from keras.backend.tensorflow_backend import clear_session
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import operator
from loguru import logger

import constants


class MultiPartnerLearning:

    def __init__(self,
                 partners_list,
                 epoch_count,
                 minibatch_count,
                 dataset,
                 multi_partner_learning_approach,
                 aggregation_weighting="uniform",
                 folder_for_starting_model=None,
                 is_early_stopping=True,
                 is_save_data=False,
                 save_folder="",
                 ):

        # Attributes related to partners
        self.partners_list = partners_list
        self.partners_count = len(partners_list)

        # Attributes related to the data and the model
        self.val_data = (dataset.x_val, dataset.y_val)
        self.test_data = (dataset.x_test, dataset.y_test)
        self.generate_new_model = dataset.generate_new_model

        # Attributes related to the multi-partner learning approach
        self.learning_approach = multi_partner_learning_approach
        self.aggregation_weighting = aggregation_weighting

        # Attributes related to iterating at different levels
        self.epoch_count = epoch_count
        self.epoch_index = 0
        self.minibatch_count = minibatch_count
        self.minibatch_index = 0
        self.is_early_stopping = is_early_stopping

        # Attributes for storing intermediate artefacts and results
        self.minibatched_x_train = [None] * self.partners_count
        self.minibatched_y_train = [None] * self.partners_count
        self.aggregation_weights = []
        self.models_weights_list = [None] * self.partners_count
        if not folder_for_starting_model:
            self.federated_model_weights = None
        else:
            model = self.generate_new_model()
            model.load_weights(folder_for_starting_model) 
            self.federated_model_weights = model.get_weight()
        self.scores_last_learning_round = [None] * self.partners_count
        self.score_matrix_per_partner = np.nan * np.zeros(shape=(self.epoch_count, self.minibatch_count, self.partners_count))
        self.score_matrix_collective_models = np.nan * np.zeros(shape=(self.epoch_count, self.minibatch_count + 1))
        self.loss_collective_models = []
        self.test_score = None
        self.nb_epochs_done = int
        self.is_save_data = is_save_data
        self.save_folder = save_folder
        self.learning_computation_time = None

        logger.debug("MultiPartnerLearning object instantiated.")

    def compute_test_score_for_single_partner(self, partner):
        """Return the score on test data of a model trained on a single partner"""

        start = timer()
        logger.info(f"## Training and evaluating model on partner with id #{partner.id}")

        # Initialize model
        model = self.generate_new_model()

        # Set if early stopping if needed
        cb = []
        if self.is_early_stopping:
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=constants.PATIENCE)
            cb.append(es)

        # Train model
        logger.info("   Training model...")
        history = model.fit(
            partner.x_train,
            partner.y_train,
            batch_size=partner.batch_size,
            epochs=self.epoch_count,
            verbose=0,
            validation_data=self.val_data,
            callbacks=cb,
        )

        # Reference the testset according to the scenario configuration
        x_test, y_test = self.test_data

        # Evaluate trained model
        model_evaluation = model.evaluate(x_test, y_test, batch_size=constants.DEFAULT_BATCH_SIZE, verbose=0)
        logger.info(f"   Model evaluation on test data: "
                    f"{list(zip(model.metrics_names, ['%.3f' % elem for elem in model_evaluation]))}")

        # Save model score on test data
        self.test_score = model_evaluation[1]  # 0 is for the loss
        self.nb_epochs_done = (es.stopped_epoch + 1) if es.stopped_epoch != 0 else self.epoch_count

        end = timer()
        self.learning_computation_time = end - start

    def compute_test_score(self, start_from_federated_model=False, ):
        """Return the score on test data of a final aggregated model trained in a federated way on each partner"""

        start = timer()

        partners_list = self.partners_list
        partners_count = self.partners_count

        epoch_count = self.epoch_count
        minibatch_count = self.minibatch_count
        is_early_stopping = self.is_early_stopping

        x_val, y_val = self.val_data
        x_test, y_test = self.test_data

        # First, if only one partner, fall back to dedicated single partner function
        if partners_count == 1:
            return self.compute_test_score_for_single_partner(partners_list[0])

        # Else, continue onto a federated learning procedure
        partners_list = sorted(partners_list, key=operator.attrgetter("id"))
        logger.info(
            f"## Training and evaluating model on partners with ids: {['#' + str(p.id) for p in partners_list]}")

        # Initialize variables
        model_to_evaluate, sequentially_trained_model = None, None
        if self.learning_approach in ['seq-pure', 'seq-with-final-agg']:
            if start_from_federated_model:
                sequentially_trained_model = self.build_model_from_weights(self.federated_model_weights)
            else:
                sequentially_trained_model = self.generate_new_model()
        else:
            if start_from_federated_model:
                self.models_weights_list = [self.federated_model_weights] * self.partners_count
            else:
                new_model = self.generate_new_model()
                for i in range(self.partners_count):
                    self.models_weights_list[i] = new_model.get_weights() 
         

        # Train model (iterate for each epoch and mini-batch)
        for epoch_index in range(epoch_count):

            self.epoch_index = epoch_index

            # Clear Keras' old models (except if the approach is sequential and the model has to persist across epochs)
            if self.learning_approach not in ['seq-pure', 'seq-with-final-agg']:
                clear_session()

            # Split the train dataset in mini-batches
            self.split_in_minibatches()

            # Iterate over mini-batches and train
            for minibatch_index in range(minibatch_count):

                self.minibatch_index = minibatch_index

                if self.learning_approach == 'fedavg':
                    self.compute_collaborative_round_fedavg()

                elif self.learning_approach in ['seq-pure', 'seq-with-final-agg']:
                    self.compute_collaborative_round_sequential(sequentially_trained_model)

                elif self.learning_approach == 'seqavg':
                    self.compute_collaborative_round_seqavg()

            # At the end of each epoch, evaluate the model for early stopping on a global validation set
            self.prepare_aggregation_weights()
            if self.learning_approach == 'seq-pure':
                model_to_evaluate = sequentially_trained_model
            elif self.learning_approach in ['fedavg', 'seq-with-final-agg', 'seqavg']:
                model_to_evaluate = self.build_model_from_weights(self.aggregate_model_weights())
            model_evaluation = model_to_evaluate.evaluate(
                x_val, y_val, batch_size=constants.DEFAULT_BATCH_SIZE, verbose=0,
            )

            current_val_loss = model_evaluation[0]
            current_val_metric = model_evaluation[1]

            self.score_matrix_collective_models[epoch_index, minibatch_count] = current_val_metric
            self.loss_collective_models.append(current_val_loss)

            logger.info(f"   Model evaluation at the end of the epoch: "
                        f"{['%.3f' % elem for elem in model_evaluation]}")

            logger.debug("      Checking if early stopping criteria are met:")
            if is_early_stopping:
                # Early stopping parameters
                if (
                        epoch_index >= constants.PATIENCE
                        and current_val_loss > self.loss_collective_models[-constants.PATIENCE]
                ):
                    logger.debug("         -> Early stopping criteria are met, stopping here.")
                    break
                else:
                    logger.debug("         -> Early stopping criteria are not met, continuing with training.")

        # After last epoch or if early stopping was triggered, evaluate model on the global testset
        logger.info("### Evaluating model on test data:")
        model_evaluation = model_to_evaluate.evaluate(
            x_test, y_test, batch_size=constants.DEFAULT_BATCH_SIZE, verbose=0)
        logger.info(f"   Model metrics names: {model_to_evaluate.metrics_names}")
        logger.info(f"   Model metrics values: {['%.3f' % elem for elem in model_evaluation]}")
        self.test_score = model_evaluation[1]  # 0 is for the loss
        self.nb_epochs_done = self.epoch_index + 1

        # Plot training history # TODO: move the data saving and plotting in dedicated functions
        if self.is_save_data:
            self.save_data()

        logger.info("Training and evaluation on multiple partners: done.")
        
        # saves the federated model weights
        self.federated_model_weights=model_evaluation.get_weights()
        
        end = timer()
        self.learning_computation_time = end - start

    def save_data(self):
        """Save figures, losses and metrics to disk"""

        history_data = {}
        history_data["loss_collective_models"] = self.loss_collective_models
        history_data["score_matrix_per_partner"] = self.score_matrix_per_partner
        history_data["score_matrix_collective_models"] = self.score_matrix_collective_models
        with open(self.save_folder / "history_data.p", 'wb') as f:
            pickle.dump(history_data, f)

        if not os.path.exists(self.save_folder / 'graphs/'):
            os.makedirs(self.save_folder / 'graphs/')
        plt.figure()
        plt.plot(self.loss_collective_models)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig(self.save_folder / "graphs/federated_training_loss.png")
        plt.close()

        plt.figure()
        plt.plot(self.score_matrix_collective_models[: self.epoch_index + 1, self.minibatch_count])
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        # plt.yscale('log')
        plt.ylim([0, 1])
        plt.savefig(self.save_folder / "graphs/federated_training_acc.png")
        plt.close()

        plt.figure()
        plt.plot(self.score_matrix_per_partner[: self.epoch_index + 1, self.minibatch_count - 1, ])
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["partner " + str(i) for i in range(self.partners_count)])
        # plt.yscale('log')
        plt.ylim([0, 1])
        plt.savefig(self.save_folder / "graphs/all_partners.png")
        plt.close()


    def compute_collaborative_round_fedavg(self):
        """Proceed to a collaborative round with a federated averaging approach"""

        logger.debug("Start new fedavg collaborative round ...")

        # Initialize variables
        epoch_index, minibatch_index = self.epoch_index, self.minibatch_index
        x_val, y_val = self.val_data

        # Starting model for each partner is the aggregated model  
        logger.debug(f"(fedavg) Minibatch n°{minibatch_index} of epoch n°{epoch_index}, "
                     f"init aggregated model for each partner with models from previous round")
        partners_model_list_for_iteration = self.init_with_agg_models()

        # Evaluate and store accuracy of mini-batch start model
        model_to_evaluate = partners_model_list_for_iteration[0]
        model_evaluation = model_to_evaluate.evaluate(x_val, y_val, batch_size=constants.DEFAULT_BATCH_SIZE, verbose=0)
        self.score_matrix_collective_models[epoch_index, minibatch_index] = model_evaluation[1]

        # Iterate over partners for training each individual model
        for partner_index, partner in enumerate(self.partners_list):

            # Reference the partner's model
            partner_model = partners_model_list_for_iteration[partner_index]

            # Train on partner local data set
            train_data_for_fit_iteration = (
                self.minibatched_x_train[partner_index][minibatch_index],
                self.minibatched_y_train[partner_index][minibatch_index],
            )
            history = self.collaborative_round_fit(
                partner_model, train_data_for_fit_iteration, self.val_data, partner.batch_size)

            # Log results of the round
            self.log_collaborative_round_partner_result(partner, partner_index, history.history["val_accuracy"][0])

            # Update the partner's model in the models' list
            self.models_weights_list[partner_index] = partner_model.get_weights()

            # Update iterative results
            self.update_iterative_results(partner_index, history)

        logger.debug("End of fedavg collaborative round.")

    def compute_collaborative_round_sequential(self, sequentially_trained_model):
        """Proceed to a collaborative round with a sequential approach"""

        logger.debug("Start new sequential collaborative round ...")

        # Initialize variables
        epoch_index, minibatch_index = self.epoch_index, self.minibatch_index
        is_last_round = minibatch_index == self.minibatch_count - 1
        x_val, y_val = self.val_data

        # Evaluate and store accuracy of mini-batch start model
        model_evaluation = sequentially_trained_model.evaluate(
            x_val, y_val, batch_size=constants.DEFAULT_BATCH_SIZE, verbose=0)
        self.score_matrix_collective_models[epoch_index, minibatch_index] = model_evaluation[1]

        # Iterate over partners for training the model sequentially
        shuffled_indexes = np.random.permutation(self.partners_count)
        logger.debug(f"(seq) Shuffled order for this sequential collaborative round: {shuffled_indexes}")
        for for_loop_idx, partner_index in enumerate(shuffled_indexes):

            partner = self.partners_list[partner_index]

            # Train on partner local data set
            train_data_for_fit_iteration = (
                self.minibatched_x_train[partner_index][minibatch_index],
                self.minibatched_y_train[partner_index][minibatch_index],
            )
            history = self.collaborative_round_fit(
                sequentially_trained_model, train_data_for_fit_iteration, self.val_data, partner.batch_size)

            # Log results of the round
            self.log_collaborative_round_partner_result(partner, for_loop_idx, history.history["val_accuracy"][0])

            # On final collaborative round, save the partner's model in the models' list
            if is_last_round:
                self.models_weights_list[partner_index] = sequentially_trained_model.get_weights()

            # Update iterative results
            self.update_iterative_results(partner_index, history)

        logger.debug("End of sequential collaborative round.")

    def compute_collaborative_round_seqavg(self):
        """Proceed to a collaborative round with a sequential averaging approach"""

        logger.debug("Start new seqavg collaborative round ...")

        # Initialize variables
        epoch_index, minibatch_index = self.epoch_index, self.minibatch_index
        x_val, y_val = self.val_data

        # Starting model for each partner is the aggregated model
        logger.debug(f"(seqavg) Minibatch n°{minibatch_index} of epoch n°{epoch_index}, "
                     f"init model by aggregating models from previous round")
        model_for_round = self.init_with_agg_model()

        # Evaluate and store accuracy of mini-batch start model
        model_evaluation = model_for_round.evaluate(x_val, y_val, batch_size=constants.DEFAULT_BATCH_SIZE, verbose=0)
        self.score_matrix_collective_models[epoch_index, minibatch_index] = model_evaluation[1]

        # Iterate over partners for training each individual model
        shuffled_indexes = np.random.permutation(self.partners_count)
        logger.debug(f"(seqavg) Shuffled order for this seqavg collaborative round: {shuffled_indexes}")
        for for_loop_idx, partner_index in enumerate(shuffled_indexes):

            partner = self.partners_list[partner_index]

            # Train on partner local data set
            train_data_for_fit_iteration = (
                self.minibatched_x_train[partner_index][minibatch_index],
                self.minibatched_y_train[partner_index][minibatch_index],
            )
            history = self.collaborative_round_fit(
                model_for_round, train_data_for_fit_iteration, self.val_data, partner.batch_size)

            # Log results
            self.log_collaborative_round_partner_result(partner, for_loop_idx, history.history["val_accuracy"][0])

            # Save the partner's model in the models' list
            self.models_weights_list[partner_index] = model_for_round.get_weights()

            # Update iterative results
            self.update_iterative_results(partner_index, history)

        logger.debug("End of seqavg collaborative round.")

    def split_in_minibatches(self):
        """Split the dataset passed as argument in mini-batches"""

        # Create the indices where to split
        split_indices = np.arange(1, self.minibatch_count + 1) / self.minibatch_count

        # Iterate over all partners and split their datasets
        for partner_index, partner in enumerate(self.partners_list):
            # Shuffle the dataset
            idx = np.random.permutation(len(partner.x_train))
            x_train, y_train = partner.x_train[idx], partner.y_train[idx]

            # Split the samples and labels
            self.minibatched_x_train[partner_index] = np.split(x_train, (split_indices[:-1] * len(x_train)).astype(int))
            self.minibatched_y_train[partner_index] = np.split(y_train, (split_indices[:-1] * len(y_train)).astype(int))

    def prepare_aggregation_weights(self):
        """Returns a list of weights for the weighted average aggregation of model weights"""

        if self.aggregation_weighting == "uniform":
            self.aggregation_weights = [1 / self.partners_count] * self.partners_count
        elif self.aggregation_weighting == "data_volume":
            partners_sizes = [len(partner.x_train) for partner in self.partners_list]
            self.aggregation_weights = partners_sizes / np.sum(partners_sizes)
        elif self.aggregation_weighting == "local_score":
            self.aggregation_weights = self.scores_last_learning_round / np.sum(self.scores_last_learning_round)
        else:
            raise NameError("The aggregation_weighting value [" + self.aggregation_weighting + "] is not recognized.")

    def aggregate_model_weights(self):
        """Aggregate model weights from the list of models, with a weighted average"""

        weights_per_layer = list(zip(*self.models_weights_list))
        new_weights = list()

        for weights_for_layer in weights_per_layer:
            avg_weights_for_layer = np.average(
                np.array(weights_for_layer), axis=0, weights=self.aggregation_weights
            )
            new_weights.append(list(avg_weights_for_layer))

        return new_weights

    def build_model_from_weights(self, new_weights):
        """Generate a new model initialized with weights passed as arguments"""

        new_model = self.generate_new_model()
        new_model.set_weights(new_weights)
        new_model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer="adam",
            metrics=["accuracy"],
        )

        return new_model


    def init_with_agg_model(self):
        """Return a new model aggregating models from model_list"""

        self.prepare_aggregation_weights()
        return self.build_model_from_weights(self.aggregate_model_weights())

    def init_with_agg_models(self):
        """Return a list with the aggregated model duplicated for each partner"""

        self.prepare_aggregation_weights()
        partners_model_list = []
        for partner in self.partners_list:
            partners_model_list.append(self.build_model_from_weights(self.aggregate_model_weights()))
        return partners_model_list

    @staticmethod
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

    def log_collaborative_round_partner_result(self, partner, partner_index, validation_score):
        """Print the validation accuracy of the collaborative round"""

        epoch_nb_str = f"Epoch {str(self.epoch_index).zfill(2)}/{str(self.epoch_count - 1).zfill(2)}"
        mb_nb_str = f"Minibatch {str(self.minibatch_index).zfill(2)}/{str(self.minibatch_count - 1).zfill(2)}"
        partner_id_str = f"Partner id #{partner.id} ({partner_index}/{self.partners_count - 1})"
        val_acc_str = f"{round(validation_score, 2)}"

        logger.debug(f"{epoch_nb_str} > {mb_nb_str} > {partner_id_str} > val_acc: {val_acc_str}")

    def update_iterative_results(self, partner_index, fit_history):
        """Update the results arrays with results from the collaboration round"""

        validation_score = fit_history.history["val_accuracy"][0]

        self.scores_last_learning_round[partner_index] = validation_score  # TO DO check if coherent

        # At the end of each mini-batch, for each partner, populate the score matrix per partner
        self.score_matrix_per_partner[self.epoch_index, self.minibatch_index, partner_index] = validation_score


def init_multi_partner_learning_from_scenario(scenario, is_save_data=True):

    mpl = MultiPartnerLearning(
        scenario.partners_list,
        scenario.epoch_count,
        scenario.minibatch_count,
        scenario.dataset,
        scenario.multi_partner_learning_approach,
        scenario.folder_of_starting_model,
        scenario.aggregation_weighting,
        scenario.is_early_stopping,
        is_save_data,
        scenario.save_folder,
    )

    return mpl
