# -*- coding: utf-8 -*-
"""
Functions for model training and evaluation (single-partner and multi-partner cases)
"""

import operator
import os
from abc import ABC, abstractmethod
from timeit import default_timer as timer

import numpy as np
from keras.backend import clear_session
from keras.callbacks import EarlyStopping
from loguru import logger
from sklearn.preprocessing import normalize

from . import constants
from .mpl_utils import History, UniformAggregator
from .partner import PartnerMpl


class MultiPartnerLearning(ABC):

    def __init__(self,
                 partners_list,
                 epoch_count,
                 minibatch_count,
                 dataset,
                 aggregation=UniformAggregator,
                 is_early_stopping=True,
                 is_save_data=False,
                 save_folder="",
                 init_model_from="random_initialization",
                 use_saved_weights=False,
                 ):

        # Attributes related to the data and the model
        self.val_data = (dataset.x_val, dataset.y_val)
        self.test_data = (dataset.x_test, dataset.y_test)
        self.dataset_name = dataset.name
        self.generate_new_model = dataset.generate_new_model
        self.init_model_from = init_model_from
        self.use_saved_weights = use_saved_weights

        # Initialize the model
        model = self.init_model()
        self.model_weights = model.get_weights()
        self.metrics_names = model.metrics_names

        # Attributes related to iterating at different levels
        self.epoch_count = epoch_count
        self.epoch_index = 0
        self.minibatch_count = minibatch_count
        self.minibatch_index = 0
        self.is_early_stopping = is_early_stopping

        # Attributes related to partners
        self.partners_list = [PartnerMpl(partner, self) for partner in partners_list]
        partners_list = sorted(partners_list, key=operator.attrgetter("id"))
        logger.info(
            f"## Preparation of model's training on partners with ids: {['#' + str(p.id) for p in partners_list]}")

        # Attributes related to the aggregation approach
        self.aggregator = aggregation(self)

        # Attributes to store results
        self.is_save_data = is_save_data
        self.save_folder = save_folder
        self.learning_computation_time = None
        self.history = History(self)

        logger.debug("MultiPartnerLearning object instantiated.")

    @property
    def partners_count(self):
        return len(self.partners_list)

    def build_model(self):
        return self.build_model_from_weights(self.model_weights)

    def save_final_model(self):
        """Save final model weights"""

        model_folder = os.path.join(self.save_folder, 'model')

        if not os.path.isdir(model_folder):
            os.makedirs(model_folder)

        np.save(os.path.join(model_folder, self.dataset_name + '_final_weights.npy'), self.model_weights)

        model_to_save = self.build_model()
        model_to_save.save_weights(os.path.join(model_folder, self.dataset_name + '_final_weights.h5'))

    def split_in_minibatches(self):
        """Split the dataset passed as argument in mini-batches"""

        for partner in self.partners_list:
            partner.split_minibatches()

    def build_model_from_weights(self, new_weights):
        """Generate a new model initialized with weights passed as arguments"""
        new_model = self.generate_new_model()
        new_model.set_weights(new_weights)
        return new_model

    def init_model(self):
        new_model = self.generate_new_model()

        if self.use_saved_weights:
            logger.info("Init model with previous coalition model")
            new_model.load_weights(self.init_model_from)
        else:
            logger.info("Init new model")

        return new_model

    def early_stop(self):
        logger.debug("      Checking if early stopping criteria are met:")
        if self.is_early_stopping:
            # Early stopping parameters
            if (
                    self.epoch_index >= constants.PATIENCE
                    and self.history.history['loss'][-1] > self.history.history['loss'][-constants.PATIENCE]
            ):
                logger.debug("         -> Early stopping criteria are met, stopping here.")
                return True
            else:
                logger.debug("         -> Early stopping criteria are not met, continuing with training.")
        else:
            return False

    def fit(self):
        """Return the score on test data of a final aggregated model trained in a federated way on each partner"""

        start = timer()
        # Train model (iterate for each epoch and mini-batch)
        while self.epoch_index < self.epoch_count:

            self.fit_epoch()  # perform an epoch on the self.model

            if self.early_stop():
                break
            self.epoch_index += 1

        # After last epoch or if early stopping was triggered, evaluate model on the global testset
        self.history.log_final_model_perf()
        # Save the model's weights
        self.save_final_model()

        end = timer()
        self.learning_computation_time = end - start
        logger.info(f"Training and evaluation on multiple partners: "
                    f"done. ({np.round(self.learning_computation_time, 3)} seconds)")

    @abstractmethod
    def fit_epoch(self):
        while self.minibatch_index < self.minibatch_count:
            self.fit_minibatch()
            self.minibatch_index += 1
            self.history.log_model_val_perf()

    @abstractmethod
    def fit_minibatch(self):
        pass


class SinglePartnerLearning(MultiPartnerLearning):
    def __init__(self,
                 partner,
                 epoch_count,
                 minibatch_count,
                 dataset,
                 is_early_stopping=True,
                 is_save_data=False,
                 save_folder="",
                 init_model_from="random_initialization",
                 use_saved_weights=False, ):
        if self.partners_count > 1:
            raise ValueError('More than one partner is provided')
        super(SinglePartnerLearning, self).__init__([partner],
                                                    epoch_count,
                                                    1,
                                                    dataset,
                                                    is_early_stopping=is_early_stopping,
                                                    is_save_data=is_save_data,
                                                    save_folder=save_folder,
                                                    init_model_from=init_model_from,
                                                    use_saved_weights=use_saved_weights,
                                                    )
        self.partner = partner

    def fit(self):
        """Return the score on test data of a model trained on a single partner"""

        start = timer()
        logger.info(f"## Training and evaluating model on partner with partner_id #{self.partner.id}")

        # Set if early stopping if needed
        cb = []
        es = None
        if self.is_early_stopping:
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=constants.PATIENCE)
            cb.append(es)

        # Train model
        logger.info("   Training model...")
        model = self.build_model()
        history = model.fit(self.partner.x_train,
                            self.partner.y_train,
                            batch_size=self.partner.batch_size,
                            epochs=self.epoch_count,
                            verbose=0,
                            validation_data=self.val_data)
        self.model_weights = model.get_weights()
        self.history.log_partner_perf(self.partner.id, 0, history.history)
        del self.history.history['model']
        # Evaluate trained model on test data
        self.history.log_final_model_perf()
        self.history.nb_epochs_done = (es.stopped_epoch + 1) if es.stopped_epoch != 0 else self.epoch_count

        end = timer()
        self.learning_computation_time = end - start

    def fit_epoch(self):
        pass

    def fit_minibatch(self):
        pass


class FederatedAverageLearning(MultiPartnerLearning):
    def __init__(self,
                 partners_list,
                 epoch_count,
                 minibatch_count,
                 dataset,
                 aggregation=UniformAggregator,
                 is_early_stopping=True,
                 is_save_data=False,
                 save_folder="",
                 init_model_from="random_initialization",
                 use_saved_weights=False,
                 ):
        # First, if only one partner, fall back to dedicated single partner function
        super(FederatedAverageLearning, self).__init__(partners_list,
                                                       epoch_count,
                                                       minibatch_count,
                                                       dataset,
                                                       aggregation,
                                                       is_early_stopping,
                                                       is_save_data,
                                                       save_folder,
                                                       init_model_from,
                                                       use_saved_weights)
        if self.partners_count == 1:
            raise ValueError('Only one partner is provided. Please use the dedicated SinglePartnerLearning class')

    def fit_epoch(self):
        # Clear Keras' old models
        clear_session()

        # Split the train dataset in mini-batches
        self.split_in_minibatches()

        # Iterate over mini-batches and train
        for i in range(self.minibatch_count):
            self.minibatch_index = i
            self.fit_minibatch()

            # At the end of each minibatch,aggregate the models

            self.model_weights = self.aggregator.aggregate_model_weights()
        self.minibatch_index = 0

    def fit_minibatch(self):
        """Proceed to a collaborative round with a federated averaging approach"""

        logger.debug("Start new fedavg collaborative round ...")

        # Starting model for each partner is the aggregated model from the previous mini-batch iteration
        logger.info(f"(fedavg) Minibatch n°{self.minibatch_index} of epoch n°{self.epoch_index}, "
                    f"init each partner's models with a copy of the global model")

        for partner in self.partners_list:
            partner.model_weights = self.model_weights

        # Evaluate and store accuracy of mini-batch start model
        self.history.log_model_val_perf()

        # Iterate over partners for training each individual model
        for partner_index, partner in enumerate(self.partners_list):
            # Reference the partner's model
            partner_model = partner.build_model()

            # Train on partner local data set
            history = partner_model.fit(partner.minibatched_x_train[self.minibatch_index],
                                        partner.minibatched_y_train[self.minibatch_index],
                                        batch_size=partner.batch_size,
                                        verbose=0,
                                        validation_data=self.val_data)

            # Log results of the round
            self.history.log_partner_perf(partner.id, partner_index, history.history)

            # Update the partner's model in the models' list
            partner.model_weights = partner_model.get_weights()

        logger.debug("End of fedavg collaborative round.")


class SequentialLearning(MultiPartnerLearning):  # seq-pure
    def __init__(self,
                 partners_list,
                 epoch_count,
                 minibatch_count,
                 dataset,
                 aggregation=UniformAggregator,
                 is_early_stopping=True,
                 is_save_data=False,
                 save_folder="",
                 init_model_from="random_initialization",
                 use_saved_weights=False,
                 ):
        super(SequentialLearning, self).__init__(partners_list,
                                                 epoch_count,
                                                 minibatch_count,
                                                 dataset,
                                                 aggregation,
                                                 is_early_stopping,
                                                 is_save_data,
                                                 save_folder,
                                                 init_model_from,
                                                 use_saved_weights)
        if self.partners_count == 1:
            raise ValueError('Only one partner is provided. Please use the dedicated SinglePartnerLearning class')

    def fit_epoch(self):
        # Clear Keras' old models
        clear_session()

        # Split the train dataset in mini-batches
        self.split_in_minibatches()

        # Iterate over mini-batches and train
        for i in range(self.minibatch_count):
            self.minibatch_index = i
            logger.info(f"(seq-pure) Minibatch n°{self.minibatch_index} of epoch n°{self.epoch_index}")
            self.fit_minibatch()

    def fit_minibatch(self):
        """Proceed to a collaborative round with a sequential averaging approach"""

        logger.debug("Start new seq collaborative round ...")

        # Starting model for each partner is the aggregated model from the previous mini-batch iteration

        model_for_round = self.build_model()

        # Evaluate and store accuracy of mini-batch start model
        self.history.log_model_val_perf()
        # Iterate over partners for training each individual model
        shuffled_indexes = np.random.permutation(self.partners_count)
        logger.debug(f"(seq) Shuffled order for this seqavg collaborative round: {shuffled_indexes}")
        for idx, partner_index in enumerate(shuffled_indexes):
            partner = self.partners_list[partner_index]

            # Train on partner local data set
            history = model_for_round.fit(partner.minibatched_y_train[self.minibatch_index],
                                          partner.minibatched_y_train[self.minibatch_index],
                                          batch_size=partner.batch_size,
                                          verbose=0,
                                          validation_data=self.val_data)

            # Log results
            self.history.log_partner_perf(partner.id, idx, history.history)

            # Save the partner's model in the models' list
            partner.model_weights = model_for_round.get_weights()
            self.model_weights = model_for_round.get_weights()

        logger.debug("End of seq collaborative round.")


class SequentialWithFinalAggLearning(SequentialLearning):
    def __init__(self,
                 partners_list,
                 epoch_count,
                 minibatch_count,
                 dataset,
                 aggregation=UniformAggregator,
                 is_early_stopping=True,
                 is_save_data=False,
                 save_folder="",
                 init_model_from="random_initialization",
                 use_saved_weights=False,
                 ):
        super(SequentialWithFinalAggLearning, self).__init__(partners_list,
                                                             epoch_count,
                                                             minibatch_count,
                                                             dataset,
                                                             aggregation,
                                                             is_early_stopping,
                                                             is_save_data,
                                                             save_folder,
                                                             init_model_from,
                                                             use_saved_weights)
        if self.partners_count == 1:
            raise ValueError('Only one partner is provided. Please use the dedicated SinglePartnerLearning class')

    def fit_epoch(self):
        # Clear Keras' old models
        clear_session()

        # Split the train dataset in mini-batches
        self.split_in_minibatches()

        # Iterate over mini-batches and train
        for i in range(self.minibatch_count):
            logger.info(f"(seq-final-agg) Minibatch n°{self.minibatch_index} of epoch n°{self.epoch_index}, "
                        f"init model with a copy of the global model")
            self.minibatch_index = i
            self.fit_minibatch()

        # At the end of each epoch, aggregate the models
        self.model_weights = self.aggregator.aggregate_model_weights()


class SequentialAverageLearning(SequentialLearning):
    def __init__(self,
                 partners_list,
                 epoch_count,
                 minibatch_count,
                 dataset,
                 aggregation=UniformAggregator,
                 is_early_stopping=True,
                 is_save_data=False,
                 save_folder="",
                 init_model_from="random_initialization",
                 use_saved_weights=False,
                 ):
        super(SequentialAverageLearning, self).__init__(partners_list,
                                                        epoch_count,
                                                        minibatch_count,
                                                        dataset,
                                                        aggregation,
                                                        is_early_stopping,
                                                        is_save_data,
                                                        save_folder,
                                                        init_model_from,
                                                        use_saved_weights)
        if self.partners_count == 1:
            raise ValueError('Only one partner is provided. Please use the dedicated SinglePartnerLearning class')

    def fit_epoch(self):
        # Clear Keras' old models
        clear_session()

        # Split the train dataset in mini-batches
        self.split_in_minibatches()

        # Iterate over mini-batches and train
        for i in range(self.minibatch_count):
            logger.info(f"(seqavg) Minibatch n°{self.minibatch_index} of epoch n°{self.epoch_index}, "
                        f"init model with a copy of the global model")
            self.minibatch_index = i
            self.fit_minibatch()

            # At the end of each minibatch, aggregate the models
            self.model_weights = self.aggregator.aggregate_model_weights()


def init_multi_partner_learning_from_scenario(scenario, is_save_data=True):
    mpl = scenario.multi_partner_learning_approach(
        scenario.partners_list,
        scenario.epoch_count,
        scenario.minibatch_count,
        scenario.dataset,
        scenario.aggregation,
        scenario.is_early_stopping,
        is_save_data,
        scenario.save_folder,
        scenario.init_model_from,
        scenario.use_saved_weights,
    )

    return mpl


class MplLabelFlip(MultiPartnerLearning):

    def __init__(self, scenario, is_save_data=False, epsilon=0.01):
        super().__init__(
            scenario.partners_list,
            scenario.epoch_count,
            scenario.minibatch_count,
            scenario.dataset,
            scenario.multi_partner_learning_approach,
            scenario.aggregation_weighting,
            scenario.is_early_stopping,
            is_save_data,
            scenario.save_folder,
            scenario.init_model_from,
            scenario.use_saved_weights,
        )

        self.epsilon = epsilon
        self.K = scenario.dataset.num_classes
        self.history_theta = [[None for _ in self.partners_list] for _ in range(self.epoch_count)]
        self.history_theta_ = [[None for _ in self.partners_list] for _ in range(self.epoch_count)]
        self.theta = [self.init_flip_proba() for _ in self.partners_list]
        self.theta_ = [None for _ in self.partners_list]

    def init_flip_proba(self):
        identity = np.identity(self.K)
        return identity * (1 - self.epsilon) + (1 - identity) * (self.epsilon / (self.K - 1))

    def fit(self):
        start = timer()
        logger.info(
            f"## Training and evaluating model on partners with ids: {['#' + str(p.id) for p in self.partners_list]}")

        logger.info("(LFlip) Very first minibatch, init new models for each partner")
        partners_models = self.init_with_models()

        self.epoch_index = 0
        while self.epoch_index < self.epoch_count:
            self.split_in_minibatches()
            self.minibatch_index = 0
            while self.minibatch_index < self.minibatch_count:
                logger.info(f"(LFLip) Minibatch n°{self.minibatch_index} of epoch n°{self.epoch_index}, "
                            f"init aggregated model for each partner with models from previous round")

                # Evaluate and store accuracy of mini-batch start model
                model_to_evaluate = partners_models[0]
                model_evaluation_val_data = self.evaluate_model(model_to_evaluate, self.val_data)
                self.score_matrix_collective_models[self.epoch_index,
                                                    self.minibatch_index] = model_evaluation_val_data[1]

                for partner_index, partner in enumerate(self.partners_list):
                    # Reference the partner's model
                    partner_model = partners_models[partner_index]
                    x_batch = self.minibatched_x_train[partner_index][self.minibatch_index]
                    y_batch = self.minibatched_y_train[partner_index][self.minibatch_index]

                    predictions = partner_model.predict(x_batch)
                    self.theta_[partner_index] = predictions  # Initialize the theta_

                    for idx, y in enumerate(y_batch):
                        self.theta_[partner_index][idx, :] *= self.theta[partner_index][:, np.argmax(y)]
                    self.theta_[partner_index] = normalize(self.theta_[partner_index], axis=0, norm='l1')
                    self.history_theta_[self.epoch_index][partner_index] = self.theta_[partner_index]

                    self.theta[partner_index] = self.theta_[partner_index].T.dot(y_batch)
                    self.theta[partner_index] = normalize(self.theta[partner_index], axis=1, norm='l1')
                    self.history_theta[self.epoch_index][partner_index] = self.theta[partner_index]

                    self.theta_[partner_index] = predictions
                    for idx, y in enumerate(y_batch):
                        self.theta_[partner_index][idx, :] *= self.theta[partner_index][:, np.argmax(y)]
                    self.theta_[partner_index] = normalize(self.theta_[partner_index], axis=0, norm='l1')

                    # draw of x_i
                    rand_idx = np.arange(len(x_batch))
                    # rand_idx =  np.random.randint(low=0, high=len(x_batch), size=(len(x_batch)))
                    flipped_minibatch_x_train = x_batch
                    flipped_minibatch_y_train = np.zeros(y_batch.shape)
                    for i, idx in enumerate(rand_idx):  # TODO vectorize
                        repartition = np.cumsum(
                            self.theta_[partner_index][idx, :])
                        a = np.random.random() - repartition  # draw
                        flipped_minibatch_y_train[i][np.argmin(np.where(a > 0, a, 0))] = 1
                        # not responsive to labels type.

                    train_data_for_fit_iteration = flipped_minibatch_x_train, flipped_minibatch_y_train

                    history = self.fit_model(
                        partner_model, train_data_for_fit_iteration, self.val_data, partner.batch_size)

                    # Log results of the round
                    model_evaluation_val_data = history.history['val_accuracy'][0]
                    self.log_collaborative_round_partner_result(partner, partner_index, model_evaluation_val_data)

                    # Update the partner's model in the models' list
                    self.save_model_for_partner(partner_model, partner_index)

                    # Update iterative results
                    self.update_iterative_results(partner_index, history)

                partners_models = self.init_with_agg_models()
                self.minibatch_index += 1

            # At the end of each epoch, evaluate the model for early stopping on a global validation set
            model_to_evaluate = partners_models[0]
            model_evaluation_val_data = self.evaluate_model(model_to_evaluate, self.val_data)

            current_val_loss = model_evaluation_val_data[0]
            current_val_metric = model_evaluation_val_data[1]

            self.score_matrix_collective_models[self.epoch_index, self.minibatch_count] = current_val_metric
            self.loss_collective_models.append(current_val_loss)

            logger.info(f"   Model evaluation at the end of the epoch: "
                        f"{['%.3f' % elem for elem in model_evaluation_val_data]}")

            self.epoch_index += 1

            logger.debug("      Checking if early stopping criteria are met:")
            if self.is_early_stopping:
                # Early stopping parameters
                if (
                        self.epoch_index >= constants.PATIENCE
                        and current_val_loss > self.loss_collective_models[-constants.PATIENCE]
                ):
                    logger.debug("         -> Early stopping criteria are met, stopping here.")
                    break
                else:
                    logger.debug("         -> Early stopping criteria are not met, continuing with training.")

        logger.info("### Evaluating model on test data:")
        model_evaluation_test_data = self.evaluate_model(model_to_evaluate, self.test_data)
        logger.info(f"   Model metrics names: {model_to_evaluate.metrics_names}")
        logger.info(f"   Model metrics values: {['%.3f' % elem for elem in model_evaluation_test_data]}")
        self.test_score = model_evaluation_test_data[1]  # 0 is for the loss
        self.nb_epochs_done = self.epoch_index

        # Plot training history
        if self.is_save_data:
            self.save_data()

        self.save_final_model_weights(model_to_evaluate)

        logger.info("Training and evaluation on multiple partners: done.")
        end = timer()
        self.learning_computation_time = end - start
