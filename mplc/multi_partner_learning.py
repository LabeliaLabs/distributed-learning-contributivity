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
from .mpl_utils import History, Aggregator
from .partner import Partner, PartnerMpl

ALLOWED_PARAMETERS = ('partners_list',
                      'epoch_count',
                      'minibatch_count',
                      'dataset',
                      'aggregation_method',
                      'is_early_stopping',
                      'is_save_data',
                      'save_folder',
                      'init_model_from',
                      'use_saved_weights')


class MultiPartnerLearning(ABC):
    def __init__(self, scenario, **kwargs):
        """

        :type scenario: Scenario
        """
        # Attributes related to the data and the model
        self.dataset = scenario.dataset
        self.partners_list = scenario.partners_list
        self.init_model_from = scenario.init_model_from
        self.use_saved_weights = scenario.use_saved_weights

        # Attributes related to iterating at different levels
        self.epoch_count = scenario.epoch_count
        self.minibatch_count = scenario.minibatch_count
        self.is_early_stopping = scenario.is_early_stopping

        # Attributes related to the aggregation approach
        self.aggregation_method = scenario.aggregation

        # Attributes to store results
        self.is_save_data = False  # default value
        self.save_folder = scenario.save_folder

        # Erase the default parameters (which mostly come from the scenario) if some parameters have been specified
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in ALLOWED_PARAMETERS)

        # Unpack dataset-related parameters
        self.val_data = (self.dataset.x_val, self.dataset.y_val)
        self.test_data = (self.dataset.x_test, self.dataset.y_test)
        self.dataset_name = self.dataset.name
        self.generate_new_model = self.dataset.generate_new_model

        # Initialize the model
        model = self.init_model()
        self.model_weights = model.get_weights()
        self.metrics_names = model.metrics_names

        # Initialize iterators
        self.epoch_index = 0
        self.minibatch_index = 0
        self.learning_computation_time = 0

        # Convert partners to Mpl partners
        for partner in self.partners_list:
            assert isinstance(partner, Partner)
        partners_list = sorted(self.partners_list, key=operator.attrgetter("id"))
        logger.info(
            f"## Preparation of model's training on partners with ids: {['#' + str(p.id) for p in partners_list]}")
        self.partners_list = [PartnerMpl(partner, self) for partner in self.partners_list]

        # Initialize aggregator
        self.aggregator = self.aggregation_method(self)
        assert isinstance(self.aggregator, Aggregator)

        # Initialize History
        self.history = History(self)

        logger.debug("MultiPartnerLearning object instantiated.")

    @property
    def partners_count(self):
        return len(self.partners_list)

    def build_model(self):
        return self.build_model_from_weights(self.model_weights)

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

    def save_final_model(self):
        """Save final model weights"""

        model_folder = os.path.join(self.save_folder, 'model')

        if not os.path.isdir(model_folder):
            os.makedirs(model_folder)

        np.save(os.path.join(model_folder, self.dataset_name + '_final_weights.npy'), self.model_weights)

        model_to_save = self.build_model()
        model_to_save.save_weights(os.path.join(model_folder, self.dataset_name + '_final_weights.h5'))

    def log_partner_perf(self, partner_id, partner_index, history):
        for key_history in self.history.metrics:
            self.history.history[partner_id][key_history][self.epoch_index,
                                                          self.minibatch_index] = history[key_history][-1]

        epoch_nb_str = f"Epoch {str(self.epoch_index).zfill(2)}/{str(self.epoch_count - 1).zfill(2)}"
        mb_nb_str = f"Minibatch {str(self.minibatch_index).zfill(2)}/{str(self.minibatch_count - 1).zfill(2)}"
        partner_id_str = f"Partner partner_id #{partner_id} ({partner_index}/{self.partners_count - 1})"
        val_acc_str = f"{round(history['val_accuracy'][-1], 2)}"

        logger.debug(f"{epoch_nb_str} > {mb_nb_str} > {partner_id_str} > val_acc: {val_acc_str}")

    def eval_and_log_model_val_perf(self):
        model = self.build_model()
        hist = model.evaluate(self.val_data[0],
                              self.val_data[1],
                              batch_size=constants.DEFAULT_BATCH_SIZE,
                              verbose=0,
                              )
        self.history.history['mpl_model']['val_loss'][self.epoch_index, self.minibatch_index] = hist[0]
        self.history.history['mpl_model']['val_accuracy'][self.epoch_index, self.minibatch_index] = hist[1]

        if self.minibatch_index >= self.minibatch_count - 1:
            epoch_nb_str = f"{str(self.epoch_index).zfill(2)}/{str(self.epoch_count - 1).zfill(2)}"
            logger.info(f"   Model evaluation at the end of the epoch "
                        f"{epoch_nb_str}: "
                        f"{['%.3f' % elem for elem in hist]}")

    def eval_and_log_final_model__test_perf(self):
        logger.info("### Evaluating model on test data:")
        model = self.build_model()
        hist = model.evaluate(self.test_data[0],
                              self.test_data[1],
                              batch_size=constants.DEFAULT_BATCH_SIZE,
                              verbose=0,
                              )
        self.history.score = hist[1]
        self.history.nb_epochs_done = self.epoch_index + 1
        logger.info(f"   Model metrics names: {self.metrics_names}")
        logger.info(f"   Model metrics values: {['%.3f' % elem for elem in hist]}")

    def split_in_minibatches(self):
        """Split the dataset passed as argument in mini-batches"""

        for partner in self.partners_list:
            partner.split_minibatches()

    def early_stop(self):
        logger.debug("      Checking if early stopping criteria are met:")
        if self.is_early_stopping:
            # Early stopping parameters
            if (
                    self.epoch_index >= constants.PATIENCE
                    and self.history.history['mpl_model']['val_loss'][self.epoch_index,
                                                                      self.minibatch_index] >
                    self.history.history['mpl_model']['val_loss'][self.epoch_index-constants.PATIENCE,
                                                                  self.minibatch_index]
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
        self.eval_and_log_final_model__test_perf()
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
            self.eval_and_log_model_val_perf()

    @abstractmethod
    def fit_minibatch(self):
        pass


class SinglePartnerLearning(MultiPartnerLearning):
    def __init__(self, scenario, partner, **kwargs):
        kwargs['partners_list'] = [partner]
        super(SinglePartnerLearning, self).__init__(scenario, **kwargs)
        if type(partner) == list:
            raise ValueError('More than one partner is provided')
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
                            validation_data=self.val_data,
                            callbacks=cb)
        self.model_weights = model.get_weights()
        self.log_partner_perf(self.partner.id, 0, history.history)
        del self.history.history['mpl_model']
        # Evaluate trained model on test data
        self.eval_and_log_final_model__test_perf()
        self.history.nb_epochs_done = (es.stopped_epoch + 1) if es.stopped_epoch != 0 else self.epoch_count

        end = timer()
        self.learning_computation_time = end - start

    def fit_epoch(self):
        pass

    def fit_minibatch(self):
        pass


class FederatedAverageLearning(MultiPartnerLearning):
    def __init__(self, scenario, **kwargs):
        # First, if only one partner, fall back to dedicated single partner function
        super(FederatedAverageLearning, self).__init__(scenario, **kwargs)
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
        self.eval_and_log_model_val_perf()

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
            self.log_partner_perf(partner.id, partner_index, history.history)

            # Update the partner's model in the models' list
            partner.model_weights = partner_model.get_weights()

        logger.debug("End of fedavg collaborative round.")


class SequentialLearning(MultiPartnerLearning):  # seq-pure
    def __init__(self, scenario, **kwargs):
        super(SequentialLearning, self).__init__(scenario, **kwargs)
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

        model_for_round = self.build_model()

        # Evaluate and store accuracy of mini-batch start model
        self.eval_and_log_model_val_perf()
        # Iterate over partners for training each individual model
        shuffled_indexes = np.random.permutation(self.partners_count)
        logger.debug(f"(seq) Shuffled order for this seqavg collaborative round: {shuffled_indexes}")
        for idx, partner_index in enumerate(shuffled_indexes):
            partner = self.partners_list[partner_index]

            # Train on partner local data set
            history = model_for_round.fit(partner.minibatched_x_train[self.minibatch_index],
                                          partner.minibatched_y_train[self.minibatch_index],
                                          batch_size=partner.batch_size,
                                          verbose=0,
                                          validation_data=self.val_data)

            # Log results
            self.log_partner_perf(partner.id, idx, history.history)

            # Save the partner's model in the models' list
            partner.model_weights = model_for_round.get_weights()
            self.model_weights = model_for_round.get_weights()

        logger.debug("End of seq collaborative round.")


class SequentialWithFinalAggLearning(SequentialLearning):
    def __init__(self, scenario, **kwargs):
        super(SequentialWithFinalAggLearning, self).__init__(scenario, **kwargs)
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
    def __init__(self, scenario, **kwargs):
        super(SequentialAverageLearning, self).__init__(scenario, **kwargs)
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


class MplLabelFlip(FederatedAverageLearning):
    def __init__(self, scenario, epsilon=0.01, **kwargs):
        super(MplLabelFlip, self).__init__(scenario, **kwargs)

        self.epsilon = epsilon
        self.K = self.dataset.num_classes
        self.history.theta = [[None for _ in self.partners_list] for _ in range(self.epoch_count)]
        self.history.theta_ = [[None for _ in self.partners_list] for _ in range(self.epoch_count)]
        for partner in self.partners_list:
            partner.theta = self.init_flip_proba()
            partner.theta_ = None

    def init_flip_proba(self):
        identity = np.identity(self.K)
        return identity * (1 - self.epsilon) + (1 - identity) * (self.epsilon / (self.K - 1))

    def fit_minibatch(self):
        """Proceed to a collaborative round with a label flipped federated averaging approach"""

        logger.debug("Start new LFlip collaborative round ...")

        # Starting model for each partner is the aggregated model from the previous mini-batch iteration
        logger.info(f"(LFlip) Minibatch n°{self.minibatch_index} of epoch n°{self.epoch_index}, "
                    f"init each partner's models with a copy of the global model")

        for partner in self.partners_list:
            partner.model_weights = self.model_weights

        # Evaluate and store accuracy of mini-batch start model
        self.eval_and_log_model_val_perf()

        # Iterate over partners for training each individual model
        for partner_index, partner in enumerate(self.partners_list):
            # Reference the partner's model
            partner_model = partner.build_model()

            x_batch = partner.minibatched_x_train[self.minibatch_index]
            y_batch = partner.minibatched_y_train[self.minibatch_index]

            predictions = partner_model.predict(x_batch)
            partner.theta_ = predictions  # Initialize the theta_

            for idx, y in enumerate(y_batch):
                partner.theta_[idx, :] *= partner.theta[:, np.argmax(y)]
            partner.theta_ = normalize(partner.theta_, axis=0, norm='l1')
            self.history.theta_[self.epoch_index][partner_index] = partner.theta_

            partner.theta = partner.theta_.T.dot(y_batch)
            partner.theta = normalize(partner.theta, axis=1, norm='l1')
            self.history.theta[self.epoch_index][partner_index] = partner.theta

            partner.theta_ = predictions
            for idx, y in enumerate(y_batch):
                partner.theta_[idx, :] *= partner.theta[:, np.argmax(y)]
            partner.theta_ = normalize(partner.theta_, axis=0, norm='l1')

            # draw of x_i
            rand_idx = np.arange(len(x_batch))
            # rand_idx =  np.random.randint(low=0, high=len(x_batch), size=(len(x_batch)))
            flipped_minibatch_x_train = x_batch
            flipped_minibatch_y_train = np.zeros(y_batch.shape)
            for i, idx in enumerate(rand_idx):  # TODO vectorize
                repartition = np.cumsum(
                    partner.theta_[idx, :])
                a = np.random.random() - repartition  # draw
                flipped_minibatch_y_train[i][np.argmin(np.where(a > 0, a, 0))] = 1
                # not responsive to labels type.
            # Train on partner local data set
            history = partner_model.fit(flipped_minibatch_x_train,
                                        flipped_minibatch_y_train,
                                        batch_size=partner.batch_size,
                                        verbose=0,
                                        validation_data=self.val_data)

            # Log results of the round
            self.log_partner_perf(partner.id, partner_index, history.history)

            # Update the partner's model in the models' list
            partner.model_weights = partner_model.get_weights()

        logger.debug("End of LFlip collaborative round.")


# Supported multi-partner learning approaches

MULTI_PARTNER_LEARNING_APPROACHES = {
    "fedavg": FederatedAverageLearning,
    "seq-pure": SequentialLearning,
    "seq-with-final-agg": SequentialWithFinalAggLearning,
    "seqavg": SequentialAverageLearning,
    "lflip": MplLabelFlip
}
