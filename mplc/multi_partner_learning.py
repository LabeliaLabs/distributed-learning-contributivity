# -*- coding: utf-8 -*-
"""
Functions for model training and evaluation (single-partner and multi-partner cases)
"""

import operator
import os
import time
from abc import ABC, abstractmethod
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from loguru import logger
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Input, Model
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping

from . import constants
from .models import NoiseAdaptationChannel
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
    name = 'abstract'

    def __init__(self, scenario, **kwargs):
        """

        :type scenario: Scenario
        """
        # Attributes related to the data and the model
        self.dataset = scenario.dataset
        self.partners_list = scenario.partners_list
        self.init_model_from = scenario.init_model_from
        self.use_saved_weights = scenario.use_saved_weights
        self.val_set = scenario.val_set
        self.test_set = scenario.test_set

        # Attributes related to iterating at different levels
        self.epoch_count = scenario.epoch_count
        self.minibatch_count = scenario.minibatch_count
        self.is_early_stopping = scenario.is_early_stopping

        # Attributes related to the _aggregation_weighting approach
        self.aggregation_method = scenario._aggregation_weighting

        # Attributes to store results
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
        self.metrics_names = self.dataset.model_metrics_names

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

        # Initialize result folder
        if self.save_folder is not None:
            if 'custom_name' in kwargs:
                self.save_folder = self.save_folder / kwargs["custom_name"]
            else:
                self.save_folder = self.save_folder / 'mpl'
            self.save_folder.mkdir()

        logger.debug("MultiPartnerLearning object instantiated.")

    def __str__(self):
        return f'{self.name}'

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

    def save_data(self):
        if self.save_folder is None:
            raise ValueError("The path to the save folder is None, history data cannot be saved, nor model weights")

        self.save_final_model()
        self.history.save_data()

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
        if self.val_set == 'global':
            hist = model.evaluate(self.val_data[0],
                                  self.val_data[1],
                                  batch_size=constants.DEFAULT_BATCH_SIZE,
                                  verbose=0,
                                  )
        elif self.val_set == 'local':
            hist = [0.0, 0.0]
            for p in self.partners_list:
                hist_partner = model.evaluate(p.x_val,
                                              p.y_val,
                                              batch_size=constants.DEFAULT_BATCH_SIZE,
                                              verbose=0,
                                              )
                hist[0] += hist_partner[0] / self.partners_count
                hist[1] += hist_partner[1] / self.partners_count
        else:
            raise ValueError("validation set should be 'local' or 'global', not {self.val_set}")

        self.history.history['mpl_model']['val_loss'][self.epoch_index, self.minibatch_index] = hist[0]
        self.history.history['mpl_model']['val_accuracy'][self.epoch_index, self.minibatch_index] = hist[1]

        if self.minibatch_index >= self.minibatch_count - 1:
            epoch_nb_str = f"{str(self.epoch_index).zfill(2)}/{str(self.epoch_count - 1).zfill(2)}"
            logger.info(f"   Model evaluation at the end of the epoch "
                        f"{epoch_nb_str}: "
                        f"{['%.3f' % elem for elem in hist]}")

    def eval_and_log_final_model_test_perf(self):
        logger.info("### Evaluating model on test data:")
        model = self.build_model()
        if self.test_set == 'global':
            hist = model.evaluate(self.test_data[0],
                                  self.test_data[1],
                                  batch_size=constants.DEFAULT_BATCH_SIZE,
                                  verbose=0,
                                  )
        elif self.test_set == 'local':
            hist = [0.0, 0.0]
            for p in self.partners_list:
                hist_partner = model.evaluate(p.x_test,
                                              p.y_test,
                                              batch_size=constants.DEFAULT_BATCH_SIZE,
                                              verbose=0,
                                              )
                hist[0] += hist_partner[0] / self.partners_count
                hist[1] += hist_partner[1] / self.partners_count
        else:
            raise ValueError("test set should be 'local' or 'global', not {self.val_set}")

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
                    self.history.history['mpl_model']['val_loss'][self.epoch_index - constants.PATIENCE,
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
        self.eval_and_log_final_model_test_perf()

        end = timer()
        self.learning_computation_time = end - start
        logger.info(f"Training and evaluation on multiple partners: "
                    f"done. ({np.round(self.learning_computation_time, 3)} seconds)")
        if self.save_folder is not None:
            self.save_data()  # Save the model weights and the history data

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
    name = 'Single partner learning'

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
        if self.val_set == 'global':
            history = model.fit(self.partner.x_train,
                                self.partner.y_train,
                                batch_size=self.partner.batch_size,
                                epochs=self.epoch_count,
                                verbose=0,
                                validation_data=self.val_data,
                                callbacks=cb)
        elif self.val_set == 'local':
            history = model.fit(self.partner.x_train,
                                self.partner.y_train,
                                batch_size=self.partner.batch_size,
                                epochs=self.epoch_count,
                                verbose=0,
                                validation_data=(self.partner.x_val, self.partner.y_val),
                                callbacks=cb)
        else:
            raise ValueError("validation set should be 'local' or 'global', not {self.val_set}")

        self.model_weights = model.get_weights()
        self.log_partner_perf(self.partner.id, 0, history.history)
        del self.history.history['mpl_model']
        # Evaluate trained model on test data
        self.eval_and_log_final_model_test_perf()
        self.history.nb_epochs_done = (es.stopped_epoch + 1) if es.stopped_epoch != 0 else self.epoch_count

        end = timer()
        self.learning_computation_time = end - start

    def fit_epoch(self):
        pass

    def fit_minibatch(self):
        pass


class FederatedAverageLearning(MultiPartnerLearning):
    name = 'Federated averaging'

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
            if self.val_set == 'global':
                history = partner_model.fit(partner.minibatched_x_train[self.minibatch_index],
                                            partner.minibatched_y_train[self.minibatch_index],
                                            batch_size=partner.batch_size,
                                            verbose=0,
                                            validation_data=self.val_data)
            elif self.val_set == 'local':
                history = partner_model.fit(partner.minibatched_x_train[self.minibatch_index],
                                            partner.minibatched_y_train[self.minibatch_index],
                                            batch_size=partner.batch_size,
                                            verbose=0,
                                            validation_data=(partner.x_val, partner.y_val))
            else:
                raise ValueError("validation set should be 'local' or 'global', not {self.val_set}")

            # Log results of the round
            self.log_partner_perf(partner.id, partner_index, history.history)

            # Update the partner's model in the models' list
            partner.model_weights = partner_model.get_weights()

        logger.debug("End of fedavg collaborative round.")


class SequentialLearning(MultiPartnerLearning):  # seq-pure
    name = 'Sequential learning'

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
            if self.val_set == 'global':
                history = model_for_round.fit(partner.minibatched_x_train[self.minibatch_index],
                                              partner.minibatched_y_train[self.minibatch_index],
                                              batch_size=partner.batch_size,
                                              verbose=0,
                                              validation_data=self.val_data)
            elif self.val_set == 'local':
                history = model_for_round.fit(partner.minibatched_x_train[self.minibatch_index],
                                              partner.minibatched_y_train[self.minibatch_index],
                                              batch_size=partner.batch_size,
                                              verbose=0,
                                              validation_data=(partner.x_val, partner.y_val))
            else:
                raise ValueError("validation set should be 'local' or 'global', not {self.val_set}")

            # Log results
            self.log_partner_perf(partner.id, idx, history.history)

            # Save the partner's model in the models' list
            partner.model_weights = model_for_round.get_weights()
            self.model_weights = model_for_round.get_weights()

        logger.debug("End of seq collaborative round.")


class SequentialWithFinalAggLearning(SequentialLearning):
    name = 'Sequential learning with final aggregation'

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
    name = 'Sequential averaged learning'

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


class MplSModel(FederatedAverageLearning):
    name = 'Federated learning with label flipping'

    def __init__(self, scenario, pretrain_epochs=0, epsilon=0.5, **kwargs):
        super(MplSModel, self).__init__(scenario, **kwargs)
        self.pretrain_epochs = pretrain_epochs
        self.epsilon = epsilon
        if pretrain_epochs > 0:
            self.pretrain_mpl = FederatedAverageLearning(scenario=scenario,
                                                         epoch_count=self.pretrain_epochs,
                                                         is_save_data=False)

    def fit(self):
        if self.pretrain_epochs > 0:
            logger.info('Start pre-train...')
            self.pretrain_mpl.fit()
            pretrain_model = self.pretrain_mpl.build_model()
            for p in self.partners_list:
                confusion = confusion_matrix(np.argmax(p.y_train, axis=1),
                                             np.argmax(pretrain_model.predict(p.x_train), axis=1),
                                             normalize='pred')
                p.noise_layer_weights = [np.log(confusion.T + 1e-8)]
            self.model_weights[:-1] = self.pretrain_mpl.model_weights[:-1]
        else:
            for p in self.partners_list:
                confusion = np.identity(10) * (1 - self.epsilon) + (self.epsilon / 10)
                p.noise_layer_weights = [np.log(confusion + 1e-8)]
        super(MplSModel, self).fit()

    def fit_minibatch(self):
        """Proceed to a collaborative round with a S-Model federated averaging approach"""

        logger.debug("Start new S-Model collaborative round ...")

        # Starting model for each partner is the aggregated model from the previous mini-batch iteration
        logger.info(f"(S-Model) Minibatch n°{self.minibatch_index} of epoch n°{self.epoch_index}, "
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

            model_input = Input(shape=self.dataset.input_shape)
            x = partner_model(model_input)
            outputs = NoiseAdaptationChannel(weights=partner.noise_layer_weights, name='s-model')(x)
            full_model = Model(inputs=model_input, outputs=outputs, name=f"full_model_partner_{partner_index}")

            full_model.compile(
                loss=partner_model.loss,
                optimizer=partner_model.optimizer,
                metrics='accuracy',
            )

            # Train on partner local data set
            history = full_model.fit(x_batch,
                                     y_batch,
                                     batch_size=partner.batch_size,
                                     verbose=0,
                                     validation_data=self.val_data)

            # Log results of the round
            self.log_partner_perf(partner.id, partner_index, history.history)

            # Update the partner's model in the models' list
            partner.noise_layer_weights = full_model.get_layer('s-model').get_weights()
            partner.model_weights = partner_model.get_weights()

        logger.debug("End of S-Model collaborative round.")


class FederatedGradients(MultiPartnerLearning):
    def __init__(self, scenario, **kwargs):
        super(FederatedGradients, self).__init__(scenario, **kwargs)
        if self.partners_count == 1:
            raise ValueError('Only one partner is provided. Please use the dedicated SinglePartnerLearning class')
        self.model = self.build_model()

    def fit_epoch(self):
        # Split the train dataset in mini-batches
        self.split_in_minibatches()
        # Iterate over mini-batches and train
        for i in range(self.minibatch_count):
            self.minibatch_index = i
            self.fit_minibatch()

        self.minibatch_index = 0

    def fit_minibatch(self):
        """Proceed to a collaborative round with a federated averaging approach"""

        logger.debug("Start new gradients fusion collaborative round ...")

        # Starting model for each partner is the aggregated model from the previous mini-batch iteration
        logger.info(f"(gradient fusion) Minibatch n°{self.minibatch_index} of epoch n°{self.epoch_index}, "
                    f"init each partner's models with a copy of the global model")

        for partner in self.partners_list:
            # Evaluate and store accuracy of mini-batch start model
            partner.model_weights = self.model_weights
        self.eval_and_log_model_val_perf()

        # Iterate over partners for training each individual model
        for partner_index, partner in enumerate(self.partners_list):
            with tf.GradientTape() as tape:
                loss = self.model.loss(partner.minibatched_y_train[self.minibatch_index],
                                       self.model(partner.minibatched_x_train[self.minibatch_index]))
            partner.grads = tape.gradient(loss, self.model.trainable_weights)

        global_grad = self.aggregator.aggregate_gradients()
        self.model.optimizer.apply_gradients(zip(global_grad, self.model.trainable_weights))
        self.model_weights = self.model.get_weights()

        for partner_index, partner in enumerate(self.partners_list):
            val_history = self.model.evaluate(self.val_data[0], self.val_data[1], verbose=False)
            history = self.model.evaluate(partner.minibatched_x_train[self.minibatch_index],
                                          partner.minibatched_y_train[self.minibatch_index], verbose=False)
            history = {
                "loss": [history[0]],
                'accuracy': [history[1]],
                'val_loss': [val_history[0]],
                'val_accuracy': [val_history[1]]
            }

            # Log results of the round
            self.log_partner_perf(partner.id, partner_index, history)

        logger.debug("End of grads-fusion collaborative round.")


####################################################
#
#       Optimized MPL approaches.
#
####################################################

# The fews methods implemented in this section benefit from few improvements,
# which make them 5th to 10th times faster (even more for FastFedGrads).
#
# The main improvements are:
#     - We build only one model, and never re-build
#     - We benefit from tf.Dataset for the batching, shuffling, and prefetching of data.
#     - We benefit from tf.function for the aggregation step.
#

class FastFedAvg(FederatedAverageLearning):
    """ In this version each partner uses its own Adam optimizer."""

    name = 'FastFedAvg'

    def __init__(self, scenario, **kwargs):
        # Attributes related to the data and the model
        self.dataset = scenario.dataset
        self.partners_list = scenario.partners_list
        self.init_model_from = scenario.init_model_from
        self.use_saved_weights = scenario.use_saved_weights
        self.val_set = scenario.val_set
        self.test_set = scenario.test_set

        # Attributes related to iterating at different levels
        self.epoch_count = scenario.epoch_count
        self.minibatch_count = scenario.minibatch_count
        self.gradient_updates_per_pass_count = scenario.gradient_updates_per_pass_count

        # Attributes related to the _aggregation approach
        self.aggregation_method = scenario._aggregation

        # Erase the default parameters (which mostly come from the scenario) if some parameters have been specified
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in ALLOWED_PARAMETERS)

        if self.partners_count == 1:
            raise ValueError('Only one partner is provided. Please use the dedicated SinglePartnerLearning class')

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
        # TODO This aggregator is deprecated, as we now use tf function, which are faster.
        # The best way to use that could be that an aggregator provides a tf.function
        self.agg_w = tf.constant(self.aggregator.aggregation_weights, dtype=tf.float32)

        # Convert the datasets into tf ones.
        self.dataset_name = self.dataset.name

        for p in self.partners_list:
            p.train_data = tf.data.Dataset.from_tensor_slices((p.x_train, p.y_train))
            p.train_data = p.train_data.shuffle(len(p.train_data), reshuffle_each_iteration=True)
            p.train_data = p.train_data.batch(p.batch_size, drop_remainder=True)
            p.train_data = p.train_data.batch(self.gradient_updates_per_pass_count)
            p.train_data = p.train_data.prefetch(1)
        self.train_dataset = [p.train_data for p in self.partners_list]

        self.val_data = tf.data.Dataset.from_tensor_slices((self.dataset.x_val, self.dataset.y_val))
        self.val_data = self.val_data.shuffle(len(self.dataset.y_val), reshuffle_each_iteration=True)
        self.val_data = self.val_data.batch(constants.DEFAULT_BATCH_SIZE).prefetch(1)

        self.test_data = tf.data.Dataset.from_tensor_slices((self.dataset.x_test, self.dataset.y_test))
        self.test_data = self.test_data.shuffle(len(self.dataset.y_test), reshuffle_each_iteration=True)
        self.test_data = self.test_data.batch(constants.DEFAULT_BATCH_SIZE).prefetch(1)

        # model related
        self.generate_new_model = self.dataset.generate_new_model
        self.model = self.init_model()  # TODO check coherence with load model.

        self.history = {}  # TODO This history definition is not the same as current mplc def
        self.end_test_score = {}

        # for early stopping purpose
        self.best = np.Inf
        self.monitor_value = 'val_loss'
        self.monitor_op = np.less
        self.min_delta = np.abs(constants.MIN_DELTA)
        self.patience = constants.PATIENCE
        self.wait = 0
        self.epochs_index = 0

        # tf.function variables
        self.init_specific_tf_variable()

        self.learning_computation_time = 0
        self.timer = 0
        self.epoch_timer = 0

    def init_specific_tf_variable(self):
        # generate tf Variables in which we will store the model weights
        self.partners_weights = [[tf.Variable(initial_value=w.read_value()) for w in
                                  self.model.trainable_weights] for _ in self.partners_list]
        self.partners_optimizers = [self.model.optimizer.from_config(self.model.optimizer.get_config()) for _ in
                                    self.partners_list]

    def log_epoch(self, epoch_number, history):
        logger.info(
            f'[{self.name}] > Epoch {str(epoch_number + 1).ljust(2)}/{self.epoch_count} -'
            f' {f"{np.round(time.time() - self.epoch_timer)} s.".ljust(6)} >'
            f' {" -- ".join(f"{key}: {str(np.round(value, 2)).ljust(5)}" for key, value in history.items())}')
        if not self.history:
            self.history = {key: [value] for key, value in history.items()}
        else:
            for key, value in history.items():
                self.history[key].append(value)

    def log_end_training(self):
        training_time = (time.time() - self.timer)
        self.end_test_score = self.model.evaluate(self.test_data, return_dict=True, verbose=False)
        logger.info(
            f'[{self.name}] > Training {self.epochs_index} epoch in {np.round(training_time, 3)} seconds.'
            f' Tests scores: '
            f'{" -- ".join(f"{key}: {np.around(value, 3)}" for key, value in self.end_test_score.items())}')
        self.learning_computation_time += training_time

    def get_epoch_history(self):
        # get the metrics computed. Here the metrics are computed on all the registered state (y, y_pred) stored,
        # which means on all the minibatches, for all partners.
        history = {m.name: float((m.result().numpy())) for m in self.model.metrics}
        # compute val accuracy and loss for global model
        val_hist = self.model.evaluate(self.val_data, return_dict=True, verbose=False)
        history.update({f'val_{key}': value for key, value in val_hist.items()})
        self.model.reset_metrics()
        return history

    def fit(self):

        # TF function definition
        @tf.function
        def aggregate_weights(weights, agg_w):
            res = list()
            for weights_per_layer in zip(*weights):
                res.append(tf.tensordot(weights_per_layer, agg_w, [0, 0]))
            return res

        @tf.function
        def fit_minibatch(model, partners_minibatches, partners_optimizers, partners_weights, agg_w):
            for p_id, minibatch in enumerate(partners_minibatches):  # minibatch == (x,y)
                # minibatch[0] in a tensor of shape=(number of batch, batch size, img).
                # We cannot iterate on tensors, so we convert this tensor to a list of
                # *number of batch* tensors with shape=(batch size, img)
                x_minibatch = tf.unstack(minibatch[0], axis=0)
                y_minibatch = tf.unstack(minibatch[1], axis=0)  # same here, with labels

                for model_w, partner_w in zip(model.trainable_weights,
                                              partners_weights[p_id]):  # set the model weights to partner's ones
                    model_w.assign(partner_w.read_value())

                for x, y in zip(x_minibatch, y_minibatch):  # iterate over batches
                    with tf.GradientTape() as tape:
                        y_pred = model(x)
                        loss = model.compiled_loss(y, y_pred)
                    model.compiled_metrics.update_state(y, y_pred)  # log the loss and accuracy
                    partners_optimizers[p_id].minimize(loss, model.trainable_weights,
                                                       tape=tape)  # perform local optimization

                for model_w, partner_w in zip(model.trainable_weights,
                                              partners_weights[p_id]):  # update the partner's weights
                    partner_w.assign(model_w.read_value())
            # at the end of the minibatch, aggregate all the local weights
            aggregated_weights = aggregate_weights(tuple(partners_weights),
                                                   agg_w)

            # inform the partners with the new weights
            for p_id in range(len(partners_minibatches)):
                for new_w, partner_w in zip(aggregated_weights, partners_weights[p_id]):
                    partner_w.assign(new_w.read_value())

        # Execution
        self.timer = time.time()
        for e in range(self.epoch_count):
            self.epoch_timer = time.time()
            for partners_minibatches in zip(
                    *self.train_dataset):  # <- partners_minibatches == [(x, y)] * nb_partners
                fit_minibatch(self.model, partners_minibatches, self.partners_optimizers,
                              self.partners_weights, self.agg_w)
            epoch_history = self.get_epoch_history()  # compute val and train acc and loss.
            # add the epoch history to self history, and log epoch number, and metrics values.
            self.log_epoch(e, epoch_history)
            self.epochs_index += 1
            if self.early_stop():
                break

        self.log_end_training()

    def early_stop(self):
        """
        If the metric monitored decreases (or increases, depends on the metrics monitored)
        during patience epoch, we stop the learning.
        """
        if self.monitor_op(self.history[self.monitor_value][-1] - self.min_delta, self.best):
            self.wait = 0
            self.best = self.history[self.monitor_value][-1]
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True
        return False


class FastFedGrad(FastFedAvg):

    def __init__(self, scenario, **kwargs):
        super(FastFedGrad, self).__init__(scenario, **kwargs)
        for p in self.partners_list:
            p.train_data = tf.data.Dataset.from_tensor_slices((p.x_train, p.y_train))
            p.train_data = p.train_data.shuffle(len(p.train_data), reshuffle_each_iteration=True)
            p.train_data = p.train_data.batch(p.batch_size, drop_remainder=True)
            p.train_data = p.train_data.prefetch(1)
        self.train_dataset = [p.train_data for p in self.partners_list]

    def init_specific_tf_variable(self):
        # generate tf Variables in which we will store the model weights
        self.partners_grads = [[tf.Variable(initial_value=w.read_value()) for w in self.model.trainable_weights]
                               for _ in self.partners_list]
        self.partners_optimizers = [self.model.optimizer.from_config(self.model.optimizer.get_config()) for _ in
                                    self.partners_list]

    def fit(self):

        # TF function definition
        @tf.function
        def aggregeted_grads(grads, agg_w):
            global_grad = list()
            for grad_per_layer in zip(*grads):
                global_grad.append(tf.tensordot(grad_per_layer, agg_w, [0, 0]))
            return global_grad

        @tf.function
        def fit_epoch(model, train_dataset, partners_grads, agg_w):
            for minibatch in zip(*train_dataset):
                for p_id, (x_partner_batch, y_partner_batch) in enumerate(minibatch):
                    with tf.GradientTape() as tape:
                        y_pred = model(x_partner_batch)
                        loss = model.compiled_loss(y_partner_batch, y_pred)
                        partners_grads[p_id] = tape.gradient(loss, model.trainable_weights)
                    model.compiled_metrics.update_state(y_partner_batch, y_pred)
                global_grad = aggregeted_grads(tuple(partners_grads), agg_w)
                model.optimizer.apply_gradients(zip(global_grad, model.trainable_weights))

        # Execution
        self.timer = time.time()
        for e in range(self.epoch_count):
            self.epoch_timer = time.time()
            fit_epoch(self.model, self.train_dataset, self.partners_grads, self.agg_w)
            epoch_history = self.get_epoch_history()  # compute val and train acc and loss.
            # add the epoch history to self history, and log epoch number, and metrics values.
            self.log_epoch(e, epoch_history)
            self.epochs_index += 1
            if self.early_stop():
                break

        self.log_end_training()


class FastSmodel(FastFedGrad):
    def __init__(self, scenario, pretrain_epochs, epsilon=0.5, **kwargs):
        super(FastSmodel, self).__init__(scenario, **kwargs)

        self.pretrain_epochs = pretrain_epochs
        self.epsilon = epsilon
        self.smodel_list = [self.generate_smodel() for _ in self.partners_list]

    def generate_smodel(self):
        smodel = tf.keras.models.Sequential()
        smodel.add(tf.keras.Input(shape=(self.model.output_shape[-1],)))
        smodel.add(NoiseAdaptationChannel(name='s-model'))
        opt = self.model.optimizer.from_config(self.model.optimizer.get_config())
        smodel.compile(optimizer=opt, loss='categorical_crossentropy', metrics=self.dataset.model_metrics_names[1:])
        return smodel

    def fit(self):
        @tf.function
        def aggregate_grads(grads, agg_w):
            global_grad = list()
            for grad_per_layer in zip(*grads):
                global_grad.append(tf.tensordot(grad_per_layer, agg_w, [0, 0]))
            return global_grad

        @tf.function
        def fit_pretrain_epoch(model, train_dataset, partners_grads, agg_w):
            for minibatch in zip(*train_dataset):
                for p_id, (x_partner_batch, y_partner_batch) in enumerate(minibatch):
                    with tf.GradientTape() as tape:
                        y_pred = model(x_partner_batch)
                        loss = model.compiled_loss(y_partner_batch, y_pred)
                        partners_grads[p_id] = tape.gradient(loss, model.trainable_weights)
                    model.compiled_metrics.update_state(y_partner_batch, y_pred)
                global_grad = aggregate_grads(tuple(partners_grads), agg_w)
                model.optimizer.apply_gradients(zip(global_grad, model.trainable_weights))

        @tf.function
        def fit_epoch(model, train_dataset, agg_w, partners_grads, smodel_list):
            for minibatch in zip(*train_dataset):
                for p_id, (x_partner_batch, y_partner_batch) in enumerate(minibatch):
                    with tf.GradientTape() as tape:
                        s_model_p = smodel_list[p_id]
                        yt_pred = model(x_partner_batch)
                        yo_pred = s_model_p(yt_pred)
                        loss = model.compiled_loss(y_partner_batch, yo_pred)
                    partners_grads[p_id] = tape.gradient(loss, model.trainable_weights)
                    s_model_p.optimizer.minimize(loss, s_model_p.trainable_weights, tape=tape)
                    model.compiled_metrics.update_state(y_partner_batch, yo_pred)
                global_grad = aggregate_grads(tuple(partners_grads), agg_w)
                model.optimizer.apply_gradients(zip(global_grad, model.trainable_weights))

        # Execution
        self.timer = time.time()
        if self.pretrain_epochs > 0:
            for p_e in range(self.pretrain_epochs):
                logger.info(f'[{self.name}] > Training {self.pretrain_epochs} pretrain epochs first.')
                self.epoch_timer = time.time()
                fit_pretrain_epoch(self.model, self.train_dataset, self.partners_grads, self.agg_w)
                epoch_history = self.get_epoch_history()  # compute val and train acc and loss.
                # add the epoch history to self history, and log epoch number, and metrics values.
                self.log_epoch(p_e, epoch_history)
                self.epochs_index += 1

            for p in self.partners_list:
                confusion = confusion_matrix(np.argmax(p.y_train, axis=1),
                                             np.argmax(self.model.predict(p.x_train), axis=1),
                                             normalize='pred')
                p.noise_layer_weights = [np.log(confusion.T + 1e-8)]
        else:
            for p in self.partners_list:
                confusion = np.identity(10) * (1 - self.epsilon) + (self.epsilon / 10)
                p.noise_layer_weights = [np.log(confusion + 1e-8)]

        for p, smodel in zip(self.partners_list, self.smodel_list):
            smodel.set_weights(p.noise_layer_weights)

        for e in range(self.epoch_count - self.pretrain_epochs):
            self.epoch_timer = time.time()
            fit_epoch(self.model, self.train_dataset, self.partners_grads, self.agg_w)
            epoch_history = self.get_epoch_history()  # compute val and train acc and loss.
            # add the epoch history to self history, and log epoch number, and metrics values.
            self.log_epoch(e, epoch_history)
            self.epochs_index += 1
            if self.early_stop():
                break

        self.log_end_training()


# FedGDO
# FedGDO stands for Federated Gradient Double Optimization.
#
# This method is inspired from Federated gradient, but with modification on the local computation of the gradient.
# In this version we use a local optimizer (partner-specific) to do several minimization steps of the local-loss
# during a minibatch. We use the sum of these weighs-updates as the gradient which is sent to the global optimizer.
# The global optimizer aggregates these gradients-like which have been sent by the partners,
# and performs a optimization step with this aggregated gradient.
#
# Here three variations of this mpl method are tested.
#
# - FedGDO with fresh local optimizer at each minibatch
# - FedGDO with persistent local optimizer
# - FedGDO with persistent and informed local optimizer

class FedGDO_fresh_lopt(FastFedAvg):
    name = 'FedGDOf'

    def init_specific_tf_variable(self):
        # generate tf Variables in which we will store the model weights.
        self.model_stateholder = [tf.Variable(initial_value=w.read_value()) for w in self.model.trainable_weights]
        self.partners_optimizers = [self.model.optimizer.from_config(self.model.optimizer.get_config()) for _ in
                                    self.partners_list]
        # TODO add the possibility to use another optimizer for local opt
        self.partners_grads = [[tf.Variable(initial_value=w.read_value()) for w in self.model.trainable_weights] for _
                               in self.partners_list]

    def fit(self):
        # TF function definition
        @tf.function
        def aggregeted_grads(grads, agg_w):
            global_grad = list()
            for grad_per_layer in zip(*grads):
                global_grad.append(tf.tensordot(grad_per_layer, agg_w, [0, 0]))
            return global_grad

        # @tf.function cannot be used as we need to re-initialized the optimizer.
        # Variables initialization is only allowed at the first call of a tf.function
        def fit_minibatch(model, model_stateholder, partners_minibatches, partners_optimizers, partners_grads, agg_w):
            for p_id, minibatch in enumerate(partners_minibatches):  # minibatch == (x,y)
                # minibatch[0] in a tensor of shape=(number of batch, batch size, img).
                # We cannot iterate on tensors, so we convert this tensor to a list of *number of batch*
                # tensors with shape=(batch size, img)
                x_minibatch = tf.unstack(minibatch[0], axis=0)
                y_minibatch = tf.unstack(minibatch[1], axis=0)  # same here, with labels

                for model_w, old_w in zip(model.trainable_weights, model_stateholder):  # store model weights
                    old_w.assign(model_w.read_value())

                for x, y in zip(x_minibatch, y_minibatch):  # iterate over batches
                    with tf.GradientTape() as tape:
                        y_pred = model(x)
                        loss = model.compiled_loss(y, y_pred)
                    model.compiled_metrics.update_state(y, y_pred)  # log the loss and accuracy
                    partners_optimizers[p_id].minimize(loss, model.trainable_weights,
                                                       tape=tape)  # perform local optimization

                for grad_per_layer, w_old, w_new in zip(partners_grads[p_id], model_stateholder,
                                                        model.trainable_weights):
                    grad_per_layer.assign(
                        (w_old - w_new))  # get the gradient as theta_before_minibatch - theta_after_minibatch

                for model_w, old_w in zip(model.trainable_weights,
                                          model_stateholder):  # reset the model's weights for the next partner
                    model_w.assign(old_w.read_value())

            global_grad = aggregeted_grads(tuple(partners_grads),
                                           agg_w)  # at the end of the minibatch, aggregate all the local gradients.
            model.optimizer.apply_gradients(
                zip(global_grad, model.trainable_weights))  # apply the global gradients with the global optimizer

        # Execution
        self.timer = time.time()
        for e in range(self.epoch_count):
            self.epoch_timer = time.time()
            for partners_minibatches in zip(*self.train_dataset):  # <- partners_minibatches == [(x, y)] * nb_partners
                self.partners_optimizers = [self.model.optimizer.from_config(self.model.optimizer.get_config()) for _ in
                                            self.partners_list]  # reset the local optimizers
                fit_minibatch(self.model, self.model_stateholder, partners_minibatches, self.partners_optimizers,
                              self.partners_grads, self.agg_w)
            epoch_history = self.get_epoch_history()  # compute val and train acc and loss.
            # add the epoch history to self history, and log epoch number, and metrics values.
            self.log_epoch(e, epoch_history)
            self.epochs_index += 1
            if self.early_stop():
                break

        self.log_end_training()


class FedGDO_persistent_lopt(FedGDO_fresh_lopt):
    name = 'FedGDOp'

    def fit(self):
        @tf.function
        def aggregeted_grads(grads, agg_w):
            global_grad = list()
            for grad_per_layer in zip(*grads):
                global_grad.append(tf.tensordot(grad_per_layer, agg_w, [0, 0]))
            return global_grad

        @tf.function
        def fit_minibatch(model, model_stateholder, partners_minibatches, partners_optimizers, partners_grads, agg_w):
            for p_id, minibatch in enumerate(partners_minibatches):  # minibatch == (x,y)
                x_minibatch = tf.unstack(minibatch[0], axis=0)
                y_minibatch = tf.unstack(minibatch[1], axis=0)
                for model_w, old_w in zip(model.trainable_weights, model_stateholder):  # store model weigths
                    old_w.assign(model_w.read_value())
                for x, y in zip(x_minibatch, y_minibatch):
                    with tf.GradientTape() as tape:
                        y_pred = model(x)
                        loss = model.compiled_loss(y, y_pred)
                    model.compiled_metrics.update_state(y, y_pred)
                    partners_optimizers[p_id].minimize(loss, model.trainable_weights, tape=tape)  # local optimization
                for grad_per_layer, w_old, w_new in zip(partners_grads[p_id], model_stateholder,
                                                        model.trainable_weights):
                    grad_per_layer.assign(
                        (w_old - w_new))  # get the gradient as theta_before_minibatch - theta_after_minibatch
                for model_w, old_w in zip(model.trainable_weights,
                                          model_stateholder):  # reset the model's weights for the next partner
                    model_w.assign(old_w.read_value())
            global_grad = aggregeted_grads(tuple(partners_grads),
                                           agg_w)  # at the end of the minibatch, aggregate the gradients.
            model.optimizer.apply_gradients(
                zip(global_grad, model.trainable_weights))  # apply the global gradients with the global optimizer

        self.timer = time.time()
        for e in range(self.epoch_count):
            self.epoch_timer = time.time()
            for partners_minibatches in zip(*self.train_dataset):  # <- partners_minibatches == [(x, y)] * nb_partners
                fit_minibatch(self.model, self.model_stateholder, partners_minibatches, self.partners_optimizers,
                              self.partners_grads, self.agg_w)
            epoch_history = self.get_epoch_history()  # compute val and train acc and loss.
            # add the epoch history to self history, and log epoch number, and metrics values.
            self.log_epoch(e, epoch_history)
            self.epochs_index += 1
            if self.early_stop():
                break

        self.log_end_training()


class FedGDO_persistent_informed_lopt(FedGDO_fresh_lopt):
    name = 'FedGDOpi'

    def fit(self):
        @tf.function
        def load_stored_weights(model, model_stateholder):
            for model_w, old_w in zip(model.trainable_weights,
                                      model_stateholder):  # reset the model's weights for the next partner
                model_w.assign(old_w.read_value())

        @tf.function
        def store_model_weights(model, model_stateholder):
            for model_w, old_w in zip(model.trainable_weights, model_stateholder):  # store model weigths
                old_w.assign(model_w.read_value())

        @tf.function
        def aggregeted_grads(grads, agg_w):
            global_grad = list()
            for grad_per_layer in zip(*grads):
                global_grad.append(tf.tensordot(grad_per_layer, agg_w, [0, 0]))
            return global_grad

        @tf.function
        def fit_minibatch(model, model_stateholder, partners_minibatches, partners_optimizers, partners_grads, agg_w):
            store_model_weights(model, model_stateholder)
            for p_id, minibatch in enumerate(partners_minibatches):  # minibatch == (x,y)
                x_minibatch = tf.unstack(minibatch[0], axis=0)
                y_minibatch = tf.unstack(minibatch[1], axis=0)
                for x, y in zip(x_minibatch, y_minibatch):
                    with tf.GradientTape() as tape:
                        y_pred = model(x)
                        loss = model.compiled_loss(y, y_pred)
                    model.compiled_metrics.update_state(y, y_pred)
                    partners_optimizers[p_id].minimize(loss, model.trainable_weights, tape=tape)  # local optimization
                for grad_per_layer, w_old, w_new in zip(partners_grads[p_id], model_stateholder,
                                                        model.trainable_weights):
                    grad_per_layer.assign(
                        (w_old - w_new))  # get the gradient as theta_before_minibatch - theta_after_minibatch
                load_stored_weights(model, model_stateholder)

            global_grad = aggregeted_grads(tuple(partners_grads),
                                           agg_w)  # at the end of the minibatch, aggregate the gradients.

            # for each local optimizer we perform an optimization step with the global gradient,
            # to update the momentums and learning rate with this global gradient.
            # We informe the local optimizer of the global optimization.

            for optimizer in partners_optimizers:
                optimizer.apply_gradients(zip(global_grad, model.trainable_weights))
                # in fact the optimization of the weights must be done by the global optimizer,
                # so we reset the weights after the optimization
                load_stored_weights(model, model_stateholder)

            model.optimizer.apply_gradients(
                zip(global_grad, model.trainable_weights))  # apply the global gradients with the global optimizer

        self.timer = time.time()
        for e in range(self.epoch_count):
            self.epoch_timer = time.time()
            for partners_minibatches in zip(*self.train_dataset):  # <- partners_minibatches == [(x, y)] * nb_partners
                fit_minibatch(self.model, self.model_stateholder, partners_minibatches, self.partners_optimizers,
                              self.partners_grads, self.agg_w)
            epoch_history = self.get_epoch_history()  # compute val and train acc and loss.
            # add the epoch history to self history, and log epoch number, and metrics values.
            self.log_epoch(e, epoch_history)

            self.epochs_index += 1
            if self.early_stop():
                break

        self.log_end_training()


# Supported multi-partner learning approaches

MULTI_PARTNER_LEARNING_APPROACHES = {
    "fedavg": FederatedAverageLearning,
    'fedgrads': FederatedGradients,
    "seq-pure": SequentialLearning,
    "seq-with-final-agg": SequentialWithFinalAggLearning,
    "seqavg": SequentialAverageLearning,
    "smodel": MplSModel,

}
