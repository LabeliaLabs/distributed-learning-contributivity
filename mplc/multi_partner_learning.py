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
from .mpl_utils import History, DatavolumeAggregator
from .partner import PartnerMpl


class MultiPartnerLearning(ABC):
    def __init__(self,
                 partners_list,
                 epoch_count,
                 minibatch_count,
                 dataset,
                 aggregation=DatavolumeAggregator,
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
                    and self.history.history['model']['loss'][-1, -1] >
                    self.history.history['model']['loss'][-constants.PATIENCE, -1]
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
        if type(partner) == list:
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
                 aggregation=DataAggregator,
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
                 aggregation=DatavolumeAggregator,
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
                 aggregation=DatavolumeAggregator,
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
                 aggregation=DatavolumeAggregator,
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


class MplLabelFlip(FederatedAverageLearning):
    def __init__(self, scenario, is_save_data=False, epsilon=0.01):
        super(MplLabelFlip, self).__init__(
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

        self.epsilon = epsilon
        self.K = scenario.dataset.num_classes
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
        self.history.log_model_val_perf()

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
            self.history.log_partner_perf(partner.id, partner_index, history.history)

            # Update the partner's model in the models' list
            partner.model_weights = partner_model.get_weights()

        logger.debug("End of LFlip collaborative round.")
