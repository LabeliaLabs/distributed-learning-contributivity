import os
import pickle
from abc import ABC, abstractmethod
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


class History:
    def __init__(self, mpl):
        """
        This object tracks the loss and the accuracy of the different models, partner's and global.

        :type mpl: MultiPartnerLearning
        """

        self.mpl = mpl
        self.save_folder = mpl.save_folder
        self.nb_epochs_done = 0
        self.score = None  # Final score evaluated on the test dataset at the end of the training
        self.metrics = ['val_accuracy', 'val_loss', 'loss', 'accuracy']
        temp_dict = {key: np.nan * np.zeros((mpl.epoch_count, mpl.minibatch_count)) for key in self.metrics}
        self.history = {partner.id: deepcopy(temp_dict) for partner in mpl.partners_list}
        self.history['mpl_model'] = {'val_accuracy': np.zeros((mpl.epoch_count, mpl.minibatch_count)),
                                     'val_loss': np.zeros((mpl.epoch_count, mpl.minibatch_count))}

    def partners_to_dataframe(self):
        temp_dict = {'Model': [],
                     'Epoch': [],
                     'Minibatch': []}
        for key in self.metrics:
            temp_dict[key] = []
        for partner_id, hist in [(key, value) for key, value in self.history.items() if key != 'mpl_model']:
            epoch_count, minibatch_count = self.history['mpl_model']['val_loss'].shape
            for epoch in range(epoch_count):
                for mb in range(minibatch_count):
                    temp_dict['Model'].append(f'partner_{partner_id}')
                    temp_dict['Epoch'].append(epoch)
                    temp_dict['Minibatch'].append(mb)
                    for metric, matrix in hist.items():
                        temp_dict[metric].append(matrix[epoch, mb])
        return pd.DataFrame.from_dict(temp_dict)

    def global_model_to_dataframe(self):
        temp_dict = {'Epoch': [],
                     'Minibatch': []}
        for key in self.history['mpl_model'].keys():
            temp_dict[key] = []
        epoch_count, minibatch_count = self.history['mpl_model']['val_loss'].shape
        for epoch in range(epoch_count):
            for mb in range(minibatch_count):
                temp_dict['Epoch'].append(epoch)
                temp_dict['Minibatch'].append(mb)
                for metric, matrix in self.history['mpl_model'].items():
                    temp_dict[metric].append(matrix[epoch, mb])
        return pd.DataFrame.from_dict(temp_dict)

    def history_to_dataframe(self):
        partners_df = self.partners_to_dataframe()
        mpl_model_df = self.global_model_to_dataframe()
        mpl_model_df['Model'] = 'mpl_model'

        return partners_df.append(mpl_model_df, ignore_index=True)

    def save_data(self, binary=False):
        """Save figures, losses and metrics to disk
            :param binary : bool, set to false by default.
                            If True, the history.history dictionary is pickled and saved in binary format.
                            If True, the pandas dataframe version of the history are saved as .csv file"""

        if self.save_folder is None:
            raise ValueError("The path to the save folder is None, history data cannot be saved")
        if binary:
            with open(self.save_folder / "history_data.p", 'wb') as f:
                pickle.dump(self.history, f)
        else:
            history_df = self.history_to_dataframe()
            history_df.to_csv(self.mpl.save_folder / "history.csv")

        if not os.path.exists(self.save_folder / 'graphs/'):
            os.makedirs(self.save_folder / 'graphs/')
        plt.figure()
        plt.plot(self.history['mpl_model']['val_loss'][:, -1])
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig(self.save_folder / "graphs/federated_training_val_loss.png")
        plt.close()

        plt.figure()
        plt.plot(self.history['mpl_model']['val_accuracy'][:, -1])
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.ylim([0, 1])
        plt.savefig(self.save_folder / "graphs/federated_training_val_acc.png")
        plt.close()

        plt.figure()
        for key, value in self.history.items():
            plt.plot(value['val_accuracy'][:, -1],
                     label=(f'partner {key}' if key != 'mpl_model' else key))
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        plt.ylim([0, 1])
        plt.savefig(self.save_folder / "graphs/all_partners_val_acc.png")
        plt.close()


#####################################
#
# Aggregators
#
#####################################


class Aggregator(ABC):
    name = 'abstract'

    def __init__(self):

        self._aggregation_weights = None
        self.mpl = None

    def __str__(self):
        return f'{self.name} aggregator'

    def aggregation_function_for_model_weights(self, mpl):
        """Build the tf.function which aggregate model weights from the list of partner's models,
        with a weighted average
        """
        self.mpl = mpl
        aggregation_weights = self.aggregation_weights()

        @tf.function
        def aggregator(global_weights, local_weights):
            for i, weights_per_layer in enumerate(zip(*local_weights)):
                global_weights[i].assign(tf.tensordot(weights_per_layer, aggregation_weights, [0, 0]))

        return aggregator

    def aggregation_function_for_model_gradients(self, mpl):
        """Build the tf.function which aggregate gradients from the list of partner's gradients,
        with a weighted average
        """
        self.mpl = mpl
        aggregation_weights = self.aggregation_weights()

        @tf.function
        def aggregator(local_grad):
            global_grad = list()
            for weights_per_layer in zip(*local_grad):
                global_grad.append(tf.tensordot(weights_per_layer, aggregation_weights, [0, 0]))

            return global_grad

        return aggregator

    @abstractmethod
    def aggregation_weights(self):
        assert self.mpl is not None, "aggregation_function_for_model_weights or grads should have been called"
        if self._aggregation_weights:
            return self._aggregation_weights
        else:
            temp = np.zeros(self.mpl.partners_count, dtype='float32')
            self._aggregation_weights = tf.constant(temp, dtype=tf.float32)


class UniformAggregator(Aggregator):
    name = 'Uniform'

    def aggregation_weights(self):
        if self._aggregation_weights:
            return self._aggregation_weights
        else:
            temp = np.ones(self.mpl.partners_count, dtype='float32')
            self._aggregation_weights = tf.constant(temp, dtype=tf.float32)


class DataVolumeAggregator(Aggregator):
    name = 'Data volume'

    def aggregation_weights(self):
        if self._aggregation_weights:
            return self._aggregation_weights
        else:
            partners_sizes = [partner.data_volume for partner in self.mpl.partners_list]
            temp = (partners_sizes / np.sum(partners_sizes).astype('float32'))
            self._aggregation_weights = tf.constant(temp, dtype=tf.float32)


class ScoresAggregator(Aggregator):
    name = 'Local scores'

    def aggregation_function_for_model_weights(self, mpl):
        """Build the tf.function which aggregate model weights from the list of partner's models,
        with a weighted average
        """
        self.mpl = mpl
        mpl._aggregation_weights = tf.Variable(np.zeros(mpl.partners_count, dtype='float32'))

        def aggregator(global_weights, local_weights):
            last_scores = np.array([partner.last_round_score for partner in mpl.partners_list])
            mpl._aggregation_weights.assign(last_scores / np.sum(last_scores))
            for i, weights_per_layer in enumerate(zip(*local_weights)):
                global_weights[i].assign(tf.tensordot(weights_per_layer, mpl._aggregation_weights, [0, 0]))

        return aggregator

    def aggregation_function_for_model_gradients(self, mpl):
        """Build the tf.function which aggregate gradients from the list of partner's gradients,
        with a weighted average
        """
        self.mpl = mpl
        mpl._aggregation_weights = tf.Variable(np.zeros(mpl.partners_count, dtype='float32'))

        @tf.function
        def aggregator(local_grad):
            global_grad = list()
            last_scores = np.array([partner.last_round_score for partner in mpl.partners_list])
            mpl._aggregation_weights.assign(last_scores / np.sum(last_scores))
            for weights_per_layer in zip(*local_grad):
                global_grad.append(tf.tensordot(weights_per_layer, mpl._aggregation_weights, [0, 0]))

            return global_grad

        return aggregator

    def aggregation_weights(self):
        return self.mpl._aggregation_weights


# Supported _aggregation weights approaches
AGGREGATORS = {
    "uniform": UniformAggregator,
    "data-volume": DataVolumeAggregator,
    "local-score": ScoresAggregator
}
