import os
import pickle
from abc import ABC
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

@tf.function
def _tf_aggregete_grads(grads, agg_w):
    global_grad = list()
    for grad_per_layer in zip(*grads):
        g = list()
        for g_p, w in zip(grad_per_layer, agg_w):
            g.append(g_p)
        global_grad.append(tf.reduce_mean(g, axis=0))
    return global_grad


class Aggregator(ABC):
    name = 'abstract'
    is_weights_constant = True

    def __init__(self, mpl):
        """
        :type mpl: MultiPartnerLearning
        """
        self.mpl = mpl
        self.aggregation_weights = np.zeros(self.mpl.partners_count, dtype='float32')

    def __str__(self):
        return f'{self.name} aggregator'

    def aggregate_model_weights(self, partial_partners_list=None):
        """Aggregate model weights from the list or partial list of partner's models, with a weighted average"""

        weights_per_layer = list(zip(*[partner.model_weights for partner in self.mpl.partners_list]))
        new_weights = list()

        for weights_for_layer in weights_per_layer:
            avg_weights_for_layer = np.average(
                np.array(weights_for_layer), axis=0, weights=self.aggregation_weights
            )
            new_weights.append(avg_weights_for_layer)

        return new_weights

    def aggregate_gradients(self):
        assert isinstance(self.aggregation_weights, list), 'Aggregation weights must be a list.'
        return _tf_aggregete_grads([p.grads for p in self.mpl.partners_list], self.aggregation_weights)


class UniformAggregator(Aggregator):
    name = 'uniform'
    is_weights_constant = True

    def __init__(self, mpl):
        super(UniformAggregator, self).__init__(mpl)
        self.aggregation_weights = list(np.ones(self.mpl.partners_count, dtype='float32') / self.mpl.partners_count)


class DataVolumeAggregator(Aggregator):
    name = 'data-volume'
    is_weights_constant = True

    def __init__(self, mpl):
        super(DataVolumeAggregator, self).__init__(mpl)
        partners_sizes = [partner.data_volume for partner in self.mpl.partners_list]
        self.aggregation_weights = list((partners_sizes / np.sum(partners_sizes).astype('float32')))


class ScoresAggregator(Aggregator):
    name = 'local-score'
    is_weights_constant = False

    def __init__(self, mpl):
        super(ScoresAggregator, self).__init__(mpl)

    def prepare_aggregation_weights(self):
        last_scores = [partner.last_round_score for partner in self.mpl.partners_list]
        self.aggregation_weights = list((last_scores / np.sum(last_scores)).astype('float32'))

    def aggregate_model_weights(self):
        self.prepare_aggregation_weights()
        super(ScoresAggregator, self).aggregate_model_weights()

    def aggregate_gradients(self):
        self.prepare_aggregation_weights()
        super(ScoresAggregator, self).aggregate_gradients()


# Supported _aggregation weights approaches
AGGREGATORS = {
    "uniform": UniformAggregator,
    "data-volume": DataVolumeAggregator,
    "local-score": ScoresAggregator
}
