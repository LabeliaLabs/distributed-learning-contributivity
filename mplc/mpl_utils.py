import os
import pickle
from abc import ABC
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from . import constants


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
        tab = {key: np.nan * np.zeros((mpl.epoch_count, mpl.minibatch_count)) for key in self.metrics}
        self.history = {partner.id: deepcopy(tab) for partner in mpl.partners_list}
        self.history['mpl_model'] = {'val_accuracy': np.zeros((mpl.epoch_count, mpl.minibatch_count)),
                                     'val_loss': np.zeros((mpl.epoch_count, mpl.minibatch_count))}

    def log_partner_perf(self, partner_id, partner_index, history):
        for key in self.metrics:
            self.history[partner_id][key][self.mpl.epoch_index, self.mpl.minibatch_index] = history[key][-1]

        epoch_nb_str = f"Epoch {str(self.mpl.epoch_index).zfill(2)}/{str(self.mpl.epoch_count - 1).zfill(2)}"
        mb_nb_str = f"Minibatch {str(self.mpl.minibatch_index).zfill(2)}/{str(self.mpl.minibatch_count - 1).zfill(2)}"
        partner_id_str = f"Partner partner_id #{partner_id} ({partner_index}/{self.mpl.partners_count - 1})"
        val_acc_str = f"{round(history['val_accuracy'][-1], 2)}"

        logger.debug(f"{epoch_nb_str} > {mb_nb_str} > {partner_id_str} > val_acc: {val_acc_str}")

    def log_model_val_perf(self):
        model = self.mpl.build_model()
        hist = model.evaluate(self.mpl.val_data[0],
                              self.mpl.val_data[1],
                              batch_size=constants.DEFAULT_BATCH_SIZE,
                              verbose=0,
                              )
        self.history['mpl_model']['val_loss'][self.mpl.epoch_index, self.mpl.minibatch_index] = hist[0]
        self.history['mpl_model']['val_accuracy'][self.mpl.epoch_index, self.mpl.minibatch_index] = hist[1]

        if self.mpl.minibatch_index >= self.mpl.minibatch_count - 1:
            logger.info(f"   Model evaluation at the end of the epoch: "
                        f"{['%.3f' % elem for elem in hist]}")

    def log_final_model_perf(self):
        logger.info("### Evaluating model on test data:")
        model = self.mpl.build_model()
        hist = model.evaluate(self.mpl.test_data[0],
                              self.mpl.test_data[1],
                              batch_size=constants.DEFAULT_BATCH_SIZE,
                              verbose=0,
                              )
        self.score = hist[1]
        self.nb_epochs_done = self.mpl.epoch_index + 1
        logger.info(f"   Model metrics names: {self.mpl.metrics_names}")
        logger.info(f"   Model metrics values: {['%.3f' % elem for elem in hist]}")

    def partners_to_dataframe(self):
        temp_dict = {'Partner': [],
                     'Epoch': [],
                     'Minibatch': []}
        for key in self.metrics:
            temp_dict[key] = []
        for partner_id, hist in [(key, value) for key, value in self.history.items() if key != 'mpl_model']:
            epoch_count, minibatch_count = self.history['mpl_model']['val_loss'].shape
            for epoch in range(epoch_count):
                for mb in range(minibatch_count):
                    temp_dict['Partner'].append(partner_id)
                    temp_dict['Epoch'].append(epoch)
                    temp_dict['Minibatch'].append(mb)
                    for metric, matrix in hist.items():
                        temp_dict[metric].append(matrix[epoch, mb])
        return pd.DataFrame.from_dict(temp_dict)

    def save_data(self):
        """Save figures, losses and metrics to disk"""

        with open(self.save_folder / "history_data.p", 'wb') as f:
            pickle.dump(self.history, f)

        if not os.path.exists(self.save_folder / 'graphs/'):
            os.makedirs(self.save_folder / 'graphs/')
        plt.figure()
        plt.plot(self.history['mpl_model']['val_loss'][:self.mpl.epoch_index + 1, self.mpl.minibatch_count])
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig(self.save_folder / "graphs/federated_training_loss.png")
        plt.close()

        plt.figure()
        plt.plot(self.history['mpl_model']['val_accuracy'][:self.mpl.epoch_index + 1, self.mpl.minibatch_count])
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.ylim([0, 1])
        plt.savefig(self.save_folder / "graphs/federated_training_acc.png")
        plt.close()

        plt.figure()
        for key, value in self.history.items():
            plt.plot(value['val_accuracy'][:self.mpl.epoch_index + 1, self.mpl.minibatch_count],
                     label=(f'partner {key}' if key != 'mpl_model' else key))
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        plt.ylim([0, 1])
        plt.savefig(self.save_folder / "graphs/all_partners.png")
        plt.close()


class Aggregator(ABC):
    def __init__(self, mpl):
        """
        :type mpl: MultiPartnerLearning
        """
        self.mpl = mpl
        self.aggregation_weights = np.zeros(self.mpl.partners_count)

    def aggregate_model_weights(self):
        """Aggregate model weights from the list of partner's models, with a weighted average"""

        weights_per_layer = list(zip(*[partner.model_weights for partner in self.mpl.partners_list]))
        new_weights = list()

        for weights_for_layer in weights_per_layer:
            avg_weights_for_layer = np.average(
                np.array(weights_for_layer), axis=0, weights=self.aggregation_weights
            )
            new_weights.append(list(avg_weights_for_layer))

        return new_weights


class UniformAggregator(Aggregator):
    def __init__(self, mpl):
        super(UniformAggregator, self).__init__(mpl)
        self.aggregation_weights = [1 / self.mpl.partners_count] * self.mpl.partners_count


class DatavolumeAggregator(Aggregator):
    def __init__(self, mpl):
        super(DatavolumeAggregator, self).__init__(mpl)
        partners_sizes = [partner.data_volume for partner in self.mpl.partners_list]
        self.aggregation_weights = partners_sizes / np.sum(partners_sizes)


class ScoresAggregator(Aggregator):
    def __init__(self, mpl):
        super(ScoresAggregator, self).__init__(mpl)

    def prepare_aggregation_weights(self):
        last_scores = [partner.last_round_score for partner in self.mpl.partners_list]
        self.aggregation_weights = last_scores / np.sum(last_scores)

    def aggregate_model_weights(self):
        self.prepare_aggregation_weights()
        super(ScoresAggregator, self).aggregate_model_weights()


# Supported aggregation weights approaches
AGGREGATORS = {
    "uniform": UniformAggregator,
    "data-volume": DatavolumeAggregator,
    "local-score": ScoresAggregator
}
