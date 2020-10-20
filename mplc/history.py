import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from . import constants


class History():
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
        self.history = {partner.id: tab.copy() for partner in mpl.partners_list}
        self.history['model'] = {'val_accuracy': np.zeros((mpl.epoch_count, mpl.minibatch_count)),
                                 'val_loss': np.zeros((mpl.epoch_count, mpl.minibatch_count))}

    def log_model_val_perf(self):
        model = self.mpl.get_model()
        hist = model.evaluate(self.mpl.val_data[0],
                              self.mpl.val_data[1],
                              batch_size=constants.DEFAULT_BATCH_SIZE,
                              verbose=0,
                              )
        self.history['model']['val_loss'][self.mpl.epoch_index, self.mpl.minibatch_index] = hist[0]
        self.history['model']['val_accuracy'][self.mpl.epoch_index, self.mpl.minibatch_index] = hist[1]

        if self.mpl.minibatch_index >= self.mpl.minibatch_count - 1:
            logger.info(f"   Model evaluation at the end of the epoch: "
                        f"{['%.3f' % elem for elem in hist]}")

    def log_partner_perf(self, partner_id, partner_index, history):
        for key in self.metrics:
            self.history[partner_id][key][self.mpl.epoch_index, self.mpl.minibatch_index] = history[key][-1]

        epoch_nb_str = f"Epoch {str(self.mpl.epoch_index).zfill(2)}/{str(self.mpl.epoch_count - 1).zfill(2)}"
        mb_nb_str = f"Minibatch {str(self.mpl.minibatch_index).zfill(2)}/{str(self.mpl.minibatch_count - 1).zfill(2)}"
        partner_id_str = f"Partner partner_id #{partner_id} ({partner_index}/{self.mpl.partners_count - 1})"
        val_acc_str = f"{round(history['val_accuracy'][-1], 2)}"

        logger.debug(f"{epoch_nb_str} > {mb_nb_str} > {partner_id_str} > val_acc: {val_acc_str}")

    def log_final_model_perf(self):
        logger.info("### Evaluating model on test data:")
        model = self.mpl.get_model()
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
        dict = {'Partner': [],
                'Epoch': [],
                'Minibatch': []}
        for key in self.metrics:
            dict[key] = []
        for partner_id, hist in [(key, value) for key, value in self.history.items() if key != 'model']:
            for metric, matrix in hist.items():
                for epoch in range(matrix.shape[0]):
                    for mb in range(matrix.shape[1]):
                        dict['Partner'].append(partner_id)
                        dict['Epoch'].append(epoch)
                        dict['Minibatch'].append(mb)
                        dict[metric].append(matrix[epoch, mb])
        return pd.DataFrame.from_dict(dict)

    def save_data(self):
        """Save figures, losses and metrics to disk"""

        with open(self.save_folder / "history_data.p", 'wb') as f:
            pickle.dump(self.history, f)

        if not os.path.exists(self.save_folder / 'graphs/'):
            os.makedirs(self.save_folder / 'graphs/')
        plt.figure()
        plt.plot(self.history['model']['val_loss'][:self.mpl.epoch_index + 1, self.mpl.minibatch_count])
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig(self.save_folder / "graphs/federated_training_loss.png")
        plt.close()

        plt.figure()
        plt.plot(self.history['model']['val_accuracy'][:self.mpl.epoch_index + 1, self.mpl.minibatch_count])
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        # plt.yscale('log')
        plt.ylim([0, 1])
        plt.savefig(self.save_folder / "graphs/federated_training_acc.png")
        plt.close()

        plt.figure()
        for key, value in self.history.items():
            plt.plot(value['val_accuracy'][:self.mpl.epoch_index + 1, self.mpl.minibatch_count],
                     label=(f'partner {key}' if key != 'model' else key))
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        # plt.yscale('log')
        plt.ylim([0, 1])
        plt.savefig(self.save_folder / "graphs/all_partners.png")
        plt.close()
