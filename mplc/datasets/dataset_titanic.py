# -*- coding: utf-8 -*-
"""
Titanic dataset.
(inspired from: https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html)
"""

from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
from joblib import dump, load
from loguru import logger
from sklearn.linear_model import LogisticRegression as skLR
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from .. import constants, dataset


def generate_new_dataset():
    # Init dataset-specific variables
    num_classes = 2
    input_shape = (26,)

    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()

    dataset_obj = dataset.Dataset(
        "titanic",
        x_train,
        x_test,
        y_train,
        y_test,
        input_shape,
        num_classes,
        generate_new_model_for_dataset,
        train_val_split_global,
        train_test_split_local,
        train_val_split_local
    )
    return dataset_obj


# Init dataset-specific functions
def preprocess_dataset_labels(y):
    """Legacy"""

    return y


def preprocess_dataset_inputs(x):
    """Feature engineering"""

    x['Fam_size'] = x['Siblings/Spouses Aboard'] + x['Parents/Children Aboard']

    x['Name_Len'] = [len(i) for i in x["Name"]]

    x['Is_alone'] = [i == 0 for i in x["Fam_size"]]

    x["Sex"] = [i == "Male" for i in x["Sex"]]

    x['Title'] = [i.split()[0] for i in x["Name"]]
    x = pd.concat([x, pd.get_dummies(x['Title'])], axis=1)

    x = pd.concat([x, pd.get_dummies(x['Pclass'])], axis=1)

    # Dropping the useless features
    x.drop('Name', axis=1, inplace=True)
    x.drop('Pclass', axis=1, inplace=True)
    x.drop('Siblings/Spouses Aboard', axis=1, inplace=True)
    x.drop('Parents/Children Aboard', axis=1, inplace=True)
    x.drop('Title', axis=1, inplace=True)
    return x.to_numpy()


def load_data():
    """Return a usable dataset"""
    path = Path(__file__).resolve().parents[0]
    folder = path / 'local_data' / 'titanic'
    if not folder.is_dir():
        Path.mkdir(folder, parents=True)
        logger.info('Titanic dataset not found. Downloading it...')
        attempts = 0
        while True:
            try:
                raw_dataset = pd.read_csv(
                    'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv',
                    index_col=False)
                break
            except (HTTPError, URLError) as e:
                if hasattr(e, 'code'):
                    temp = e.code
                else:
                    temp = e.errno
                logger.debug(
                    f'URL fetch failure on '
                    f'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv : '
                    f'{temp} -- {e.reason}')
                if attempts < constants.NUMBER_OF_DOWNLOAD_ATTEMPTS:
                    sleep(2)
                    attempts += 1
                else:
                    raise

        raw_dataset.to_csv((folder / "titanic.csv").resolve())
    else:
        raw_dataset = pd.read_csv((folder / "titanic.csv").resolve())
    x = raw_dataset.drop('Survived', axis=1)
    x = preprocess_dataset_inputs(x)
    y = raw_dataset['Survived']
    y = y.to_numpy()

    x_train, x_test, y_train, y_test = train_test_split_global(x, y)

    return (x_train, y_train), (x_test, y_test)


# Model structure and generation
def generate_new_model_for_dataset():
    """Return a LogisticRegression Classifier"""

    clf = LogisticRegression()
    clf.classes_ = np.array([0, 1])
    clf.metrics_names = ["log_loss", "Accuracy"]  # Mimic Keras's
    return clf


# train, test, val splits

def train_test_split_local(x, y):
    return train_test_split(x, y, test_size=0.1, random_state=42)


def train_val_split_local(x, y):
    return train_test_split(x, y, test_size=0.1, random_state=42)


def train_val_split_global(x, y):
    return train_test_split(x, y, test_size=0.1, random_state=42)


def train_test_split_global(x, y):
    return train_test_split(x, y, test_size=0.1, random_state=42)


class LogisticRegression(skLR):
    def __init__(self):
        super(LogisticRegression, self).__init__(max_iter=10000, warm_start=1, random_state=0)
        self.coef_ = None
        self.intercept_ = None

    def fit(self, x_train,
            y_train,
            batch_size,
            epochs,
            verbose,
            validation_data):
        history = super(LogisticRegression, self).fit(x_train, y_train)
        [loss, acc] = self.evaluate(x_train, y_train)
        [val_loss, val_acc] = self.evaluate(*validation_data)
        # Mimic Keras' history
        history.history = {
            'loss': [loss],
            'accuracy': [acc],
            'val_loss': [val_loss],
            'val_accuracy': [val_acc]
        }

        return history

    def evaluate(self, x_eval, y_eval, **kwargs):
        if self.coef_ is None:
            model_evaluation = [0] * 2
        else:
            loss = log_loss(y_eval, self.predict(x_eval))  # mimic keras model evaluation
            accuracy = self.score(x_eval, y_eval)
            model_evaluation = [loss, accuracy]

        return model_evaluation

    def save_weight(self, path):
        if self.coef_ is None:
            raise ValueError('The model has never been fit, coef and intercept are set to None')
        if '.h5' in path:
            logger.debug('Automatically switch file format from .h5 to .npy')
            path.replace('.h5', '.npy')
        np.save(path, self.get_weights())

    def load_weights(self, path):
        if '.h5' in path:
            logger.debug('Automatically switch file format from .h5 to .npy')
            path.replace('.h5', '.npy')
        weights = load(path)
        self.set_weight(weights)

    def get_weights(self):
        return np.concatenate((self.coef_, self.intercept_.reshape(1, 1)), axis=1)

    def set_weight(self, weights):
        self.coef_ = weights[0][:-1].reshape(1, -1)
        self.intercept_ = weights[0][-1].reshape(1)

    def save_model(self, path):
        if '.h5' in path:
            logger.debug('Automatically switch file format from .h5 to .joblib')
            path.replace('.h5', '.joblib')
        dump(self, path)

    @staticmethod
    def load_model(path):
        if '.h5' in path:
            logger.debug('Automatically switch file format from .h5 to .joblib')
            path.replace('.h5', '.joblib')
        return load(path)
