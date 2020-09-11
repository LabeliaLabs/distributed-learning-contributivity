# -*- coding: utf-8 -*-
"""
Titanic dataset.
(inspired from: https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html)
"""
import os
from pathlib import Path
from loguru import logger

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from . import dataset


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
        preprocess_dataset_labels,
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
    repertoire = str(path) + '/local_data/titanic/'
    if not Path(repertoire).is_dir():
        os.makedirs(repertoire)
        os.chdir(repertoire)
        logger.info('Titanic dataset not found. Downloading it...')
        raw_dataset = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv',
                                  index_col=False)
        raw_dataset.to_csv(repertoire + "titanic.csv")
    else:
        os.chdir(repertoire)

    raw_dataset = pd.read_csv(repertoire + 'titanic.csv')
    x = raw_dataset.drop('Survived', axis=1)
    x = preprocess_dataset_inputs(x)
    y = raw_dataset['Survived']
    y = y.to_numpy()

    x_train, x_test, y_train, y_test = train_test_split_global(x, y)

    return (x_train, y_train), (x_test, y_test)


# Model structure and generation
def generate_new_model_for_dataset():
    """Return a LogisticRegression Classifier"""

    clf = LogisticRegression(max_iter=10000, warm_start=1, random_state=0)
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
