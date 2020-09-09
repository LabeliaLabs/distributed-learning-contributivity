# -*- coding: utf-8 -*-
"""
Titanic dataset.
(inspired from: https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html)
"""

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
import numpy as np
import keras
from sklearn.linear_model import LogisticRegression

# Init dataset-specific variables
num_classes = 1
input_shape = (26,)


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


# Download train and test sets
def load_data():
    """Return a usable dataset"""

    url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    raw_dataset = pd.read_csv(url, index_col=False)
    x = raw_dataset.drop('Survived', axis=1)
    x = preprocess_dataset_inputs(x)
    y = raw_dataset['Survived']
    y = y.to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.1,
        random_state=42
        )

    return (x_train, y_train), (x_test, y_test)


# Model structure and generation
def generate_new_model_for_dataset():
    """Return a LogisticRegression Classifier"""

    clf = LogisticRegression(max_iter=10000, warm_start=1, random_state=0)
    clf.classes_ = np.array([0, 1])
    clf.metrics_names = ["log_loss", "Accuracy"]  # Mimic Keras's
    return clf


# Load data
(x_train, y_train), (x_test, y_test) = load_data()
