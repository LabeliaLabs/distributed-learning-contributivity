# -*- coding: utf-8 -*-
"""
Titanic dataset.
(inspired from: https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html)
"""

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
import requests
import pandas as pd
import keras
import os

# Init dataset-specific variables
num_classes = 2
input_shape = (6,)

# Init dataset-specific functions
def preprocess_dataset_labels(y):

    """
    Change label to categorical values
    """

    y = keras.utils.to_categorical(y,num_classes=num_classes)
    return y

def preprocess_dataset_inputs(x):

    """
    Return usable boolean values for 'Sex' atribute
    """

    x["Sex"] = [i=="Male" for i in x["Sex"]]
    return x.to_numpy()

# download train and test sets
def load_data():

    """
    Return a usable dataset
    """

    raw_dataset = pd.read_csv('https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv', index_col=False )
    x = raw_dataset.drop(['Name','Survived'], axis=1)
    x = preprocess_dataset_inputs(x)
    y = raw_dataset['Survived']
    y = y.to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    return (x_train, y_train), (x_test, y_test)

# Model structure and generation
def generate_new_model_for_dataset():

    """
    Return a deep learning model from scratch
    Source : https://www.kaggle.com/bananachips/titanic-with-keras
    """

    model = Sequential()
    model.add(Dense(256,activation="relu", input_shape=input_shape))
    model.add(keras.layers.core.Dropout(0.3))
    model.add(Dense(128,activation="relu"))
    model.add(keras.layers.core.Dropout(0.2))
    model.add(Dense(64,activation="relu"))
    model.add(Dense(2,activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    return model

# Load data
(x_train, y_train), (x_test, y_test) = load_data()
