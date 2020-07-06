# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb

(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)

data = np.concatenate((training_data, testing_data), axis=0)

targets = np.concatenate((training_targets, testing_targets), axis=0)

def preprocess_dataset_inputs(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i in range(len(sequences)):
        j = 0
        while j < len(sequences[i]):
            results[i][j] = sequences[i][j]
            j = j + 1
    return results

def preprocess_dataset_labels(y):
    y = keras.utils.to_categorical(y, num_classes)
    return y

#input_shape = keras.Input(shape=(25000,10000), dtype="int64")
input_shape = (25000,10000)
num_classes = 2
data = preprocess_dataset_inputs(data)

targets = np.array(targets).astype("int64")

x_test = data[:25000]
y_test = targets[:25000]
x_train = data[25000:]
y_train = targets[25000:]

print("x_train:", x_train[0:2])
print("x_train shape:", x_train.shape)
print("y_train:", x_train[0:2])
print("y_train shape:", y_train.shape)

def generate_new_model_for_dataset():

    model = models.Sequential()

    # Input - Layer
    model.add(layers.Dense(50, activation = "relu", input_shape=input_shape))
    # Hidden - Layers
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation = "relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation = "relu"))
    # Output- Layer
    model.add(layers.Dense(1, activation = "sigmoid"))
    model.summary()


    model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
    )

    return model
