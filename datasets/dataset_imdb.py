# -*- coding: utf-8 -*-
"""
IMDB dataset.
inspired from:
https://keras.io/examples/nlp/text_classification_from_scratch/
https://builtin.com/data-science/how-build-neural-network-keras
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb

nb_words = 3000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=nb_words)
# Argument num_words => only the num_words first more frequent words are used
# The others are replaced by 2
# A Review start by the number 1
# The longest review is less than 3000 words long

def preprocess_dataset_inputs(data):
    data_list = []
    for sample in data:
        sample += [2] * (nb_words - len(sample)) # 2 is the over_of_vocabulary number
        data_list.append(sample)

    data_array = np.array(data_list)
    data_array = data_array.reshape(data_array.shape[0], data_array.shape[1], 1)

    return data_array

def preprocess_dataset_labels(y): # Do not preprocess labels here! => it's done in another script
    y = keras.utils.to_categorical(y, num_classes)

    return y

# Padding
nb_words = 3000

input_shape = (nb_words,1)

# Number of labels
num_classes = 2

x_train = preprocess_dataset_inputs(x_train)
x_test = preprocess_dataset_inputs(x_test)

def generate_new_model_for_dataset():
    model = models.Sequential()

    # Input - Layer
    model.add(layers.Dense(50, activation = "relu", input_shape=input_shape))

    # Hidden - Layers
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation = "relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation = "relu"))
    model.add(layers.Flatten()) # Adapt shape of output

    # Output- Layer
    model.add(layers.Dense(2, activation = "sigmoid")) # 2 => One-Hot Encoding!! with categorical_crossentropy

    # model.summary() # Debugging

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

    return model
