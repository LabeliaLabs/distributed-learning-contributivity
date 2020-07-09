# -*- coding: utf-8 -*-
"""
IMDB dataset.
inspired from:
https://keras.io/examples/nlp/bidirectional_lstm_imdb/
"""

import keras
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras import models, layers
from keras.datasets import imdb

# Definition of dataset-specific parameters 
maxlen = 200 # We only keep the 200 first words of the reviews
out_of_vocabulary = 2
num_classes = 2 # 2 differents labels 
skip_top_review = 35 # We skip the 35 most frequent words
size_vocabulary = 20000 # We keep the 20000 first more frequent words in the datasets - Also the dimentionality of the input of the Embedding layer
# The others are replaced by out_of_vocabulary - A Review start by the number 1 (start_char)

# Not used here, but important in another module
input_shape = keras.Input(shape=(None,), dtype="int32")

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=size_vocabulary, skip_top=35, start_char=1, oov_char=out_of_vocabulary, index_from=3)

def preprocess_dataset_labels(y): # Do not preprocess labels here! => it's done in another script
    y = to_categorical(y, num_classes)
    return y

# Standardize size of review at maxlen words: pre-padding (with 0) and pre-truncating
def preprocess_dataset_inputs(x):
    return pad_sequences(x_train, maxlen=maxlen)

x_train = preprocess_dataset_inputs(x_train)
x_test = preprocess_dataset_inputs(x_test)

def generate_new_model_for_dataset():
    model = models.Sequential()

    model = models.Sequential()
    model.add(layers.Embedding(size_vocabulary, 128)) # Embed each integer in a 128-dimensional vector
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(64)))

    model.add(layers.Dense(2, activation="softmax"))

    #model.summary() # Debugging

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

    return model
