# -*- coding: utf-8 -*-
"""
IMDB dataset.
inspired from:
https://keras.io/examples/nlp/text_classification_from_scratch/
https://builtin.com/data-science/how-build-neural-network-keras
"""

import keras
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras import models, layers
from keras.datasets import imdb

out_of_vocabulary = 2
# We keep the 10000 first more frequent words in the datasets
# Also the dimentionality of the input of the Embedding layer
size_vocabulary = 20000
# We only keep the 200 first words of the reviews
maxlen = 200
# Number of labels
num_classes = 2

# Not used here, but important another module
input_shape = keras.Input(shape=(None,), dtype="int32")

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=size_vocabulary, skip_top=35, start_char=1, oov_char=out_of_vocabulary, index_from=3)

# We skip the 35 most frequent words
# The others are replaced by 2
# A Review start by the number 1

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

    # Embed each integer in a 128-dimensional vector
    inputs = keras.Input(shape=(None,), dtype="int32")

    model = models.Sequential()
    model.add(layers.Embedding(size_vocabulary, 128))
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(64)))

    model.add(layers.Dense(2, activation="softmax"))

    #model.summary() # Debugging

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

    return model
