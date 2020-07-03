# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from keras.datasets import imdb
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPool1D, Embedding
from keras.models import Sequential
from keras import layers
import keras


# Load Data, marche que si on enleve **kwargs ==> c'est quoi?
(x_train, y_train),(x_test, y_test)=tf.keras.datasets.imdb.load_data(
    path='imdb.npz', num_words=None, skip_top=50, maxlen=None,
    seed=113, start_char=1, oov_char=2, index_from=3)

input_shape = tf.keras.Input(shape=(None,), dtype="int64")
#input_shape = (None, None)

# ATTTTTEENNNTTIIOONN
num_classes = 2 # Car necessaire dans scenario pour construire un
# objet Dataset, je ne sais pas ou il est utilise dans le code ensuite

def preprocess_dataset_labels(y):
    y = tf.keras.utils.to_categorical(y, num_classes)
    return y

batch_size = 32

max_features = 20000
embedding_dim = 128
sequence_length = 500

# Model structure and generation
def generate_new_model_for_dataset():
    """ Return a CNN model from scratch based on given batch_size"""
    
    # Sequential groups a linear stack of layer into a tf.kera.Model
    model = Sequential() # Instancie un objet de type Sequential

    # Next, we add a layer to map those vocab indices into a 
    # space of dimensionality 'embedding_dim'.
    #model.add(Embedding(max_features, embedding_dim))
    #model.add(Dropout(0.5))

    # Conv1D + global max pooling
    #model.add(Conv1D(128, 7, padding="valid", activation="relu", strides=3))
    #model.add(Conv1D(128, 7, padding="valid", activation="relu", strides=3))
    #model.add(GlobalMaxPool1D())

    # We add a vanilla hidden layer:
    #model.add(Dense(128, activation="relu"))
    #model.add(Dropout(0.5))

   # model.add(Dense(1, activation="sigmoid"))

    # Input - Layer
    model.add(layers.Dense(50, activation = "relu", input_shape=input_shape))
    # Hidden - Layers
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation = "relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation = "relu"))
    # Output- Layer
    model.add(layers.Dense(1, activation = "sigmoid"))

    model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer="adam",
            metrics=['accuracy'])

    return model
