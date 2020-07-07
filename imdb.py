import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb
from keras.layers import Flatten


nb_words = 2000
nb_movies = 1000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=nb_words)
data = np.concatenate((x_train, x_test), axis=0)

def preprocess_dataset_inputs(data):
    data_list = []
    for sample in data[:nb_movies]:     #on se limite a un nombre nb_movies de review
        sample += [2] * (nb_words - len(sample))
        data_list.append(sample)
    data_array = np.array(data_list)
    data_array = data_array.reshape(data_array.shape[0], data_array.shape[1], 1)

    return data_array

def aaaapreprocess_dataset_inputs(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i in range(len(sequences)):
        j = 0
        while j < len(sequences[i]):
            results[i][j] = sequences[i][j]
            j = j + 1
    return results

def preprocess_dataset_labels(y): #ne pas appeler dans ce scripte: appeler autre part deja
    y = keras.utils.to_categorical(y, num_classes)
    return y

#input_shape = keras.Input(shape=(25000,10000), dtype="int64")
input_shape = (nb_words,1)
num_classes = 2

#data = preprocess_dataset_inputs(data)
x_train = preprocess_dataset_inputs(data)
x_test = preprocess_dataset_inputs(data[nb_movies:])

print("y_train:", y_train[0:2])
print("y_train shape:", y_train.shape)

y_train = preprocess_dataset_labels(y_train[:nb_movies])
y_test = preprocess_dataset_labels(y_test[:nb_movies])

y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)

#y_test = y_test.flatten()
#y_test = y_test.reshape(y_test.shape[0], 1)

#y_train = y_train.flatten()
#y_train = y_train.reshape(y_train.shape[0], 1)

print("x_train:", x_train[0:2])
print("x_train shape:", x_train.shape)
print("y_train:", y_train[0:2])
print("y_train shape:", y_train.shape)

print("x_test:", x_test[0:2])
print("x_test shape:", x_test.shape)
print("y_test:", y_test[0:2])
print("y_test shape:", y_test.shape)

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

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

    return model

print("Creation du model et compilation:")

model = generate_new_model_for_dataset()

print("OK Creation Model et Compilation:")
print("Fit Du Model:")
results = model.fit(x_train, y_train, epochs= 2, batch_size = 50, validation_data = (x_test,y_test))
print("Fin Fit du Model:")
print(np.mean(results.history["val_accuracy"]))

