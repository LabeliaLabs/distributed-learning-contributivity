# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:33:15 2019

@author: RGOUSSAULT
inspired from: https://keras.io/examples/mnist_cnn/
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

import sys
sys.path.append("..")
import constants
import utils


#%% Constants

epochs = 10
nodes_count = 10


#%% Load data

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Shuffle train data
train_idx = np.arange(len(y_train))
np.random.shuffle(train_idx)
train_idx_idx_list = np.array_split(train_idx, nodes_count)

# Shuffle test data
test_idx = np.arange(len(y_test))
np.random.shuffle(test_idx)
test_idx_idx_list = np.array_split(test_idx, nodes_count)


x_train = x_train[train_idx_idx_list[0], :]
x_test = x_test[test_idx_idx_list[0]]

y_train = y_train[
    train_idx_idx_list[0],
]
y_test = y_test[test_idx_idx_list[0]]


#%% Preprocess data
x_train = utils.preprocess_input(x_train)
x_test = utils.preprocess_input(x_test)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, constants.NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, constants.NUM_CLASSES)

# Plot first image
first_image = x_train[0, :, :, 0] * 255
plt.gray()
plt.imshow(first_image)

# Split to get validation dataset
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=42
)


#%% Build model

model = utils.generate_new_cnn_model()
print(model.summary())


#%% Train

history = model.fit(
    x_train,
    y_train,
    batch_size=constants.BATCH_SIZE,
    epochs=epochs,
    verbose=1,
    validation_data=(x_val, y_val),
)


#%% Plot history

plt.figure()
plt.plot(history.history["loss"], "+-")
plt.plot(history.history["val_loss"], "+-")
plt.title("Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(("Train", "Val"))
plt.show()

plt.figure()
plt.plot(history.history["accuracy"], "+-")
plt.plot(history.history["val_accuracy"], "+-")
plt.title("Accuracy")
plt.ylabel("Acc")
plt.xlabel("Epoch")
plt.legend(("Train", "Val"))
plt.show()


#%% Evaluation

score = model.evaluate(x_test, y_test, verbose=0)
print("\nTest loss:", score[0])
print("Test accuracy:", score[1])
