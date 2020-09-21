# -*- coding: utf-8 -*-

import tensorflow as tf
from keras import layers
from keras.datasets import imdb
from keras.models import Sequential

# Load Data
from mplc import dataset


def generate_new_dataset():
    attempts = 0
    while True:
        try:
            (x_train, y_train), (x_test, y_test) = imdb.load_data(
                path='imdb.npz', num_words=None, skip_top=50, maxlen=None,
                seed=113, start_char=1, oov_char=2, index_from=3)
            break
        except (HTTPError, URLError) as e:
            if hasattr(e, 'code'):
                temp = e.code
            else:
                temp = e.errno
            logger.debug(
                f'URL fetch failure on '
                f'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz : '
                f'{temp} -- {e.reason}')
            if attempts < constants.NUMBER_OF_DOWNLOAD_ATTEMPTS:
                sleep(2)
                attempts += 1
            else:
                raise

    input_shape = tf.keras.Input(shape=(None,), dtype="int64")
    max_features = 20000
    embedding_dim = 128
    sequence_length = 500

    # 2 Classes: review positive or negative
    num_classes = 2

    # Pre-process inputs
    x_train = preprocess_dataset_inputs(x_train)
    x_test = preprocess_dataset_inputs(x_test)

    y_train = preprocess_dataset_labels(y_train)
    y_test = preprocess_dataset_labels(y_test)

    dataset_obj = dataset.Dataset(
        "imdb",
        x_train,
        x_test,
        y_train,
        y_test,
        input_shape,
        num_classes,
        generate_new_model_for_dataset,
        train_val_split_global,
        train_test_split_local,
        train_val_split_local
    )
    return dataset_obj


def preprocess_dataset_labels(y):
    y = tf.keras.utils.to_categorical(y, num_classes)
    return y





# Model structure and generation
def generate_new_model_for_dataset():
    """ Return a CNN model from scratch based on given batch_size"""

    # Sequential groups a linear stack of layer into a tf.kera.Model
    model = Sequential()  # Instancie un objet de type Sequential

    # Next, we add a layer to map those vocab indices into a 
    # space of dimensionality 'embedding_dim'.
    # model.add(Embedding(max_features, embedding_dim))
    # model.add(Dropout(0.5))

    # Conv1D + global max pooling
    # model.add(Conv1D(128, 7, padding="valid", activation="relu", strides=3))
    # model.add(Conv1D(128, 7, padding="valid", activation="relu", strides=3))
    # model.add(GlobalMaxPool1D())

    # We add a vanilla hidden layer:
    # model.add(Dense(128, activation="relu"))
    # model.add(Dropout(0.5))

    # model.add(Dense(1, activation="sigmoid"))

    # Input - Layer
    model.add(layers.Dense(50, activation="relu", input_shape=input_shape))
    # Hidden - Layers
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    # Output- Layer
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer="adam",
                  metrics=['accuracy'])

    return model
