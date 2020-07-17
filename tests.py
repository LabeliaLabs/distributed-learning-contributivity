# Pytest doc: https://docs.pytest.org/en/latest/getting-started.html
#create-your-first-test
# pytest tests.py
# pytest -k TestDemoClass tests.py
# pytest -k "test_ok" tests.py
# pytest tests.py --pdb

import utils
import yaml

import pytest

import numpy as np

from tensorflow.keras.datasets import cifar10
from datasets import dataset_cifar10 as cf10
from partner import Partner
# ici rassembler bouts de code pour creer une liste
# de scenario (config!) dont on se servira dans les tests

class Test_partner:

    def test_corrupt_labels_type(self):
        with pytest.raises(TypeError):
            part = Partner(partner_id=0)
            part.corrupt_labels(part)

    def test_corrupt_labels_type_elem(self):
        with pytest.raises(TypeError):
            part = Partner(partner_id=0)
            (x_train, y_train), (x_test, y_test) = cf10.cifar10.load_data()
            part.y_train = cf10.preprocess_dataset_labels(y_train)
            part.y_train = part.y_train.astype("float64")
            part.corrupt_labels(part)

    def test_shuffle_labels_type(self):
        with pytest.raises(TypeError):
            part = Partner(partner_id=0)
            part.shuffle_labels(part)

    def test_shuffle_labels_type_elem(self):
        with pytest.raises(TypeError):
            part = Partner(partner_id=0)
            (x_train, y_train), (x_test, y_test) = cf10.cifar10.load_data()
            part.y_train = cf10.preprocess_dataset_labels(y_train)
            part.y_train = part.y_train.astype("float64")
            part.shuffle_labels(part)


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = cf10.preprocess_dataset_inputs(x_train)

class Test_dataset_cifar10:

    def test_preprocess_dataset_inputs_type(self):
        assert x_train.dtype == "float32"

    def test_preprocess_dataset_inputs_activation(self):
        greater_than_0 = not False in np.greater_equal(x_train, 0)
        lower_than_1 = not True in np.greater(x_train, 1)
        assert (greater_than_0 and lower_than_1)

    def test_inputs_shape(self):
        assert x_train.shape[1:] == cf10.input_shape

class TestDemoClass:

    def test_ok(self):
        """
        Demo test
        """
        ok = "ok"
        assert "ok" in ok

    def test_ko(self):
        """
        Demo test 2
        """
        ko = "ko"
        assert "ok" not in ko

    def test_load_cfg(self):
        """
        Check if the two config files are present
        and loaded with the load_cfg method
        """
        config_file = utils.load_cfg("config.yml")
        config_quick_debug_file = utils.load_cfg("config_quick_debug.yml")
        assert config_file and config_quick_debug_file

    def test_load_config_files(self):
        """
        Check if the two config files are present
        and loaded with the load method
        """
        with open("config.yml", "r") as config_file:
            assert yaml.load(config_file, Loader=yaml.FullLoader)
        with open("config_quick_debug.yml", "r") as config_quick_debug_file:
            assert yaml.load(config_quick_debug_file, Loader=yaml.FullLoader)
