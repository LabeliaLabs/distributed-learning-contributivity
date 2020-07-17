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

class Test_dataset_cifar10:

    def test_preprocess_dataset_inputs_type(self):
        x = cf10.preprocess_dataset_inputs(np.arange(20.))
        assert x.dtype == "float32"

    def test_preprocess_dataset_inputs_activation(self):
        x= cf10.preprocess_dataset_inputs(np.arange(20.))
        assert all( c <= 1 and c >= 0 for c in x)

    def test_inputs_shape(self):
        (x_train, y_train), (x_test, y_test) = cf10.cifar10.load_data()
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
