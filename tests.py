# Pytest doc: https://docs.pytest.org/en/latest/getting-started.html#create-your-first-test

# pytest tests.py
# pytest -k TestDemoClass tests.py
# pytest -k "test_ok" tests.py
# pytest tests.py --pdb

import utils
import yaml

import pytest

import numpy as np

from tensorflow.keras.datasets import cifar10,mnist
from datasets import dataset_cifar10 as data_cf
from datasets import dataset_mnist as data_mn
from partner import Partner
from dataset import Dataset
from scenario import Scenario
from multi_partner_learning import MultiPartnerLearning
from pathlib import Path


@pytest.fixture(scope="module", params=["a","b"])
def ab(request):
    yield request.param

@pytest.fixture(scope="module", params=["c","d"])
def cd(request,ab):
    yield ab + request.param

def test(cd):
    assert cd in {"ac","ad","bc","bd"}


@pytest.fixture(scope="class", params=["cifar10","mnist"])
def create_partner(request):
    """Instantiate partner object"""
    part = Partner(partner_id=0)
    if request.param == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        part.y_train = data_cf.preprocess_dataset_labels(y_train)
    if request.param == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        part.y_train = data_mn.preprocess_dataset_labels(y_train)
    yield part

class Test_partner:

    def test_corrupt_labels_type(self):
        """partner.y_train should be a numpy.ndarray"""
        with pytest.raises(TypeError):
            part = Partner(partner_id=0)
            part.corrupt_labels()

    def test_corrupt_labels_type_elem(self, create_partner):
        """corrupt_labels raise TypeError if partner.y_train isn't float32"""
        with pytest.raises(TypeError):
            part = create_partner
            part.y_train = part.y_train.astype("float64")
            part.corrupt_labels(part)

    def test_shuffle_labels_type(self):
        """shuffle_labels should be a numpy.ndarray"""
        with pytest.raises(TypeError):
            part = Partner(partner_id=0)
            part.shuffle_labels(part)

    def test_shuffle_labels_type_elem(self, create_partner):
        """shuffle_labels raise TypeError if partner.y_train isn't float32"""
        with pytest.raises(TypeError):
            part = create_partner
            part.y_train = part.y_train.astype("float64")
            part.shuffle_labels(part)

@pytest.fixture(scope="class", params=["cifar10","mnist"])
def create_Dataset(request):
    """"""
    name = request.param
    if name == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        input_shape = data_cf.input_shape
        num_classes = data_cf.num_classes
        preprocess_dataset_labels = data_cf.preprocess_dataset_labels
        generate_new_model_for_dataset = data_cf.generate_new_model_for_dataset
    if name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        input_shape = data_mn.input_shape
        num_classes = data_mn.num_classes
        preprocess_dataset_labels = data_mn.preprocess_dataset_labels
        generate_new_model_for_dataset = data_mn.generate_new_model_for_dataset

    dataset = Dataset(
            dataset_name=name,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            input_shape=input_shape,
            num_classes=num_classes,
            preprocess_dataset_labels=preprocess_dataset_labels,
            generate_new_model_for_dataset=generate_new_model_for_dataset
            )
    yield dataset


class Test_Dataset:

    def test_generate_new_model(self, create_Dataset):
        assert create_Dataset.name in {"cifar10","mnist"}

@pytest.fixture
def create_partner_list(create_partner):
    yield [create_partner] * 3


@pytest.fixture
def create_MultiPartnerLearning(create_Dataset, create_partner_list):
    data = create_Dataset
    part_list = create_partner_list
    mpl = MultiPartnerLearning(
            partners_list=part_list,
            epoch_count=2,
            minibatch_count=2,
            dataset=data,
            multi_partner_learning_approach="fedavg",
            aggregation_weighting="uniform",
            is_early_stopping=True,
            is_save_data=False,
            save_folder="",
            )
    yield mpl

def test_mpl(create_MultiPartnerLearning):
    assert type(create_MultiPartnerLearning) == MultiPartnerLearning


@pytest.fixture
def create_scenario(create_MultiPartnerLearning, create_Dataset, create_partner_list):
    params = {"dataset_name": "cifar10", "partners_count":3, "amounts_per_partner": [0.2, 0.5, 0.3], "samples_split_option": ["basic","random"], "multi_partner_learning_aproach":"fedavg", "aggregation_weighting": "uniform", "methods": ["Shapley values", "Independent scores"], "gradient_updates_per_pass_count": 5}

    full_experiment_name = "unit-test-pytest"
    experiment_path = Path.cwd() / "experiments" / full_experiment_name

    scenar = Scenario(
            params=params,
            experiment_path=experiment_path,
            scenario_id=0,
            n_repeat=1
            )

    scenar.mpl = create_MultiPartnerLearning
    scenar.dataset

    yield scenar

def test_scenar(create_scenario):
    assert type(create_scenario) == Scenario

@pytest.fixture(scope="class")
def create_cifar10_x_train():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = data_cf.preprocess_dataset_inputs(x_train)
    return x_train

class Test_dataset_cifar10:

    def test_preprocess_dataset_inputs_type(self, create_cifar10_x_train):
        """x_train type should be float32"""
        assert create_cifar10_x_train.dtype == "float32"

    def test_preprocess_dataset_inputs_activation(self, create_cifar10_x_train):
        """x_train activation should be >=0 and <=1"""
        x_train = create_cifar10_x_train
        greater_than_0 = not False in np.greater_equal(x_train, 0)
        lower_than_1 = not True in np.greater(x_train, 1)
        assert (greater_than_0 and lower_than_1)

    def test_inputs_shape(self, create_cifar10_x_train):
        """the shape of the elements of x_train is input_shape"""
        x_train = create_cifar10_x_train
        assert x_train.shape[1:] == data_cf.input_shape


@pytest.fixture(scope="class")
def create_mnist_x_train():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = data_mn.preprocess_dataset_inputs(x_train)
    return x_train

class Test_dataset_mnist:

    def test_preprocess_dataset_inputs_type(self, create_mnist_x_train):
        """x_train type should be float32"""
        assert create_mnist_x_train.dtype == "float32"

    def test_preprocess_dataset_inputs_activation(self, create_mnist_x_train):
        """x_train activation should be >=0 and <=1"""
        x_train = create_mnist_x_train
        greater_than_0 = not False in np.greater_equal(x_train, 0)
        lower_than_1 = not True in np.greater(x_train, 1)
        assert (greater_than_0 and lower_than_1)

    def test_inputs_shape(self, create_mnist_x_train):
        """the shape of the elements of x_train is input_shape"""
        x_train = create_mnist_x_train
        assert x_train.shape[1:] == data_mn.input_shape


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
