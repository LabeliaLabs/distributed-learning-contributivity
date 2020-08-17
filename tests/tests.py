# -*- coding: utf-8 -*-
"""
This enables to parameterize unit tests - the tests are run by Travis each time you commit to the github repo
"""

#########
#
# Help on Tests:
#
##########

# Some usefull commands:
#
# pytest tests.py
# pytest -k TestDemoClass tests.py
# pytest -k "test_ok" tests.py

# Start the interactive Python debugger on errors or KeyboardInterrupt.
# pytest tests.py --pdb

# --collect-only, --co  only collect tests, don't execute them.
# pytest tests.py --co

# Main documentation:
# https://docs.pytest.org/en/latest/contents.html

# Gettig Started
# https://docs.pytest.org/en/latest/getting-started.html#group-multiple-tests-in-a-class

# Parametrize to generate parameters combinations
# https://docs.pytest.org/en/latest/example/parametrize.html#paramexamples

# Fixture to initialize test functions
# https://docs.pytest.org/en/latest/fixture.html

# Test architecture
# https://docs.pytest.org/en/latest/goodpractices.html#test-discovery

import sys
sys.path.append("../distributed-learning-contributivity/")


import utils
import yaml
import pytest
import numpy as np

from pathlib import Path

from tensorflow.keras.datasets import cifar10,mnist

from datasets import dataset_cifar10 as data_cf
from datasets import dataset_mnist as data_mn
import multi_partner_learning


from partner import Partner
from dataset import Dataset
from scenario import Scenario
from contributivity import Contributivity
from multi_partner_learning import MultiPartnerLearning

######
# Fixture Iterate: to generate the combination of parameters
# of the Scenario Partner MPL Dataset Objects
######

@pytest.fixture(scope='class', params=['cifar10', 'mnist'])
def iterate_dataset_name(request):
    yield request.param

@pytest.fixture(scope='class', params=[['basic', 'random'], ['advanced', [[4, 'shared'], [6, 'shared'], [4, 'specific']]]], ids=['basic','advanced'])
def iterate_samples_split_option(request):
    yield request.param


######
# Fixture Create: to generate the objects that are used in the test functions - use the 'iterate' fixtures to generate their parameters
# It's probably better to maintain their independence in order to be free to create weird objects, then give them to the test functions
######

# create_Mpl uses create_Dataset and create_Contributivity uses create_Scenario

@pytest.fixture(scope='class')
def create_Partner(iterate_dataset_name):
    """Instantiate partner object"""
    part = Partner(partner_id=0)
    dataset_name = iterate_dataset_name

    if dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        part.y_train = data_cf.preprocess_dataset_labels(y_train)
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        part.y_train = data_mn.preprocess_dataset_labels(y_train)
    yield part


@pytest.fixture(scope='class')
def create_Dataset(iterate_dataset_name):
    dataset_name = iterate_dataset_name

    if dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        input_shape = data_cf.input_shape
        num_classes = data_cf.num_classes
        preprocess_dataset_labels = data_cf.preprocess_dataset_labels
        generate_new_model_for_dataset = data_cf.generate_new_model_for_dataset
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        input_shape = data_mn.input_shape
        num_classes = data_mn.num_classes
        preprocess_dataset_labels = data_mn.preprocess_dataset_labels
        generate_new_model_for_dataset = data_mn.generate_new_model_for_dataset

    dataset = Dataset(
            dataset_name=dataset_name,
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


@pytest.fixture(scope='class')
def create_MultiPartnerLearning(create_Dataset):
    data = create_Dataset

    # Create partners_list (this is not a fixture):
    part_list = create_partners_list(data.name, 3)

    mpl = MultiPartnerLearning(
            partners_list=part_list,
            epoch_count=2,
            minibatch_count=2,
            dataset=data,
            multi_partner_learning_approach="fedavg",
            aggregation_weighting="uniform",
            is_early_stopping=True,
            is_save_data=False,
            save_folder=""
            )

    yield mpl


@pytest.fixture(scope='class')
def create_Scenario(iterate_dataset_name, iterate_samples_split_option):

    dataset_name = iterate_dataset_name
    samples_split_option = iterate_samples_split_option

    params = {"dataset_name": dataset_name}
    params.update({"partners_count": 3, "amounts_per_partner": [0.2, 0.5, 0.3], "samples_split_option": samples_split_option, "corrupted_datasets": ["not_corrupted"]*3})
    params.update({"methods":["Shapley values", "Independent scores"], "multi_partner_learning_approach":"fedavg", "aggregation_weighting": "uniform"})
    params.update({"gradient_updates_per_pass_count": 5, "epoch_count": 2, "minibatch_count": 2, "is_early_stopping": True})
    params.update({"is_quick_demo": False})


    full_experiment_name = "unit-test-pytest"
    experiment_path = Path.cwd() / "experiments" / full_experiment_name

    # scenar.dataset object is created inside the Scenario constructor
    scenar = Scenario(
            params=params,
            experiment_path=experiment_path,
            scenario_id=0,
            n_repeat=1
            )

    scenar.partners_list = create_partners_list(scenar.dataset.name, scenar.partners_count)

    scenar.mpl = multi_partner_learning.init_multi_partner_learning_from_scenario(scenario=scenar, is_save_data=True)

    yield scenar


@pytest.fixture(scope='class')
def create_Contributivity(create_Scenario):
    scenar = create_Scenario
    contri = Contributivity(scenario=scenar)

    yield contri


######
#
# Sub-function of fixture create to generate a sub-object without a call to another fixture create
#
######

# This is not a pytest.fixture!
def create_partners_list(dataset_name, partners_count):
    partners_list = []

    for i in range(partners_count):
        part = Partner(partner_id=i)

        if dataset_name == "cifar10":
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            part.y_train = data_cf.preprocess_dataset_labels(y_train)
        if dataset_name == "mnist":
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            part.y_train = data_mn.preprocess_dataset_labels(y_train)
        partners_list.append(part)

    return partners_list


######
#
# Tests modules with Objects
#
######


class Test_Partner:

    def test_corrupt_labels_type(self):
        """partner.y_train should be a numpy.ndarray"""
        with pytest.raises(TypeError):
            part = Partner(partner_id=0)
            part.corrupt_labels()


    def test_corrupt_labels_type_elem(self, create_Partner):
        """corrupt_labels raise TypeError if partner.y_train isn't float32"""
        with pytest.raises(TypeError):
            part = create_Partner
            part.y_train = part.y_train.astype("float64")
            part.corrupt_labels(part)


    def test_shuffle_labels_type(self):
        """shuffle_labels should be a numpy.ndarray"""
        with pytest.raises(TypeError):
            part = Partner(partner_id=0)
            part.shuffle_labels(part)


    def test_shuffle_labels_type_elem(self, create_Partner):
        """shuffle_labels raise TypeError if partner.y_train isn't float32"""
        with pytest.raises(TypeError):
            part = create_Partner
            part.y_train = part.y_train.astype("float64")
            part.shuffle_labels(part)


class Test_Dataset:

    def test_generate_new_model(self, create_Dataset):
        assert create_Dataset.name in {"cifar10","mnist"}


    def test_train_val_split(self, create_Dataset):
        """train_val_split is used once, just after Dataset being instantiated - this is written to prevent its call from another place"""
        data = create_Dataset
        data.x_val = data.x_train[::]
        with pytest.raises(Exception):
            data.train_val_split()


class Test_Mpl:

    def test_Mpl(self, create_MultiPartnerLearning):
        mpl = create_MultiPartnerLearning
        assert type(mpl) == MultiPartnerLearning


class Test_Scenario:

    def test_scenar(self, create_Scenario):
        assert type(create_Scenario) == Scenario

    class Test_instantiate_scenario_partners:

        def test_raiseException(self, create_Scenario):
            scenar = create_Scenario
            with pytest.raises(Exception):
                scenar.instantiate_scenario_partners()


class Test_Contributivity:

    def test_Contributivity(self, create_Contributivity):
        contri = create_Contributivity
        assert type(contri) == Contributivity


######
#
# Test supported datasets
#
######

@pytest.fixture(scope='class')
def create_cifar10_x():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = data_cf.preprocess_dataset_inputs(x_train)
    x_test = data_cf.preprocess_dataset_inputs(x_test)
    return x_train, x_test


@pytest.fixture(scope='class')
def create_mnist_x():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = data_mn.preprocess_dataset_inputs(x_train)
    x_test = data_mn.preprocess_dataset_inputs(x_test)
    return x_train, x_test


class Test_dataset_cifar10:

    def test_preprocess_dataset_inputs_type(self, create_cifar10_x):
        """x_train and x_test type should be float32"""
        x_train, x_test = create_cifar10_x
        assert x_train.dtype == "float32"
        assert x_test.dtype == "float32"

    def test_preprocess_dataset_inputs_activation(self, create_cifar10_x):
        """x_train and x_test activation should be >=0 and <=1"""
        x_train, x_test = create_cifar10_x
        greater_than_0 = not False in np.greater_equal(x_train, 0)
        lower_than_1 = not True in np.greater(x_train, 1)
        assert (greater_than_0 and lower_than_1)

        greater_than_0 = not False in np.greater_equal(x_test, 0)
        lower_than_1 = not True in np.greater(x_test, 1)
        assert (greater_than_0 and lower_than_1)

    def test_inputs_shape(self, create_cifar10_x):
        """the shape of the elements of x_train and x_test should be equal to input_shape"""
        x_train, x_test = create_cifar10_x
        assert x_train.shape[1:] == data_cf.input_shape
        assert x_test.shape[1:] == data_cf.input_shape


class Test_dataset_mnist:

    def test_preprocess_dataset_inputs_type(self, create_mnist_x):
        """x_train and x_test type should be float32"""
        x_train, x_test = create_mnist_x
        assert x_train.dtype == "float32"
        assert x_test.dtype == "float32"

    def test_preprocess_dataset_inputs_activation(self, create_mnist_x):
        """x_train and x_test activation should be >=0 and <=1"""
        x_train, x_test = create_mnist_x
        greater_than_0 = not False in np.greater_equal(x_train, 0)
        lower_than_1 = not True in np.greater(x_train, 1)
        assert (greater_than_0 and lower_than_1)

        greater_than_0 = not False in np.greater_equal(x_test, 0)
        lower_than_1 = not True in np.greater(x_test, 1)
        assert (greater_than_0 and lower_than_1)


    def test_inputs_shape(self, create_mnist_x):
        """the shape of the elements of x_train and x_test should be equal to input_shape"""
        x_train, x_test = create_mnist_x
        assert x_train.shape[1:] == data_mn.input_shape
        assert x_test.shape[1:] == data_mn.input_shape


#####
#
# Test Demo and config files
#
######

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
        config_file = utils.load_cfg("../config.yml")
        config_quick_debug_file = utils.load_cfg("../config_quick_debug.yml")
        assert config_file and config_quick_debug_file

    def test_load_config_files(self):
        """
        Check if the two config files are present
        and loaded with the load method
        """
        with open("../config.yml", "r") as config_file:
            assert yaml.load(config_file, Loader=yaml.FullLoader)
        with open("../config_quick_debug.yml", "r") as config_quick_debug_file:
            assert yaml.load(config_quick_debug_file, Loader=yaml.FullLoader)
