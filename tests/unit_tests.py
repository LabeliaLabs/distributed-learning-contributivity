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

# -v run the tests in verbose mode, outputting one line per test
# pytest -v tests.py

# A "test_" prefix in classes and methods is needed to make a test discoverable by pytest

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

from pathlib import Path

import numpy as np
import pytest
import yaml

from mplc import constants, utils
from mplc.contributivity import Contributivity
from mplc.dataset import Mnist, Cifar10, Imdb, Titanic, Esc50
from mplc.mpl_utils import UniformAggregator
from mplc.multi_partner_learning import FederatedAverageLearning
from mplc.partner import Partner
from mplc.scenario import Scenario


######
# These are outdated comments, but they
# Fixture Create: to generate the objects that are used in the test functions,
#  use the 'iterate' fixtures to generate their parameters.
# It's probably better to maintain their independence in order
# to be free to create weird objects, then give them to the test functions.
######

# create_Mpl uses create_Dataset and create_Contributivity uses create_Scenario

@pytest.fixture(scope="class", params=(Mnist, Cifar10, Titanic, Imdb, Esc50))
def create_all_datasets(request):
    return request.param()


@pytest.fixture(scope="class")
def create_MultiPartnerLearning(create_all_datasets):
    data = create_all_datasets
    # Create partners_list (this is not a fixture):
    scenario = Scenario(3, [0.3, 0.3, 0.4], dataset=data)
    mpl = FederatedAverageLearning(
        scenario,
        epoch_count=2,
        minibatch_count=2,
        dataset=data,
        aggregation=UniformAggregator,
        is_early_stopping=True,
        is_save_data=False,
        save_folder="",
    )

    yield mpl


@pytest.fixture(scope="class",
                params=((Mnist, ["basic", "random"]),
                        (Mnist, ["advanced", [[4, "shared"], [6, "shared"], [4, "specific"]]]),
                        (Cifar10, ["basic", "random"]),
                        (Cifar10, ["advanced", [[4, "shared"], [6, "shared"], [4, "specific"]]])),
                ids=['Mnist - basic', 'Mnist - advanced', 'Cifar10 - basic', 'Cifar10 - advanced'])
def create_Scenario(request):
    dataset = request.param[0]()
    samples_split_option = request.param[1]

    params = {"dataset": dataset}
    params.update(
        {
            "partners_count": 3,
            "amounts_per_partner": [0.2, 0.5, 0.3],
            "samples_split_option": samples_split_option,
            "corrupted_datasets": ["not_corrupted"] * 3,
        }
    )
    params.update(
        {
            "methods": ["Shapley values", "Independent scores"],
            "multi_partner_learning_approach": "fedavg",
            "aggregation": "uniform",
        }
    )
    params.update(
        {
            "gradient_updates_per_pass_count": 5,
            "epoch_count": 2,
            "minibatch_count": 2,
            "is_early_stopping": True,
        }
    )
    params.update({"init_model_from": "random_initialization"})
    params.update({"is_quick_demo": False})

    full_experiment_name = "unit-test-pytest"
    experiment_path = (
            Path.cwd() / constants.EXPERIMENTS_FOLDER_NAME / full_experiment_name
    )

    # scenar.dataset object is created inside the Scenario constructor
    scenar = Scenario(
        **params, experiment_path=experiment_path, scenario_id=0, repeats_count=1
    )

    scenar.mpl = scenar.multi_partner_learning_approach(scenar, is_save_data=True)

    scenar.instantiate_scenario_partners()
    # Split data according to scenario and then pre-process successively...
    # ... train data, early stopping validation data, test data
    if scenar.samples_split_type == "basic":
        scenar.split_data()
    elif scenar.samples_split_type == "advanced":
        scenar.split_data_advanced()
    scenar.compute_batch_sizes()
    scenar.data_corruption()

    return scenar


@pytest.fixture(scope="class")
def create_Contributivity(create_Scenario):
    scenar = create_Scenario
    contri = Contributivity(scenario=scenar)

    return contri


######
#
# Sub-function of fixture create to generate a sub-object without a call to another fixture create
#
######

######
#
# Tests modules with Objects
#
######

class Test_Scenario:
    def test_scenar(self, create_Scenario):
        assert type(create_Scenario) == Scenario

    def test_raiseException(self, create_Scenario):
        scenar = create_Scenario
        with pytest.raises(Exception):
            scenar.instantiate_scenario_partners()

    def test_corrupt_labels_type(self):
        """partner.y_train should be a numpy.ndarray"""
        with pytest.raises(TypeError):
            part = Partner(partner_id=0)
            part.corrupt_labels()

    def test_corrupt_labels_type_elem(self, create_Scenario):
        """corrupt_labels raise TypeError if partner.y_train isn't float32"""
        scenar = create_Scenario
        with pytest.raises(TypeError):
            part = scenar.partners_list[0]
            part.y_train = part.y_train.astype("float64")
            part.corrupt_labels(part)

    def test_shuffle_labels_type(self):
        """shuffle_labels should be a numpy.ndarray"""
        with pytest.raises(TypeError):
            part = Partner(partner_id=0)
            part.shuffle_labels(part)

    def test_shuffle_labels_type_elem(self, create_Scenario):
        """shuffle_labels raise TypeError if partner.y_train isn't float32"""
        scenar = create_Scenario
        with pytest.raises(TypeError):
            part = scenar.partners_list[0]
            part.y_train = part.y_train.astype("float64")
            part.shuffle_labels(part)


class Test_Mpl:
    def test_Mpl(self, create_MultiPartnerLearning):
        mpl = create_MultiPartnerLearning
        assert type(mpl) == FederatedAverageLearning


class Test_Contributivity:
    def test_Contributivity(self, create_Contributivity):
        contri = create_Contributivity
        assert type(contri) == Contributivity


######
#
# Test supported datasets
#
######

class Test_Dataset:

    def test_train_split_global(self, create_all_datasets):
        """train_val_split is used once, just after Dataset being instantiated
         - this is written to prevent its call from another place"""
        data = create_all_datasets
        assert len(data.x_val) < len(data.x_train)
        assert len(data.x_test) < len(data.x_train)
        with pytest.raises(Exception):
            data.train_val_split_global()

    def test_local_split(self, create_all_datasets):
        data = create_all_datasets
        x_train, x_val, y_train, y_val = data.train_val_split_local(data.x_train, data.y_train)
        assert len(x_train) == len(y_train)
        assert len(x_val) == len(y_val)
        x_train, x_test, y_train, y_test = data.train_val_split_local(data.x_train, data.y_train)
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)

    def test_data_shape(self, create_all_datasets):
        data = create_all_datasets
        assert len(data.x_train) == len(data.y_train), "Number of train label is not equal to the number of data"
        assert len(data.x_val) == len(data.y_val), "Number of val label is not equal to the number of data"
        assert len(data.x_test) == len(data.y_test), "Number of test label is not equal to the number of data"

        if data.num_classes > 2:
            assert data.y_train[np.random.randint(len(data.y_train))].shape == (data.num_classes,)
            assert data.y_val[np.random.randint(len(data.y_val))].shape == (data.num_classes,)
            assert data.y_test[np.random.randint(len(data.y_test))].shape == (data.num_classes,)
        assert data.x_train[np.random.randint(len(data.x_train))].shape == data.input_shape
        assert data.x_test[np.random.randint(len(data.x_test))].shape == data.input_shape
        assert data.x_val[np.random.randint(len(data.x_val))].shape == data.input_shape

    def test_generate_new_model(self, create_all_datasets):
        dataset = create_all_datasets
        model1 = dataset.generate_new_model()
        assert callable(model1.fit), ".fit() method is required for model"
        assert callable(model1.evaluate), ".evaluate() method is required for model"
        assert callable(model1.save_weights), ".save_weights() method is required for model"
        assert callable(model1.load_weights), ".load_weights() method is required for model"
        assert callable(model1.get_weights), ' .get_weights() method is required for model'
        assert callable(model1.set_weights), ".set_weights() method is required for model"


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
