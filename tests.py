# Pytest doc: https://docs.pytest.org/en/latest/getting-started.html#create-your-first-test

# Usage:
# pytest -k "METHOD" FILE.py 
# pytest -k CLASS FILE.py

# Pdb option: run into the debugger if/when a test is failing
# pytest FILE.py --pdb

import pytest
import utils
import yaml


class TestClass:

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
        Test that the two config files are present
        and loaded with the load_cfg method
        """
        config_file = utils.load_cfg("config.yml")
        config_quick_debug_file = utils.load_cfg("config_quick_debug.yml")
        assert config_file and config_quick_debug_file


    def test_config_files(self):
        with open("config.yml", "r") as config_file:
            assert yaml.load(config_file, Loader=yaml.FullLoader)
        with open("config_quick_debug.yml", "r") as config_quick_debug_file:
            assert yaml.load(config_quick_debug_file, Loader=yaml.FullLoader)
    

    def test_config(self):
        config = utils.load_cfg("config.yml")
        assert config["experiment_name"] and config["n_repeats"] and config["scenario_params_list"]


    @pytest.mark.parametrize('expected', [
        ("my_example"),
        ])
    def test_eval(self, expected):
        with open("config.yml", "r") as config_file: 
            config = yaml.load(config_file, Loader=yaml.FullLoader)
            assert config
            assert config["experiment_name"] == expected
            assert config["n_repeats"] > 0
            assert type(config["scenario_params_list"]) == list
            assert type(config["scenario_params_list"][0]) == dict
