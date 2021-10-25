# -*- coding: utf-8 -*-
"""
This enables to parameterize a desired scenario to mock a multi-partner ML project.
"""

import datetime
import os
import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder

from mplc.multi_partner_learning import MULTI_PARTNER_LEARNING_APPROACHES
from mplc.multi_partner_learning.utils import AGGREGATORS, Aggregator
from . import contributivity, constants, utils
from . import dataset as dataset_module
from .corruption import Corruption, NoCorruption, IMPLEMENTED_CORRUPTION, Duplication
from .partner import Partner
from .splitter import Splitter, IMPLEMENTED_SPLITTERS


class Scenario:
    def __init__(
            self,
            partners_count,
            amounts_per_partner,
            active_partners_count=1,
            dataset=constants.MNIST,
            dataset_proportion=1,
            samples_split_option='random',
            corruption_parameters=None,
            init_model_from="random_initialization",
            multi_partner_learning_approach="fedavg",
            aggregation="data-volume",
            gradient_updates_per_pass_count=constants.DEFAULT_GRADIENT_UPDATES_PER_PASS_COUNT,
            minibatch_count=constants.DEFAULT_BATCH_COUNT,
            epoch_count=constants.DEFAULT_EPOCH_COUNT,
            is_early_stopping=True,
            contributivity_methods=None,
            is_quick_demo=False,
            save_path=constants.SINGLE_SCENARIOS_FOLDER_NAME,
            scenario_id=1,
            val_set='global',
            test_set='global',
            **kwargs,
    ):
        """

        :param partners_count: int, number of partners. Example: partners_count = 3
        :param amounts_per_partner:  [float]. Fractions of the
        original dataset each partner receives to mock a collaborative ML scenario where each partner provides data
        for the ML training.
        :param active_partners_count: int, the size of the subset of partners that will participate in
        each collaborative learning round, this parameter is only used when 'drfa' is specified as a
        learning approach.
        :param dataset: dataset.Dataset object, or its string identifier. Default is MNIST.
        :param dataset_proportion: float (default: 1)
        :param samples_split_option: Splitter object, or its string identifier (for instance 'random', or 'stratified')
                                     Define the strategy to use to split the data samples between the partners.
                                     Default, RandomSplitter.
        :param corruption_parameters: list of Corruption object, or its string identifier, one for each partner.
                                      Enable to artificially corrupt partner's data.
                                      For instance: [Permutation(proportion=0.2), 'random', 'not-corrupted']
        :param init_model_from: None (default) or path
        :param multi_partner_learning_approach: 'fedavg' (default), 'seq-pure', 'seq-with-final-agg' or 'seqavg'
                                                Define the multi-partner learning approach
        :param aggregation:Aggregator object, or string identifier: 'data_volume' (default), 'uniform' or 'local_score'
        :param gradient_updates_per_pass_count: int
        :param minibatch_count: int
        :param epoch_count: int
        :param is_early_stopping: boolean. Stop the training if scores on val_set reach a plateau
        :param contributivity_methods: A declarative list `[]` of the contributivity measurement methods to be executed.
        :param is_quick_demo: boolean. Useful for debugging
        :param save_path: path where to save the scenario outputs (relative to current working directory)
        :param scenario_id: str
        :param **kwargs:
        """

        # ---------------------------------------------------------------------
        # Initialization of the dataset defined in the config of the experiment
        # ---------------------------------------------------------------------

        # Raise Exception if unknown parameters in the config of the scenario

        params_known = [
            "dataset",
            "dataset_proportion",
            "val_set",
            "test_set"
        ]  # Dataset related
        params_known += [
            "contributivity_methods",
            "multi_partner_learning_approach",
            "aggregation",
        ]  # Federated learning related
        params_known += [
            "partners_count",
            "active_partners_count",
            "amounts_per_partner",
            "corruption_parameters",
            "samples_split_option",
            "samples_split_configuration"
        ]  # Partners related
        params_known += [
            "gradient_updates_per_pass_count",
            "epoch_count",
            "minibatch_count",
            "is_early_stopping",
        ]  # Computation related
        params_known += ["init_model_from"]  # Model related
        params_known += ["is_quick_demo"]
        params_known += ["is_run_as_part_of_an_experiment",
                         "save_path",
                         "scenario_name",
                         "repeat_count",
                         ]

        unrecognised_parameters = [x for x in kwargs.keys() if (x not in params_known and not x.startswith('mpl_'))]
        if len(unrecognised_parameters) > 0:
            for x in unrecognised_parameters:
                logger.debug(f"Unrecognised parameter: {x}")
            raise Exception(
                f"Unrecognised parameters {unrecognised_parameters}, check your configuration"
            )

        # Get and verify which dataset is configured
        if isinstance(dataset, dataset_module.Dataset):
            self.dataset = dataset
        elif isinstance(dataset, str):
            # Reference the object corresponding to the dataset selected and initialize it
            if dataset == constants.MNIST:  # default
                self.dataset = dataset_module.Mnist()
            elif dataset == constants.CIFAR10:
                self.dataset = dataset_module.Cifar10()
            elif dataset == constants.TITANIC:
                self.dataset = dataset_module.Titanic()
            elif dataset == constants.ESC50:
                self.dataset = dataset_module.Esc50()
            elif dataset == constants.IMDB:
                self.dataset = dataset_module.Imdb()
            elif dataset == constants.FMNIST:
                self.dataset = dataset_module.Fmnist()
            else:
                raise Exception(
                    f"Dataset named '{dataset}' is not supported (yet). You can construct your own "
                    f"dataset object, or even add it by contributing to the project !"
                )
            logger.debug(f"Dataset selected: {self.dataset.name}")
        else:
            raise AttributeError(f'The dataset parameter cannot be an {type(dataset)}.'
                                 f' Please provides a Dataset instance or a string identifier')
        # Proportion of the dataset the computation will used
        self.dataset_proportion = dataset_proportion
        assert (
                self.dataset_proportion > 0
        ), "Error in the config file, dataset_proportion should be > 0"
        assert (
                self.dataset_proportion <= 1
        ), "Error in the config file, dataset_proportion should be <= 1"

        if self.dataset_proportion < 1:
            self.dataset.shorten_dataset_proportion(self.dataset_proportion)
        else:
            logger.debug("The full dataset will be used (dataset_proportion is configured to 1)")
            logger.debug(
                f"Computation use the full dataset for scenario #{scenario_id}"
            )

        # --------------------------------------
        #  Definition of collaborative scenarios
        # --------------------------------------

        # Partners mock different partners in a collaborative data science project
        self.partners_list = []  # List of all partners defined in the scenario
        self.partners_count = partners_count  # Number of partners in the scenario

        # For configuring the respective sizes of the partners' datasets
        # (% of samples of the dataset for each partner, ...
        # ... has to sum to 1, and number of items has to equal partners_count)
        self.amounts_per_partner = amounts_per_partner
        if np.sum(self.amounts_per_partner) != 1:
            raise ValueError("The sum of the amount per partners you provided isn't equal to 1")
        if len(self.amounts_per_partner) != self.partners_count:
            raise AttributeError(f"The amounts_per_partner list should have a size ({len(self.amounts_per_partner)}) "
                                 f"equals to partners_count ({self.partners_count})")

        #  To configure how validation set and test set will be organized.
        if test_set in ['local', 'global']:
            self.test_set = test_set
        else:
            raise ValueError(f'Test set can be \'local\' or \'global\' not {test_set}')
        if val_set in ['local', 'global']:
            self.val_set = val_set
        else:
            raise ValueError(f'Validation set can be \'local\' or \'global\' not {val_set}')

        # To configure if data samples are split between partners randomly or in a stratified way...
        # ... so that they cover distinct areas of the samples space
        if isinstance(samples_split_option, Splitter):
            if self.val_set != samples_split_option.val_set:
                logger.warning('The validation set organisation (local/global) is differently configured between the '
                               'provided Splitter and Scenario')
            if self.test_set != samples_split_option.test_set:
                logger.warning('The test set organisation (local/global) is differently configured between the '
                               'provided Splitter and Scenario')
            self.splitter = samples_split_option
        else:
            splitter_param = {'amounts_per_partner': self.amounts_per_partner,
                              'val_set': self.val_set,
                              'test_set': self.test_set,
                              }
            if "samples_split_configuration" in kwargs.keys():
                splitter_param.update({'configuration': kwargs["samples_split_configuration"]})
            self.splitter = IMPLEMENTED_SPLITTERS[samples_split_option](**splitter_param)

        # To configure if the data of the partners are corrupted or not (useful for testing contributivity measures)
        if corruption_parameters:
            self.corruption_parameters = list(
                map(lambda x: x if isinstance(x, Corruption) else IMPLEMENTED_CORRUPTION[x](),
                    corruption_parameters))
        else:
            self.corruption_parameters = [NoCorruption() for _ in range(self.partners_count)]  # default

        # ---------------------------------------------------
        #  Configuration of the distributed learning approach
        # ---------------------------------------------------

        self.mpl = None

        # Multi-partner learning approach
        self.multi_partner_learning_approach = multi_partner_learning_approach
        try:
            self._multi_partner_learning_approach = MULTI_PARTNER_LEARNING_APPROACHES[
                multi_partner_learning_approach]
        except KeyError:
            text_error = f"Multi-partner learning approach '{multi_partner_learning_approach}' is not a valid "
            text_error += "approach. List of supported approach : "
            for key in MULTI_PARTNER_LEARNING_APPROACHES.keys():
                text_error += f"{key}, "
            raise KeyError(text_error)

        # Number of active partners per collaborative learning round
        self.active_partners_count = active_partners_count
        if self._multi_partner_learning_approach == 'drfa':
            assert (
                    1 <= self.active_partners_count < partners_count
            ), "Number of active partners must be strictly smaller than the total number of partners"

        # Define how federated learning aggregation steps are weighted...
        # ... Toggle between 'uniform' (default) and 'data_volume'
        if isinstance(aggregation, Aggregator):
            self.aggregation = aggregation
        else:
            try:
                self.aggregation = AGGREGATORS[aggregation]
            except KeyError:
                raise ValueError(f"aggregation approach '{aggregation}' is not a valid approach. ")

        # Number of epochs, mini-batches and fit_batches in ML training
        self.epoch_count = epoch_count
        assert (
                self.epoch_count > 0
        ), "Error: in the provided config file, epoch_count should be > 0"

        self.minibatch_count = minibatch_count
        assert (
                self.minibatch_count > 0
        ), "Error: in the provided config file, minibatch_count should be > 0"

        self.gradient_updates_per_pass_count = gradient_updates_per_pass_count
        assert self.gradient_updates_per_pass_count > 0, (
            "Error: in the provided config file, "
            "gradient_updates_per_pass_count should be > 0 "
        )

        # Early stopping stops ML training when performance increase is not significant anymore
        # It is used to optimize the number of epochs and the execution time
        self.is_early_stopping = is_early_stopping

        # Model used to initialise model
        self.init_model_from = init_model_from
        if init_model_from == "random_initialization":
            self.use_saved_weights = False
        else:
            self.use_saved_weights = True

        # -----------------------------------------------------------------
        #  Configuration of contributivity measurement contributivity_methods to be tested
        # -----------------------------------------------------------------

        # List of contributivity measures selected and computed in the scenario
        self.contributivity_list = []

        # Contributivity methods
        self.contributivity_methods = []
        if contributivity_methods is not None:
            for method in contributivity_methods:
                if method in constants.CONTRIBUTIVITY_METHODS:
                    self.contributivity_methods.append(method)
                else:
                    raise Exception(f"Contributivity method '{method}' is not in contributivity_methods list.")

        # -------------
        # Miscellaneous
        # -------------

        # Misc.
        self.scenario_id = scenario_id
        self.repeat_count = kwargs.get('repeat_count', 1)

        # The quick demo parameters overwrites previously defined parameters to make the scenario faster to compute
        self.is_quick_demo = is_quick_demo
        if self.is_quick_demo and self.dataset_proportion < 1:
            raise Exception("Don't start a quick_demo without the full dataset")

        if self.is_quick_demo:
            # Use less data and/or less epochs to speed up the computations
            if len(self.dataset.x_train) > constants.TRAIN_SET_MAX_SIZE_QUICK_DEMO:
                index_train = np.random.choice(
                    self.dataset.x_train.shape[0],
                    constants.TRAIN_SET_MAX_SIZE_QUICK_DEMO,
                    replace=False,
                )
                index_val = np.random.choice(
                    self.dataset.x_val.shape[0],
                    constants.VAL_SET_MAX_SIZE_QUICK_DEMO,
                    replace=False,
                )
                index_test = np.random.choice(
                    self.dataset.x_test.shape[0],
                    constants.TEST_SET_MAX_SIZE_QUICK_DEMO,
                    replace=False,
                )
                self.dataset.x_train = self.dataset.x_train[index_train]
                self.dataset.y_train = self.dataset.y_train[index_train]
                self.dataset.x_val = self.dataset.x_val[index_val]
                self.dataset.y_val = self.dataset.y_val[index_val]
                self.dataset.x_test = self.dataset.x_test[index_test]
                self.dataset.y_test = self.dataset.y_test[index_test]
            self.epoch_count = 3
            self.minibatch_count = 2

        # -----------------
        # Output parameters
        # -----------------

        now = datetime.datetime.now()
        now_str = now.strftime("%Y-%m-%d_%Hh%M")
        self.scenario_name = kwargs.get('scenario_name',
                                        f"scenario_{self.scenario_id}_repeat_{self.repeat_count}_{now_str}_"
                                        f"{uuid.uuid4().hex[:3]}")  # to distinguish identical names
        if re.search(r'\s', self.scenario_name):
            raise ValueError(
                f'The scenario name "{self.scenario_name}"cannot be written with space character, please use '
                f'underscore or dash.')
        self.short_scenario_name = f"{self.partners_count}_{self.amounts_per_partner}"

        # We add an attribute for logging if a scenario is run as part of an Experiment (defaulting to False)
        self.is_run_as_part_of_an_experiment = False

        # We define the save folder for graphs and results
        self.save_folder = self.define_scenario_path(save_path)
        print(self.save_folder)

        # -------------------------------------------------------------------
        # Select in the kwargs the parameters to be transferred to sub object
        # -------------------------------------------------------------------

        self.mpl_kwargs = {}
        for key, value in kwargs.items():
            if key.startswith('mpl_'):
                self.mpl_kwargs[key.replace('mpl_', '')] = value

        # -----------------------
        # Provision the scenario
        # -----------------------

        self.instantiate_scenario_partners()
        self.split_data()
        self.compute_batch_sizes()
        self.apply_data_alteration_configuration()

        # ------------------------------------------------
        # Print the description of the scenario configured
        # ------------------------------------------------

        self.log_scenario_description()

    def define_scenario_path(self, save_path=constants.SINGLE_SCENARIOS_FOLDER_NAME):
        """Define the path where to save results of a scenario"""

        scenario_path = Path.cwd() / save_path / self.scenario_name

        # Check if scenario folder already exists
        if scenario_path.exists():
            logger.warning(f"Scenario folder {scenario_path} already exists")
            scenario_path = Path(f"{scenario_path}_{uuid.uuid4().hex[:3]}")  # to distinguish identical names
            logger.warning(f"Scenario folder has been renamed to: {scenario_path}")

        # scenario_path.mkdir(parents=True, exist_ok=False)

        return scenario_path

    @property
    def nb_samples_used(self):
        if len(self.partners_list) == 0:
            return len(self.dataset.x_train)
        else:
            return sum([p.final_nb_samples for p in self.partners_list])

    @property
    def final_relative_nb_samples(self):
        return [p.final_nb_samples / self.nb_samples_used for p in self.partners_list]

    def copy(self, **kwargs):
        params = self.__dict__.copy()
        for key in ['partners_list',
                    '_multi_partner_learning_approach',
                    'mpl',
                    'aggregation',
                    'use_saved_weights',
                    'contributivity_list',
                    'scenario_name',
                    'short_scenario_name',
                    'save_folder',
                    'splitter',
                    ]:
            del params[key]
        if 'is_quick_demo' in kwargs and kwargs['is_quick_demo'] != self.is_quick_demo:
            raise ValueError("Attribute 'is_quick_demo' cannot be modified between copies.")
        params['save_path'] = self.save_folder.parents[0]
        params['samples_split_option'] = self.splitter.copy()
        params['aggregation'] = self.aggregation.name

        params.update(kwargs)

        return Scenario(**params)

    def log_scenario_description(self):
        """Log the description of the scenario configured"""

        # Describe scenario
        logger.info("Description of data scenario configured:")
        logger.info(f"   Number of partners defined: {self.partners_count}")
        logger.info(f"   Data distribution scenario chosen: {self.splitter}")
        logger.info(f"   Multi-partner learning approach: {self.multi_partner_learning_approach}")
        logger.info(f"   Weighting option: {self.aggregation.name}")
        logger.info(f"   Iterations parameters: "
                    f"{self.epoch_count} epochs > "
                    f"{self.minibatch_count} mini-batches > "
                    f"{self.gradient_updates_per_pass_count} gradient updates per pass")

        # Describe data
        logger.info(f"Data loaded: {self.dataset.name}")
        if self.is_quick_demo:
            logger.info("   Quick demo configuration: number of data samples and epochs "
                        "are limited to speed up the run")
        logger.info(
            f"   {len(self.dataset.x_train)} train data with {len(self.dataset.y_train)} labels"
        )
        logger.info(
            f"   {len(self.dataset.x_val)} val data with {len(self.dataset.y_val)} labels"
        )
        logger.info(
            f"   {len(self.dataset.x_test)} test data with {len(self.dataset.y_test)} labels"
        )

    def append_contributivity(self, contributivity_method):
        self.contributivity_list.append(contributivity_method)

    def instantiate_scenario_partners(self):
        """Create the partners_list"""
        if len(self.partners_list) > 0:
            raise Exception('Partners have already been initialized')
        self.partners_list = [Partner(i, corruption=self.corruption_parameters[i]) for i in range(self.partners_count)]

    def split_data(self):
        self.splitter.split(self.partners_list, self.dataset)
        return 0

    def plot_data_distribution(self, save=False, display=True):
        """
        Plot the partners' labels distribution of the train set per class
        :param save: boolean (default is False), True to save the plot, False otherwise
        :param display: boolean (default is True), True to show the plot, False otherwise
        """

        save_folder = None
        if save:
            save_folder = self.save_folder / 'graphs'

        lb = LabelEncoder().fit([str(y) for y in self.dataset.y_train])

        # Get the list and ratio of corrupted partners
        corrupted_partners = {}
        num_occ_partner_per_class = []
        num_occ_partner_per_class_corrupted = []
        for partner in self.partners_list:
            if not isinstance(partner.corruption, NoCorruption):
                corrupted_partners[partner.id] = partner.corruption.proportion
                num_occ_partner_per_class.append(np.bincount(lb.transform([str(y) for y in partner.y_train_true]),
                                                             minlength=self.dataset.num_classes))
            else:
                num_occ_partner_per_class.append(np.bincount(lb.transform([str(y) for y in partner.y_train]),
                                                             minlength=self.dataset.num_classes))
            num_occ_partner_per_class_corrupted.append(np.bincount(lb.transform([str(y) for y in partner.y_train]),
                                                                   minlength=self.dataset.num_classes))

        # Plot the train set distribution of the train set without corruption
        num_occ_per_class_train = np.bincount(lb.transform([str(y) for y in self.dataset.y_train]),
                                              minlength=self.dataset.num_classes)
        num_occ_partner_per_class = np.transpose(num_occ_partner_per_class)
        ratio_partner_per_class = num_occ_partner_per_class / num_occ_per_class_train[:, None]
        utils.create_data_distribution_graph('train_set_ground_truth', ratio_partner_per_class,
                                             corrupted_partners=corrupted_partners,
                                             save_folder=save_folder, display=display)

        # Plot the train set distribution of the train set with corruption
        if len(corrupted_partners) != 0:
            num_occ_partner_per_class_corrupted = np.transpose(num_occ_partner_per_class_corrupted)
            num_occ_per_class_train += np.sum(num_occ_partner_per_class_corrupted - num_occ_partner_per_class, axis=1)
            ratio_partner_per_class = num_occ_partner_per_class_corrupted / num_occ_per_class_train[:, None]
            utils.create_data_distribution_graph('train_set_corrupted_ground_truth', ratio_partner_per_class,
                                                 corrupted_partners=corrupted_partners,
                                                 save_folder=save_folder, display=display)

    def compute_batch_sizes(self):

        # For each partner we compute the batch size in multi-partner and single-partner setups
        batch_size_min = 1
        batch_size_max = constants.MAX_BATCH_SIZE

        if self.partners_count == 1:
            p = self.partners_list[0]
            batch_size = int(len(p.x_train) / self.gradient_updates_per_pass_count)
            p.batch_size = np.clip(batch_size, batch_size_min, batch_size_max)
        else:
            for p in self.partners_list:
                batch_size = int(
                    len(p.x_train)
                    / (self.minibatch_count * self.gradient_updates_per_pass_count)
                )
                p.batch_size = np.clip(batch_size, batch_size_min, batch_size_max)

        for p in self.partners_list:
            logger.debug(f"   Compute batch sizes, partner #{p.id}: {p.batch_size}")

    def apply_data_alteration_configuration(self):
        """perform corruption on partner if needed"""
        for partner in self.partners_list:
            if isinstance(partner.corruption, Duplication):
                if not partner.corruption.duplicated_partner_id:
                    data_volume = np.array([p.data_volume for p in self.partners_list if p.id != partner.id])
                    ids = np.array([p.id for p in self.partners_list if p.id != partner.id])
                    candidates = ids[data_volume >= partner.data_volume * partner.corruption.proportion]
                    partner.corruption.duplicated_partner_id = np.random.choice(candidates)
                partner.corruption.set_duplicated_partner(self.partners_list)
            partner.corrupt()

    def to_dataframe(self):

        df = pd.DataFrame()
        dict_results = {}

        # Scenario definition parameters
        dict_results["scenario_name"] = self.scenario_name
        dict_results["short_scenario_name"] = self.short_scenario_name
        dict_results["dataset_name"] = self.dataset.name
        dict_results["train_data_samples_count"] = len(self.dataset.x_train)
        dict_results["test_data_samples_count"] = len(self.dataset.x_test)
        dict_results["partners_count"] = self.partners_count
        dict_results["dataset_fraction_per_partner"] = self.amounts_per_partner
        dict_results["samples_split_option"] = str(self.splitter)
        dict_results["nb_samples_used"] = self.nb_samples_used
        dict_results["final_relative_nb_samples"] = self.final_relative_nb_samples

        # Multi-partner learning approach parameters
        dict_results["multi_partner_learning_approach"] = self.multi_partner_learning_approach
        dict_results["aggregation_weighting"] = self.aggregation.name
        dict_results["epoch_count"] = self.epoch_count
        dict_results["minibatch_count"] = self.minibatch_count
        dict_results["gradient_updates_per_pass_count"] = self.gradient_updates_per_pass_count
        dict_results["is_early_stopping"] = self.is_early_stopping
        dict_results["mpl_test_score"] = self.mpl.history.score
        dict_results["mpl_nb_epochs_done"] = self.mpl.history.nb_epochs_done
        dict_results["learning_computation_time_sec"] = self.mpl.learning_computation_time

        if not self.contributivity_list:
            df = df.append(dict_results, ignore_index=True)

        for contrib in self.contributivity_list:

            # Contributivity data
            dict_results["contributivity_method"] = contrib.name
            dict_results["contributivity_scores"] = contrib.contributivity_scores
            dict_results["contributivity_stds"] = contrib.scores_std
            dict_results["computation_time_sec"] = contrib.computation_time_sec
            dict_results["first_characteristic_calls_count"] = contrib.first_charac_fct_calls_count

            for i in range(self.partners_count):
                # Partner-specific data
                dict_results["partner_id"] = i
                dict_results["dataset_fraction_of_partner"] = self.amounts_per_partner[i]
                dict_results["contributivity_score"] = contrib.contributivity_scores[i]
                dict_results["contributivity_std"] = contrib.scores_std[i]

                df = df.append(dict_results, ignore_index=True)

        return df

    def run(self):

        # -----------------
        # Preliminary steps
        # -----------------

        if not self.is_run_as_part_of_an_experiment:
            self.save_folder.mkdir(parents=True, exist_ok=False)
            self.plot_data_distribution(save=True, display=False)

        logger.info(f"Now starting running scenario {self.scenario_name}")

        # -----------------------------------------------------
        # Instantiate and run the distributed learning approach
        # -----------------------------------------------------

        self.mpl = self._multi_partner_learning_approach(self, custom_name='main_mpl', **self.mpl_kwargs)
        self.mpl.fit()

        # -------------------------------------------------------------------------
        # Instantiate and run the contributivity measurement contributivity_methods
        # -------------------------------------------------------------------------

        for method in self.contributivity_methods:
            logger.info(f"{method}")
            contrib = contributivity.Contributivity(scenario=self)
            contrib.compute_contributivity(method)
            self.append_contributivity(contrib)
            logger.info(f"Evaluating contributivity with {method}: {contrib}")

        # ------------
        # Save results
        # ------------

        # When a scenario is run as par of an Experiment, saving results is handled by the Experiment
        if not self.is_run_as_part_of_an_experiment:  # When a scenario is ran in a standalone way
            df_results = self.to_dataframe()

            with open(self.save_folder / constants.RESULT_FILE_NAME, "a") as f:
                df_results.to_csv(f, header=f.tell() == 0, index=False)
                logger.info(
                    f"(Scenario {self.scenario_name}) Results saved to {constants.RESULT_FILE_NAME} in folder "
                    f"{os.path.relpath(self.save_folder)}")

        return 0
