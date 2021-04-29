from mplc.scenario import Scenario
from mplc.experiment import Experiment
from mplc.multi_partner_learning import FederatedAverageLearning
from mplc.corruption import Randomize
import pathlib

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

NB_EPOCH = 40
NB_MINIBATCH = 20
NB_GRAD_UPDATE = 8

sc1 = Scenario(
                scenario_id = 1,
                partners_count=2,
                dataset_name='mnist',
                amounts_per_partner = [0.7, 0.3],
                samples_split_option='stratified',
                multi_partner_learning_approach='fedavg', #default
                aggregation_weighting='data-volume', #default
                contributivity_methods=["Independent scores","Shapley values"],
                epoch_count=NB_EPOCH,
                minibatch_count = NB_MINIBATCH,
                gradient_updates_per_pass_count=NB_GRAD_UPDATE,
            )

# sc2 = Scenario(
#                 scenario_id = 2,
#                 partners_count=2,
#                 dataset_name='cifar10',
#                 amounts_per_partner = [0.7, 0.3],
#                 samples_split_option='random', #default
#                 corruption_parameters=['not-corrupted',Randomize(proportion=0.5)],
#                 multi_partner_learning_approach='fedavg', #default
#                 aggregation_weighting='data-volume', #default
#                 contributivity_methods=["Independent scores","Shapley values"],
#                 epoch_count=NB_EPOCH,
#                 minibatch_count = NB_MINIBATCH,
#                 gradient_updates_per_pass_count=NB_GRAD_UPDATE,
#             )

# sc3 = Scenario(
#                 scenario_id=3,
#                 partners_count=3,
#                 dataset_name='cifar10',
#                 amounts_per_partner=[0.42, 0.42, 0.16],
#                 samples_split_option='flexible',
#                 samples_split_configuration=[[0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], 
#                                                 [0.4, 0.4, 0.4, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
#                                                 [0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#                                                 ],
#                 multi_partner_learning_approach='fedavg', #default
#                 aggregation_weighting='data-volume', #default
#                 contributivity_methods=["Independent scores","Shapley values"],
#                 epoch_count=NB_EPOCH,
#                 minibatch_count=NB_MINIBATCH,
#                 gradient_updates_per_pass_count=NB_GRAD_UPDATE,
#             )

# sc4 = Scenario(
#                 scenario_id=4,
#                 partners_count=5,
#                 dataset_name='cifar10',
#                 amounts_per_partner=[0.25, 0.2, 0.2, 0.2, 0.15],
#                 samples_split_option='random',  # default
#                 corruption_parameters=['not-corrupted', 'random', 'not-corrupted', 'not-corrupted', 'not-corrupted'],
#                 multi_partner_learning_approach='fedavg',  # default
#                 aggregation_weighting='data-volume',  # default
#                 contributivity_methods=["Independent scores"],  # "Shapley values" too long?
#                 epoch_count=NB_EPOCH,
#                 minibatch_count=NB_MINIBATCH,
#                 gradient_updates_per_pass_count=NB_GRAD_UPDATE,
#             )

# sc5 = Scenario(
#                 scenario_id=5,
#                 partners_count=11,
#                 dataset_name='mnist',
#                 amounts_per_partner=[0.8/9.0]*9 + [0.1]*2,
#                 samples_split_option='advanced',
#                 samples_split_configuration=[[8, 'shared']]*9 + [[1, 'specific']]*2,
#                 corruption_parameters=['not-corrupted']*5 + ['random'] + ['not-corrupted']*5,
#                 multi_partner_learning_approach='fedavg',  # default
#                 aggregation_weighting='data-volume',  # default
#                 contributivity_methods=["Independent scores"], # "Shapley values" too long?
#                 epoch_count=NB_EPOCH,
#                 minibatch_count=NB_MINIBATCH,
#                 gradient_updates_per_pass_count=NB_GRAD_UPDATE,
#             )

exp = Experiment(
                experiment_name='benchmarks', 
                #nb_repeats=10, 
                scenarios_list=[sc1], #,sc2,sc3,sc4,sc5
                is_save=True,
                experiment_path=pathlib.Path('saved_benchmarks')
    )

exp.run()