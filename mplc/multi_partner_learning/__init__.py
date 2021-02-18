from . import basic_mpl
from . import fast_mpl

# Supported multi-partner learning approaches

MULTI_PARTNER_LEARNING_APPROACHES = {
    "fedavg": basic_mpl.FederatedAverageLearning,
    'fedgrads': basic_mpl.FederatedGradients,
    "seq-pure": basic_mpl.SequentialLearning,
    "seq-with-final-agg": basic_mpl.SequentialWithFinalAggLearning,
    "seqavg": basic_mpl.SequentialAverageLearning,
    "fedavg-smodel": fast_mpl.FedAvgSmodel,
    'fast-fedavg': fast_mpl.FastFedAvg,
    'fast-fedgrads': fast_mpl.FastFedGrad,
    'fast-fedavg-smodel': fast_mpl.FastFedSmodel,
    'fast-fedgrad-smodel': fast_mpl.FastGradSmodel

}
