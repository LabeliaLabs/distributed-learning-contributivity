from . import basic_mpl
from . import fast_mpl

# Supported multi-partner learning approaches
BASIC_MPL_APPROACHES = {
    "fedavg": basic_mpl.FederatedAverageLearning,
    'fedgrads': basic_mpl.FederatedGradients,
    "seq-pure": basic_mpl.SequentialLearning,
    "seq-with-final-agg": basic_mpl.SequentialWithFinalAggLearning,
    "seqavg": basic_mpl.SequentialAverageLearning,
    "fedavg-smodel": basic_mpl.FedAvgSmodel,
    "ensemble": basic_mpl.EnsemblePredictions,
    "single-partner": basic_mpl.SinglePartnerLearning,
    "drfa": basic_mpl.DistributionallyRobustFederatedAveragingLearning}

FAST_TF_MPL_APPROACHES = {
    'fast-fedavg': fast_mpl.FastFedAvg,
    'fast-fedgrads': fast_mpl.FastFedGrad,
    'fast-fedavg-smodel': fast_mpl.FastFedAvgSmodel,
    'fast-fedgrad-smodel': fast_mpl.FastGradSmodel,
    'fast-fedgdo': fast_mpl.FastFedGDO
}

MULTI_PARTNER_LEARNING_APPROACHES = BASIC_MPL_APPROACHES.copy()
MULTI_PARTNER_LEARNING_APPROACHES.update(FAST_TF_MPL_APPROACHES)
