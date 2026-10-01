import numpy as np

from neurotune.features.extraction import FEATURE_NAMES, feature_dict, feature_matrix
from neurotune.search.space import SearchSpace


def test_features_finite_and_complete(limits):
    configs = SearchSpace().valid_configs(limits, 2)[:200]
    X = feature_matrix([((257, 511, 129), c) for c in configs], limits, 2)
    assert X.shape == (len(configs), len(FEATURE_NAMES))
    assert np.all(np.isfinite(X))


def test_padding_efficiency(limits):
    from neurotune.search.space import KernelConfig

    cfg = KernelConfig(64, 64, 32, 8, 4, 3)
    assert feature_dict((128, 128, 128), cfg, limits, 2)["pad_efficiency"] == 1.0
    assert feature_dict((129, 128, 128), cfg, limits, 2)["pad_efficiency"] < 0.7
