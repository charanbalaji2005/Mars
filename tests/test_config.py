import copy

import pytest

from neurotune.config import config_from_dict, load_config
from neurotune.errors import ConfigError


def test_pilot_config_loads():
    cfg = load_config("configs/experiments/pilot.yaml")
    assert cfg.workload.shapes[3] == (257, 511, 129)
    assert cfg.search.strategies == ("random", "learned", "tpe")
    assert cfg.benchmark.report_statistic == "median"
    assert len(cfg.fingerprint()) == 16


def test_transfer_config_loads():
    cfg = load_config("configs/experiments/transfer_generalization.yaml")
    assert not set(cfg.workload.shapes) & set(cfg.collect_shapes), "evaluation shapes must be held out"


def test_all_errors_reported_at_once(config_dict):
    bad = copy.deepcopy(config_dict)
    bad["workload"]["dtype"] = "int8"
    bad["benchmark"]["repetitions"] = 1
    bad["search"]["strategies"] = ["random", "genetic"]
    bad["benchmark"]["typo_key"] = 3
    with pytest.raises(ConfigError) as err:
        config_from_dict(bad)
    msg = str(err.value)
    for fragment in ("workload.dtype", "benchmark.repetitions", "genetic", "typo_key"):
        assert fragment in msg


@pytest.mark.parametrize("shapes", [[[1, 2]], [[0, 4, 4]], [[4, 4, 4], [4, 4, 4]], [["a", 1, 1]], []])
def test_bad_shapes(config_dict, shapes):
    config_dict["workload"]["shapes"] = shapes
    with pytest.raises(ConfigError):
        config_from_dict(config_dict)


def test_initial_trials_must_be_below_budget(config_dict):
    config_dict["search"]["initial_trials"] = 12
    with pytest.raises(ConfigError, match="initial_trials"):
        config_from_dict(config_dict)


def test_bool_is_not_int(config_dict):
    config_dict["benchmark"]["repetitions"] = True
    with pytest.raises(ConfigError):
        config_from_dict(config_dict)


def test_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config("does/not/exist.yaml")
