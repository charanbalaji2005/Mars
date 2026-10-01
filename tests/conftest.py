"""Shared fixtures.

`SyntheticRunner` is a TEST DOUBLE: it returns latencies from a made-up analytical
function so the search, storage, resume and reporting logic can be exercised
without a GPU. Its numbers say nothing about real kernel performance.
"""
from __future__ import annotations

import copy
import math
import random

import pytest

from neurotune.benchmark.runner import TrialResult
from neurotune.benchmark.timing import summarize
from neurotune.config import config_from_dict
from neurotune.features.extraction import feature_dict
from neurotune.search.space import DeviceLimits, derive_seed

BASE_CONFIG = {
    "experiment": {"name": "test", "seed": 1, "output_dir": "OUT"},
    "workload": {"kernel": "matmul", "dtype": "fp16", "accumulation": "fp32",
                 "shapes": [[128, 128, 128], [257, 511, 129]]},
    "search": {"initial_trials": 4, "max_trials": 12, "seeds": 2},
    "benchmark": {"warmup_iterations": 1, "repetitions": 5},
    "collect": {"samples_per_shape": 30},
}


def synthetic_latency_ms(shape, cfg, limits) -> float:
    f = feature_dict(shape, cfg, limits, 2)
    base = 2 * shape[0] * shape[1] * shape[2] / 2e10
    penalty = (1 / f["pad_efficiency"]) * (1 / math.sqrt(f["last_wave_util"]))
    penalty *= 1 + 0.15 * abs(math.log2(cfg.block_k) - 5) + 0.1 * abs(cfg.num_warps - 4) / 4
    penalty *= 1 + 0.05 * abs(cfg.num_stages - 3)
    return base * penalty + 0.004


class SyntheticRunner:
    def __init__(self, limits: DeviceLimits | None = None, fail_keys=()):
        self.limits = limits or DeviceLimits.assumed_rtx4050_laptop()
        self.device_name = self.limits.name
        self.fail_keys = set(fail_keys)
        self.calls = 0

    def _samples(self, mean, key):
        rng = random.Random(derive_seed(key))
        return [mean * (1 + rng.gauss(0, 0.01)) for _ in range(5)]

    def run(self, shape, config):
        self.calls += 1
        if config.num_stages == 5 and config.block_k == 128 or config.key() in self.fail_keys:
            return TrialResult("out_of_resources", error="OutOfResources: synthetic", wall_s=0.01)
        mean = synthetic_latency_ms(shape, config, self.limits)
        return TrialResult("ok", first_call_s=0.2, wall_s=0.05,
                           timing=summarize(self._samples(mean, (shape, config.key()))),
                           max_abs_err=0.001, max_rel_err=0.0005)

    def run_reference(self, shape):
        mean = 2 * shape[0] * shape[1] * shape[2] / 3e10 + 0.003
        return TrialResult("ok", wall_s=0.01, timing=summarize(self._samples(mean, shape)))

    def validate(self, shape, config):
        return TrialResult("ok", max_abs_err=0.001, max_rel_err=0.0005, first_call_s=0.1)

    def close(self):
        pass


@pytest.fixture
def config_dict(tmp_path):
    data = copy.deepcopy(BASE_CONFIG)
    data["experiment"]["output_dir"] = str(tmp_path / "artifacts")
    return data


@pytest.fixture
def config(config_dict):
    return config_from_dict(config_dict)


@pytest.fixture
def limits():
    return DeviceLimits.assumed_rtx4050_laptop()


@pytest.fixture
def synthetic_ctx(limits):
    from neurotune.experiments import GpuContext

    return GpuContext(limits=limits, hardware={"limits": limits.to_dict()}, runner=SyntheticRunner(limits))
