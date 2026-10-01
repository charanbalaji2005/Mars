"""Budgeted search loop shared by all strategies.

Budget accounting (identical for every strategy):
* one unit = one configuration actually sent to the GPU (compile + validate + time),
  including trials that fail to compile, run out of resources, or are incorrect;
* statically invalid configurations never reach the GPU and cost nothing;
* re-proposals of an already measured configuration are skipped and cost nothing.

Resume: stored trials for (experiment, strategy, seed, shape) are replayed into a
fresh strategy in their original order, then the loop continues until the budget is used.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..benchmark.runner import TrialRunner, to_record
from ..logging_utils import get_logger, log_event
from ..storage.experiment_store import ExperimentStore
from .base import SearchStrategy
from .space import KernelConfig, Shape

MAX_DUPLICATE_PROPOSALS_FACTOR = 20


@dataclass
class SearchOutcome:
    shape: Shape
    strategy: str
    seed: int
    budget: int
    trials_used: int = 0
    resumed_trials: int = 0
    duplicates_skipped: int = 0
    failures: int = 0
    best_key: str | None = None
    best_ms: float | None = None
    exhausted: bool = False
    trajectory: list[float | None] = field(default_factory=list)


def run_search(*, store: ExperimentStore, experiment_id: str, runner: TrialRunner, strategy: SearchStrategy,
               shape: Shape, dtype: str, seed: int, budget: int, statistic: str = "median",
               include_samples: bool = True, logger: logging.Logger | None = None) -> SearchOutcome:
    logger = logger or get_logger("search")
    outcome = SearchOutcome(tuple(shape), strategy.name, seed, budget)
    measured: dict[str, float | None] = {}

    def note(key: str, latency: float | None) -> None:
        if latency is None:
            outcome.failures += 1
        elif outcome.best_ms is None or latency < outcome.best_ms:
            outcome.best_ms, outcome.best_key = latency, key
        outcome.trajectory.append(outcome.best_ms)

    history = store.trials(experiment_ids=[experiment_id], strategy=strategy.name, seed=seed, shape=shape)
    for rec in sorted(history, key=lambda r: r.trial_index):
        cfg = rec.kernel_config
        latency = rec.latency(statistic) if rec.ok else None
        strategy.observe(cfg, latency)
        measured[rec.config_key] = latency
        note(rec.config_key, latency)
    outcome.resumed_trials = outcome.trials_used = len(history)
    if history:
        log_event(logger, "resuming search", shape="x".join(map(str, shape)), strategy=strategy.name,
                  seed=seed, completed=len(history), budget=budget)

    max_dupes = MAX_DUPLICATE_PROPOSALS_FACTOR * max(budget, 1)
    while outcome.trials_used < budget:
        cfg: KernelConfig | None = strategy.propose()
        if cfg is None:
            outcome.exhausted = True
            break
        key = cfg.key()
        if key in measured:
            outcome.duplicates_skipped += 1
            if outcome.duplicates_skipped > max_dupes:
                log_event(logger, "strategy keeps proposing measured configs; stopping", logging.WARNING,
                          strategy=strategy.name)
                break
            continue
        predicted = strategy.last_prediction_ms
        result = runner.run(shape, cfg)
        latency = result.latency(statistic)
        store.record_trial(to_record(result, experiment_id=experiment_id, strategy=strategy.name, seed=seed,
                                     trial_index=outcome.trials_used, shape=shape, dtype=dtype, config=cfg,
                                     device_name=runner.device_name, predicted_ms=predicted,
                                     include_samples=include_samples))
        strategy.observe(cfg, latency)
        measured[key] = latency
        note(key, latency)
        outcome.trials_used += 1
        log_event(logger, "trial", level=logging.DEBUG, strategy=strategy.name, seed=seed,
                  shape="x".join(map(str, shape)), index=outcome.trials_used, config=key,
                  status=result.status, latency_ms=latency, best_ms=outcome.best_ms)
    return outcome
