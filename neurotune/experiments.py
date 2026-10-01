"""Experiment orchestration for the CLI commands.

Each function validates everything it can before GPU work, records hardware and
software metadata, and wraps the run in a session that marks the experiment
`complete`, `interrupted` (Ctrl-C) or `failed`. A run killed without cleanup is
left as `running` and is reported as incomplete; `--resume ID` continues it.
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from .benchmark.runner import TrialResult, TrialRunner, to_record
from .config import STRATEGIES, ExperimentConfig
from .errors import ConfigError, StoreError
from .logging_utils import add_file_handler, get_logger, log_event
from .models.latency_predictor import Measurement, aggregate_measurements
from .search.base import make_strategy
from .search.optimizer import SearchOutcome, run_search
from .search.space import (DEFAULT_CONFIG, DTYPE_BYTES, DeviceLimits, KernelConfig, SearchSpace, Shape,
                           derive_seed, load_space, seeded_order)
from .storage.experiment_store import ExperimentStore
from .workloads.matmul import MatmulWorkload

log = get_logger("experiments")


@dataclass
class GpuContext:
    limits: DeviceLimits
    hardware: dict
    runner: TrialRunner


def gpu_context(config: ExperimentConfig) -> GpuContext:
    """Discover the device and build the Triton runner (fails clearly without a GPU)."""
    from .benchmark.runner import TritonMatmulRunner
    from .hardware.discovery import hardware_metadata, require_gpu

    limits = require_gpu(config.workload.dtype)
    check_shapes_fit(config, limits)
    hardware = hardware_metadata(limits, full=config.hardware.record_device_metadata)
    return GpuContext(limits, hardware, TritonMatmulRunner(config, limits))


def check_shapes_fit(config: ExperimentConfig, limits: DeviceLimits) -> None:
    errors = []
    for shape in sorted(set(config.workload.shapes) | set(config.collect_shapes)):
        need = MatmulWorkload(shape, config.workload.dtype).memory_estimate_bytes()
        if need > 0.6 * limits.total_memory_bytes:
            errors.append(f"shape {list(shape)} needs ~{need / 2**30:.2f} GiB, too large for "
                          f"{limits.total_memory_bytes / 2**30:.1f} GiB on {limits.name}")
    if errors:
        raise ConfigError("workload does not fit on the device:\n  - " + "\n  - ".join(errors))


def _software() -> dict:
    from .hardware.discovery import software_versions

    return software_versions()


@contextmanager
def session(store: ExperimentStore, kind: str, config: ExperimentConfig, hardware: dict,
            resume_id: str | None = None, notes: dict | None = None) -> Iterator[str]:
    if resume_id:
        exp = store.get_experiment(resume_id)
        if exp["kind"] != kind:
            raise StoreError(f"experiment {resume_id} is a '{exp['kind']}' experiment, not '{kind}'")
        if exp["config_fingerprint"] != config.fingerprint():
            raise StoreError(f"configuration changed since {resume_id} was started "
                             f"(fingerprint {exp['config_fingerprint']} != {config.fingerprint()}); "
                             "resuming would mix protocols. Start a new experiment instead.")
        if exp["status"] == "complete":
            log_event(log, "experiment already complete; nothing to resume", experiment=resume_id)
        stored_device = exp["hardware"].get("limits", {}).get("name")
        current_device = hardware.get("limits", {}).get("name")
        if stored_device and current_device and stored_device != current_device:
            raise StoreError(f"{resume_id} was run on {stored_device}, current device is {current_device}")
        exp_id = resume_id
        store.set_status(exp_id, "running")
        resumes = exp["notes"].get("resumes", 0) + 1
        store.update_notes(exp_id, {"resumes": resumes, **(notes or {})})
    else:
        notes_init = dict(notes or {})
        try:
            from .hardware.discovery import gpu_telemetry_snapshot
            telemetry = gpu_telemetry_snapshot()
            if telemetry:
                notes_init["gpu_telemetry_at_start"] = telemetry
        except Exception:
            pass
        exp_id = store.create_experiment(kind, config.experiment.name, config.to_dict(), hardware, _software(),
                                         fingerprint=config.fingerprint(), notes=notes_init)
    add_file_handler(Path(config.experiment.output_dir) / "logs" / f"{exp_id}.jsonl")
    log_event(log, "experiment started", experiment=exp_id, kind=kind, resumed=bool(resume_id))
    try:
        yield exp_id
    except KeyboardInterrupt:
        store.set_status(exp_id, "interrupted", note="interrupted by user")
        log_event(log, "experiment interrupted; resume with --resume", logging.WARNING, experiment=exp_id)
        raise
    except BaseException as exc:
        store.set_status(exp_id, "failed", note=f"{type(exc).__name__}: {exc}")
        log_event(log, "experiment failed", logging.ERROR, experiment=exp_id, error=str(exc))
        raise
    else:
        try:
            from .hardware.discovery import gpu_telemetry_snapshot, nvidia_smi_snapshot

            store.update_notes(exp_id, {
                "nvidia_smi_at_end": nvidia_smi_snapshot(),
                "gpu_telemetry_at_end": gpu_telemetry_snapshot(),
            })
        except Exception:
            pass
        store.set_status(exp_id, "complete")
        log_event(log, "experiment complete", experiment=exp_id)


def _space(config: ExperimentConfig) -> SearchSpace:
    return load_space(config.search.space_file)


def _record(store, exp_id, runner, config, strategy, seed, index, shape, cfg: KernelConfig | None,
            result: TrialResult, predicted=None) -> None:
    store.record_trial(to_record(result, experiment_id=exp_id, strategy=strategy, seed=seed, trial_index=index,
                                 shape=shape, dtype=config.workload.dtype, config=cfg,
                                 device_name=runner.device_name, predicted_ms=predicted,
                                 include_samples=config.reporting.include_raw_samples))


def ensure_baselines(store: ExperimentStore, exp_id: str, runner: TrialRunner, config: ExperimentConfig,
                     shapes: Sequence[Shape]) -> None:
    """Default Triton configuration and torch.matmul for every shape (once per experiment)."""
    for shape in shapes:
        if not store.trials(experiment_ids=[exp_id], strategy="default", shape=shape):
            result = runner.run(shape, DEFAULT_CONFIG)
            _record(store, exp_id, runner, config, "default", 0, 0, shape, DEFAULT_CONFIG, result)
            log_event(log, "default config", shape="x".join(map(str, shape)), status=result.status,
                      latency_ms=result.latency(config.benchmark.report_statistic))
        if not store.trials(experiment_ids=[exp_id], strategy="reference", shape=shape):
            result = runner.run_reference(shape)
            _record(store, exp_id, runner, config, "reference", 0, 0, shape, None, result)
            log_event(log, "torch.matmul reference", shape="x".join(map(str, shape)), status=result.status,
                      latency_ms=result.latency(config.benchmark.report_statistic))


# --------------------------------------------------------------------------- commands
def run_benchmark(config: ExperimentConfig, *, store: ExperimentStore, ctx: GpuContext) -> str:
    with session(store, "benchmark", config, ctx.hardware) as exp_id:
        ensure_baselines(store, exp_id, ctx.runner, config, config.workload.shapes)
    return exp_id


def run_collect(config: ExperimentConfig, *, store: ExperimentStore, ctx: GpuContext,
                resume_id: str | None = None, samples_per_shape: int | None = None) -> str:
    space, dtype_bytes = _space(config), DTYPE_BYTES[config.workload.dtype]
    n_samples = samples_per_shape or config.collect.samples_per_shape
    shapes = config.collect_shapes
    notes = {"search_space": space.to_dict(), "samples_per_shape": n_samples,
             "prune_report": {"x".join(map(str, s)): space.prune_report(ctx.limits, dtype_bytes, s) for s in shapes}}
    with session(store, "collect", config, ctx.hardware, resume_id, notes) as exp_id:
        ensure_baselines(store, exp_id, ctx.runner, config, shapes)
        for shape in shapes:
            candidates = space.valid_configs(ctx.limits, dtype_bytes, shape)
            order = seeded_order(candidates, derive_seed(config.experiment.seed, "collect", *shape))[:n_samples]
            done = {r.config_key for r in store.trials(experiment_ids=[exp_id], strategy="collect", shape=shape)}
            log_event(log, "collecting", shape="x".join(map(str, shape)), valid=len(candidates),
                      sampling=len(order), already_done=len(done))
            for index, cfg in enumerate(order):
                if cfg.key() in done:
                    continue
                result = ctx.runner.run(shape, cfg)
                _record(store, exp_id, ctx.runner, config, "collect", config.experiment.seed, index, shape, cfg, result)
                if (index + 1) % 25 == 0:
                    log_event(log, "progress", shape="x".join(map(str, shape)), done=index + 1, total=len(order))
    return exp_id


def load_prior(path: str | Path, device_name: str, dtype: str) -> list[Measurement]:
    with ExperimentStore(path, create=False) as prior_store:
        rows = prior_store.trials(kinds=["collect"], status="ok", device_name=device_name, dtype=dtype,
                                  exclude_strategies=["reference"])
    return aggregate_measurements((r.shape, r.kernel_config, r.median_ms) for r in rows if r.config)


def run_optimize(config: ExperimentConfig, *, store: ExperimentStore, ctx: GpuContext, strategies: Sequence[str],
                 budget: int, seeds: int | None = None, initial_trials: int | None = None,
                 shapes: Sequence[Shape] | None = None, prior_dataset: str | None = None,
                 resume_id: str | None = None, kappa: float | None = None) -> tuple[str, list[SearchOutcome]]:
    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown:
        raise ConfigError(f"unknown strategies {unknown}; choose from {list(STRATEGIES)}")
    if budget < 1:
        raise ConfigError("--budget must be >= 1")
    n_init = initial_trials if initial_trials is not None else config.search.initial_trials
    n_seeds = seeds or config.search.seeds
    kappa_val = kappa if kappa is not None else config.search.kappa
    target_shapes = tuple(shapes) if shapes else config.workload.shapes
    missing = [s for s in target_shapes if s not in config.workload.shapes]
    if missing:
        raise ConfigError(f"shapes {missing} are not listed in workload.shapes")
    if "learned" in strategies and n_init >= budget:
        raise ConfigError(f"initial trials ({n_init}) must be smaller than the budget ({budget}) "
                          "or learned search is just random search")
    space, dtype_bytes = _space(config), DTYPE_BYTES[config.workload.dtype]
    prior = load_prior(prior_dataset, ctx.runner.device_name, config.workload.dtype) if prior_dataset else None
    protocol = {"strategies": list(strategies), "budget": budget, "seeds": n_seeds, "initial_trials": n_init,
                "kappa": kappa_val, "shapes": [list(s) for s in target_shapes],
                "prior_dataset": prior_dataset, "prior_measurements": len(prior or []),
                "primary_metric": config.evaluation.primary_metric, "within_pct": config.evaluation.within_pct,
                "search_space": space.to_dict()}
    outcomes: list[SearchOutcome] = []
    with session(store, "optimize", config, ctx.hardware, resume_id) as exp_id:
        stored = store.get_experiment(exp_id)["notes"].get("protocol")
        if stored and stored != json.loads(json.dumps(protocol)):
            raise StoreError(f"{exp_id} was started with a different protocol (strategies/budget/seeds/shapes); "
                             "resume with the original arguments")
        store.update_notes(exp_id, {"protocol": protocol})
        ensure_baselines(store, exp_id, ctx.runner, config, target_shapes)
        for shape in target_shapes:
            candidates = space.valid_configs(ctx.limits, dtype_bytes, shape)
            for strategy_name in strategies:
                for s in range(n_seeds):
                    seed = config.experiment.seed + s
                    strategy = make_strategy(strategy_name, candidates, shape=shape,
                                             seed=derive_seed(seed, *shape), initial_trials=n_init,
                                             device=ctx.limits, dtype_bytes=dtype_bytes,
                                             kappa=kappa_val, prior=prior)
                    outcome = run_search(store=store, experiment_id=exp_id, runner=ctx.runner, strategy=strategy,
                                         shape=shape, dtype=config.workload.dtype, seed=seed, budget=budget,
                                         statistic=config.benchmark.report_statistic,
                                         include_samples=config.reporting.include_raw_samples)
                    outcomes.append(outcome)
                    log_event(log, "search finished", shape="x".join(map(str, shape)), strategy=strategy_name,
                              seed=seed, trials=outcome.trials_used, best_ms=outcome.best_ms,
                              best=outcome.best_key, failures=outcome.failures,
                              duplicates_skipped=outcome.duplicates_skipped)
    return exp_id, outcomes


def run_validate(config: ExperimentConfig, *, ctx: GpuContext, shapes: Sequence[Shape],
                 sampled_configs: int = 4) -> list[dict]:
    """Correctness of the default config plus a few sampled valid configs on each shape."""
    space, dtype_bytes = _space(config), DTYPE_BYTES[config.workload.dtype]
    results = []
    for shape in shapes:
        candidates = space.valid_configs(ctx.limits, dtype_bytes, shape)
        configs = [DEFAULT_CONFIG] + [c for c in seeded_order(candidates, derive_seed("validate", *shape))
                                      if c != DEFAULT_CONFIG][:sampled_configs]
        for cfg in configs:
            r = ctx.runner.validate(shape, cfg)
            results.append({"shape": list(shape), "config": cfg.key(), "status": r.status, "error": r.error,
                            "max_abs_err": r.max_abs_err, "max_rel_err": r.max_rel_err,
                            "first_call_s": r.first_call_s})
    return results
