"""`neurotune train`: build a dataset from stored trials, evaluate, fit, and save."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..errors import NeuroTuneError
from ..search.space import DTYPE_BYTES, DeviceLimits
from ..storage.experiment_store import ExperimentStore
from .latency_predictor import LatencyPredictor, Measurement, aggregate_measurements, design_matrix, evaluate


def load_dataset(store: ExperimentStore, *, device_name: str | None = None, dtype: str | None = None,
                 include_optimize: bool = False) -> tuple[list[Measurement], DeviceLimits, str, dict]:
    kinds = ["collect", "benchmark"] + (["optimize"] if include_optimize else [])
    rows = [r for r in store.trials(kinds=kinds, status="ok", exclude_strategies=["reference"]) if r.config]
    if not rows:
        raise NeuroTuneError("no successful Triton trials found; run `neurotune collect` first")
    devices = Counter(r.device_name for r in rows)
    if device_name is None:
        if len(devices) > 1:
            raise NeuroTuneError(f"dataset mixes devices {dict(devices)}; choose one with --device")
        device_name = next(iter(devices))
    dtypes = Counter(r.dtype for r in rows if r.device_name == device_name)
    if dtype is None:
        if len(dtypes) > 1:
            raise NeuroTuneError(f"dataset mixes dtypes {dict(dtypes)}; choose one with --dtype")
        dtype = next(iter(dtypes))
    rows = [r for r in rows if r.device_name == device_name and r.dtype == dtype]
    if not rows:
        raise NeuroTuneError(f"no trials for device {device_name!r} and dtype {dtype!r}")
    limits = None
    for exp_id in sorted({r.experiment_id for r in rows}):
        stored = store.get_experiment(exp_id)["hardware"].get("limits")
        if stored and stored.get("name") == device_name:
            limits = DeviceLimits.from_dict(stored)
            break
    if limits is None:
        raise NeuroTuneError(f"no device limits recorded for {device_name!r}")
    measurements = aggregate_measurements((r.shape, r.kernel_config, r.median_ms) for r in rows)
    info = {"device": device_name, "dtype": dtype, "raw_trials": len(rows), "unique_pairs": len(measurements),
            "experiments": sorted({r.experiment_id for r in rows}), "kinds": kinds}
    return measurements, limits, dtype, info


def train(dataset: str | Path, output_dir: str | Path, *, device_name: str | None = None,
          dtype: str | None = None, include_optimize: bool = False, seed: int = 0) -> dict:
    with ExperimentStore(dataset, create=False) as store:
        measurements, limits, dtype, info = load_dataset(store, device_name=device_name, dtype=dtype,
                                                         include_optimize=include_optimize)
    dtype_bytes = DTYPE_BYTES[dtype]
    evaluation = evaluate(measurements, limits, dtype_bytes, seed=seed)
    X, y, _ = design_matrix(measurements, limits, dtype_bytes)
    model = LatencyPredictor(seed=seed).fit(X, y)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": str(dataset), **info, "device_limits": limits.to_dict(), "seed": seed,
               "evaluation": evaluation, "feature_importances": model.feature_importances()}
    model.save(output_dir / "latency_predictor.joblib", metadata=summary)
    (output_dir / "predictor_evaluation.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
