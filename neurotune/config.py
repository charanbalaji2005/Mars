"""Experiment configuration loading and validation.

Every command validates its configuration completely, and reports *all* problems
at once, before touching the GPU.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

Shape = tuple[int, int, int]

KERNELS = ("matmul",)
DTYPES = ("fp16", "bf16")
ACCUMULATIONS = ("fp32",)
STRATEGIES = ("random", "learned", "tpe")
STATISTICS = ("median", "mean", "min")
PRIMARY_METRICS = ("best_latency_at_budget", "trials_to_within_pct")
MAX_DIM = 16384
_MISSING = object()


@dataclass(frozen=True)
class ExperimentSection:
    name: str
    seed: int = 42
    output_dir: str = "artifacts"


@dataclass(frozen=True)
class HardwareSection:
    require_cuda: bool = True
    record_device_metadata: bool = True


@dataclass(frozen=True)
class WorkloadSection:
    kernel: str
    dtype: str
    accumulation: str
    shapes: tuple[Shape, ...]


@dataclass(frozen=True)
class SearchSection:
    initial_trials: int = 10
    max_trials: int = 50
    strategies: tuple[str, ...] = STRATEGIES
    seeds: int = 3
    kappa: float = 1.0
    space_file: str = ""


@dataclass(frozen=True)
class BenchmarkSection:
    warmup_iterations: int = 10
    repetitions: int = 30
    report_statistic: str = "median"
    validate_before_timing: bool = True
    flush_l2: bool = True


@dataclass(frozen=True)
class CorrectnessSection:
    rtol: float = 0.01
    atol: float = 0.01
    reject_nan_inf: bool = True


@dataclass(frozen=True)
class CollectSection:
    samples_per_shape: int = 150
    shapes: tuple[Shape, ...] = ()


@dataclass(frozen=True)
class EvaluationSection:
    primary_metric: str = "best_latency_at_budget"
    within_pct: float = 5.0
    bootstrap_resamples: int = 2000


@dataclass(frozen=True)
class ReportingSection:
    include_raw_samples: bool = True
    include_compile_time: bool = True
    include_speedup: bool = True
    include_prediction_error: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: ExperimentSection
    hardware: HardwareSection
    workload: WorkloadSection
    search: SearchSection
    benchmark: BenchmarkSection
    correctness: CorrectnessSection
    collect: CollectSection
    evaluation: EvaluationSection
    reporting: ReportingSection

    @property
    def collect_shapes(self) -> tuple[Shape, ...]:
        return self.collect.shapes or self.workload.shapes

    @property
    def db_path(self) -> Path:
        return Path(self.experiment.output_dir) / "matmul_trials.db"

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def _is_type(value: Any, typ: type) -> bool:
    if typ is bool:
        return isinstance(value, bool)
    if typ is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if typ is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, typ)


class _Reader:
    """Reads one YAML section, collecting errors instead of raising on the first one."""

    def __init__(self, data: Any, section: str, errors: list[str]):
        self.section, self.errors, self.used = section, errors, set()
        if data is None:
            data = {}
        if not isinstance(data, dict):
            errors.append(f"'{section}' must be a mapping, got {type(data).__name__}")
            data = {}
        self.data = data

    def get(self, key, typ, default=_MISSING, *, choices=None, minimum=None, maximum=None):
        self.used.add(key)
        where = f"{self.section}.{key}"
        fallback = None if default is _MISSING else default
        if key not in self.data:
            if default is _MISSING:
                self.errors.append(f"{where} is required")
            return fallback
        value = self.data[key]
        if not _is_type(value, typ):
            self.errors.append(f"{where} must be of type {typ.__name__}, got {value!r}")
            return fallback
        if typ is float:
            value = float(value)
        if choices is not None and value not in choices:
            self.errors.append(f"{where} must be one of {list(choices)}, got {value!r}")
        if minimum is not None and value < minimum:
            self.errors.append(f"{where} must be >= {minimum}, got {value!r}")
        if maximum is not None and value > maximum:
            self.errors.append(f"{where} must be <= {maximum}, got {value!r}")
        return value

    def get_choices_list(self, key, default, choices) -> tuple[str, ...]:
        self.used.add(key)
        where = f"{self.section}.{key}"
        if key not in self.data:
            return tuple(default)
        value = self.data[key]
        if not isinstance(value, list) or not value:
            self.errors.append(f"{where} must be a non-empty list")
            return tuple(default)
        bad = [v for v in value if v not in choices]
        if bad:
            self.errors.append(f"{where} contains unsupported values {bad}; supported: {list(choices)}")
        if len(set(value)) != len(value):
            self.errors.append(f"{where} contains duplicates")
        return tuple(v for v in value if v in choices)

    def get_shapes(self, key, *, required: bool) -> tuple[Shape, ...]:
        self.used.add(key)
        where = f"{self.section}.{key}"
        if key not in self.data:
            if required:
                self.errors.append(f"{where} is required")
            return ()
        return parse_shapes(self.data[key], where, self.errors)

    def finish(self) -> None:
        unknown = sorted(set(self.data) - self.used)
        if unknown:
            self.errors.append(f"'{self.section}' has unknown key(s): {unknown}")


def parse_shapes(value: Any, where: str, errors: list[str]) -> tuple[Shape, ...]:
    if not isinstance(value, list) or not value:
        errors.append(f"{where} must be a non-empty list of [M, N, K] triples")
        return ()
    shapes: list[Shape] = []
    for i, item in enumerate(value):
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 3
            or not all(_is_type(x, int) for x in item)
        ):
            errors.append(f"{where}[{i}] must be a list of three integers [M, N, K], got {item!r}")
            continue
        if not all(1 <= x <= MAX_DIM for x in item):
            errors.append(f"{where}[{i}] dimensions must be within [1, {MAX_DIM}], got {list(item)}")
            continue
        shape = (int(item[0]), int(item[1]), int(item[2]))
        if shape in shapes:
            errors.append(f"{where}[{i}] duplicates an earlier shape {list(shape)}")
            continue
        shapes.append(shape)
    return tuple(shapes)


def config_from_dict(data: Any) -> ExperimentConfig:
    errors: list[str] = []
    if not isinstance(data, dict):
        raise ConfigError("configuration root must be a mapping")
    sections = {"experiment", "hardware", "workload", "search", "benchmark",
                "correctness", "collect", "evaluation", "reporting"}
    unknown = sorted(set(data) - sections)
    if unknown:
        errors.append(f"unknown top-level section(s): {unknown}")
    if "experiment" not in data:
        errors.append("section 'experiment' is required")
    if "workload" not in data:
        errors.append("section 'workload' is required")

    r = _Reader(data.get("experiment"), "experiment", errors)
    experiment = ExperimentSection(
        name=r.get("name", str),
        seed=r.get("seed", int, 42, minimum=0),
        output_dir=r.get("output_dir", str, "artifacts"),
    )
    r.finish()

    r = _Reader(data.get("hardware"), "hardware", errors)
    hardware = HardwareSection(
        require_cuda=r.get("require_cuda", bool, True),
        record_device_metadata=r.get("record_device_metadata", bool, True),
    )
    r.finish()

    r = _Reader(data.get("workload"), "workload", errors)
    workload = WorkloadSection(
        kernel=r.get("kernel", str, choices=KERNELS),
        dtype=r.get("dtype", str, choices=DTYPES),
        accumulation=r.get("accumulation", str, "fp32", choices=ACCUMULATIONS),
        shapes=r.get_shapes("shapes", required=True),
    )
    r.finish()

    r = _Reader(data.get("search"), "search", errors)
    search = SearchSection(
        initial_trials=r.get("initial_trials", int, 10, minimum=1),
        max_trials=r.get("max_trials", int, 50, minimum=2),
        strategies=r.get_choices_list("strategies", STRATEGIES, STRATEGIES),
        seeds=r.get("seeds", int, 3, minimum=1, maximum=100),
        kappa=r.get("kappa", float, 1.0, minimum=0.0),
        space_file=r.get("space_file", str, ""),
    )
    r.finish()
    if isinstance(search.initial_trials, int) and isinstance(search.max_trials, int):
        if search.initial_trials >= search.max_trials:
            errors.append("search.initial_trials must be smaller than search.max_trials")
    if search.space_file and not Path(search.space_file).is_file():
        errors.append(f"search.space_file '{search.space_file}' does not exist")

    r = _Reader(data.get("benchmark"), "benchmark", errors)
    benchmark = BenchmarkSection(
        warmup_iterations=r.get("warmup_iterations", int, 10, minimum=0),
        repetitions=r.get("repetitions", int, 30, minimum=3, maximum=10_000),
        report_statistic=r.get("report_statistic", str, "median", choices=STATISTICS),
        validate_before_timing=r.get("validate_before_timing", bool, True),
        flush_l2=r.get("flush_l2", bool, True),
    )
    r.finish()

    r = _Reader(data.get("correctness"), "correctness", errors)
    correctness = CorrectnessSection(
        rtol=r.get("rtol", float, 0.01, minimum=0.0),
        atol=r.get("atol", float, 0.01, minimum=0.0),
        reject_nan_inf=r.get("reject_nan_inf", bool, True),
    )
    r.finish()

    r = _Reader(data.get("collect"), "collect", errors)
    collect = CollectSection(
        samples_per_shape=r.get("samples_per_shape", int, 150, minimum=1),
        shapes=r.get_shapes("shapes", required=False),
    )
    r.finish()

    r = _Reader(data.get("evaluation"), "evaluation", errors)
    evaluation = EvaluationSection(
        primary_metric=r.get("primary_metric", str, "best_latency_at_budget", choices=PRIMARY_METRICS),
        within_pct=r.get("within_pct", float, 5.0, minimum=0.0, maximum=100.0),
        bootstrap_resamples=r.get("bootstrap_resamples", int, 2000, minimum=100),
    )
    r.finish()

    r = _Reader(data.get("reporting"), "reporting", errors)
    reporting = ReportingSection(
        include_raw_samples=r.get("include_raw_samples", bool, True),
        include_compile_time=r.get("include_compile_time", bool, True),
        include_speedup=r.get("include_speedup", bool, True),
        include_prediction_error=r.get("include_prediction_error", bool, True),
    )
    r.finish()

    if errors:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(errors))
    return ExperimentConfig(experiment, hardware, workload, search, benchmark,
                            correctness, collect, evaluation, reporting)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse YAML in {path}: {exc}") from exc
    try:
        return config_from_dict(data)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
