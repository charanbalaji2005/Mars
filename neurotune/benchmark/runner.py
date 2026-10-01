"""Trial execution: compile, validate, then time one configuration on one shape.

`TrialRunner` is the interface the search loop depends on. `TritonMatmulRunner`
is the real GPU implementation; tests substitute a synthetic runner so the
search/storage/reporting logic can be checked without a GPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from ..config import ExperimentConfig
from ..errors import EnvironmentCheckError, FatalGPUError
from ..search.space import DeviceLimits, KernelConfig, Shape
from ..storage.experiment_store import REFERENCE_KEY, TrialRecord
from .timing import TimingStats

FATAL_MARKERS = ("illegal memory access", "unspecified launch failure", "device-side assert",
                 "misaligned address", "illegal instruction", "cuda error: an illegal")


@dataclass
class TrialResult:
    status: str
    error: str | None = None
    first_call_s: float | None = None
    wall_s: float = 0.0
    timing: TimingStats | None = None
    max_abs_err: float | None = None
    max_rel_err: float | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def latency(self, statistic: str = "median") -> float | None:
        return self.timing.statistic(statistic) if (self.ok and self.timing) else None


class TrialRunner(Protocol):
    device_name: str

    def run(self, shape: Shape, config: KernelConfig) -> TrialResult: ...

    def run_reference(self, shape: Shape) -> TrialResult: ...

    def validate(self, shape: Shape, config: KernelConfig) -> TrialResult: ...

    def close(self) -> None: ...


def classify_exception(exc: BaseException) -> str:
    """Map an exception raised while compiling/running a trial to a trial status."""
    name = type(exc).__name__
    module = type(exc).__module__ or ""
    msg = str(exc).lower()
    if any(marker in msg for marker in FATAL_MARKERS):
        return "fatal"
    if name == "OutOfResources" or "out of resource" in msg:
        return "out_of_resources"
    if name == "OutOfMemoryError" or "out of memory" in msg:
        return "oom"
    if "compil" in name.lower() or module.startswith("triton"):
        return "compile_error"
    return "runtime_error"


def to_record(result: TrialResult, *, experiment_id: str, strategy: str, seed: int, trial_index: int,
              shape: Shape, dtype: str, config: KernelConfig | None, device_name: str,
              predicted_ms: float | None = None, include_samples: bool = True) -> TrialRecord:
    t = result.timing
    return TrialRecord(
        experiment_id=experiment_id, strategy=strategy, seed=seed, trial_index=trial_index,
        shape=tuple(shape), dtype=dtype, config_key=config.key() if config else REFERENCE_KEY,
        config=config.to_dict() if config else None, status=result.status, error=result.error,
        first_call_s=result.first_call_s, wall_s=result.wall_s,
        median_ms=t.median_ms if t else None, mean_ms=t.mean_ms if t else None,
        std_ms=t.std_ms if t else None, min_ms=t.min_ms if t else None,
        p25_ms=t.p25_ms if t else None, p75_ms=t.p75_ms if t else None,
        n_samples=t.n if t else None,
        samples_ms=list(t.samples_ms) if (t and include_samples) else None,
        max_abs_err=result.max_abs_err, max_rel_err=result.max_rel_err,
        predicted_ms=predicted_ms, device_name=device_name,
    )


class TritonMatmulRunner:
    """Runs Triton matmul trials on the current CUDA device.

    Inputs and the FP32 reference are generated once per shape and reused across
    configurations, so every configuration is compared against identical data.
    """

    def __init__(self, config: ExperimentConfig, limits: DeviceLimits):
        try:
            import torch
        except ImportError as exc:
            raise EnvironmentCheckError("PyTorch is not installed; see README 'Installation'") from exc
        if not torch.cuda.is_available():
            raise EnvironmentCheckError("CUDA is not available to PyTorch")
        from ..kernels import triton_matmul  # noqa: F401  (fails early if Triton is missing)

        self.torch = torch
        self.config = config
        self.limits = limits
        self.device_name = limits.name
        self._shape: Shape | None = None
        self.a = self.b = self.ref = self.out = None
        self._flush = None
        if config.benchmark.flush_l2:
            from .timing import make_flush_buffer
            self._flush = make_flush_buffer(limits.l2_cache_bytes)

    # ------------------------------------------------------------------ data
    def _prepare(self, shape: Shape) -> None:
        if self._shape == tuple(shape):
            return
        from ..kernels.reference import reference_fp32
        from ..workloads.matmul import MatmulWorkload

        self._release_inputs()
        wl = MatmulWorkload(tuple(shape), self.config.workload.dtype, self.config.experiment.seed)
        free, _ = self.torch.cuda.mem_get_info()
        if wl.memory_estimate_bytes() > 0.8 * free:
            raise MemoryError(f"out of memory: shape {shape} needs ~{wl.memory_estimate_bytes() / 2**20:.0f} MiB, "
                              f"{free / 2**20:.0f} MiB free")
        self.a, self.b = wl.make_inputs("cuda")
        self.ref = reference_fp32(self.a, self.b)
        self.out = self.torch.empty((shape[0], shape[1]), dtype=self.a.dtype, device="cuda")
        self._shape = tuple(shape)

    def _release_inputs(self) -> None:
        self.a = self.b = self.ref = self.out = None
        self._shape = None
        self.torch.cuda.empty_cache()

    def close(self) -> None:
        self._release_inputs()
        self._flush = None

    # ----------------------------------------------------------------- trials
    def _check(self):
        from .correctness import check

        c = self.config.correctness
        return check(self.out, self.ref, rtol=c.rtol, atol=c.atol, reject_nan_inf=c.reject_nan_inf)

    def _guard(self, fn, t0: float) -> TrialResult:
        try:
            return fn()
        except Exception as exc:  # classified, recorded, and never silently dropped
            status = classify_exception(exc)
            if status == "fatal":
                raise FatalGPUError(f"{type(exc).__name__}: {exc}") from exc
            if status == "oom":
                self._release_inputs()
            return TrialResult(status, error=f"{type(exc).__name__}: {str(exc)[:500]}",
                               wall_s=time.perf_counter() - t0)

    def validate(self, shape: Shape, config: KernelConfig) -> TrialResult:
        """Correctness only (no timing)."""
        from ..kernels import triton_matmul

        t0 = time.perf_counter()

        def body() -> TrialResult:
            self._prepare(shape)
            self.out.fill_(float("nan"))  # unwritten outputs must fail the check
            t1 = time.perf_counter()
            triton_matmul.matmul(self.a, self.b, config, out=self.out)
            self.torch.cuda.synchronize()
            first = time.perf_counter() - t1
            corr = self._check()
            return TrialResult("ok" if corr.passed else "incorrect", error=None if corr.passed else corr.message,
                               first_call_s=first, wall_s=time.perf_counter() - t0,
                               max_abs_err=corr.max_abs_err, max_rel_err=corr.max_rel_err)

        return self._guard(body, t0)

    def run(self, shape: Shape, config: KernelConfig) -> TrialResult:
        from ..kernels import triton_matmul
        from .timing import time_cuda

        bench = self.config.benchmark
        t0 = time.perf_counter()

        def body() -> TrialResult:
            self._prepare(shape)
            self.out.fill_(float("nan"))
            fn = lambda: triton_matmul.matmul(self.a, self.b, config, out=self.out)  # noqa: E731
            t1 = time.perf_counter()
            fn()  # first call: includes JIT compilation unless Triton's on-disk cache already has it
            self.torch.cuda.synchronize()
            first = time.perf_counter() - t1
            corr = None
            if bench.validate_before_timing:
                corr = self._check()
                if not corr.passed:
                    return TrialResult("incorrect", error=corr.message, first_call_s=first,
                                       wall_s=time.perf_counter() - t0,
                                       max_abs_err=corr.max_abs_err, max_rel_err=corr.max_rel_err)
            stats = time_cuda(fn, bench.warmup_iterations, bench.repetitions, self._flush)
            if corr is None:
                corr = self._check()
                if not corr.passed:
                    return TrialResult("incorrect", error=corr.message, first_call_s=first,
                                       wall_s=time.perf_counter() - t0, timing=stats,
                                       max_abs_err=corr.max_abs_err, max_rel_err=corr.max_rel_err)
            return TrialResult("ok", first_call_s=first, wall_s=time.perf_counter() - t0, timing=stats,
                               max_abs_err=corr.max_abs_err, max_rel_err=corr.max_rel_err)

        return self._guard(body, t0)

    def run_reference(self, shape: Shape) -> TrialResult:
        """Latency of torch.matmul (cuBLAS) on the same inputs, for context."""
        from .timing import time_cuda

        bench = self.config.benchmark
        t0 = time.perf_counter()

        def body() -> TrialResult:
            self._prepare(shape)
            out = self.torch.empty_like(self.out)
            fn = lambda: self.torch.matmul(self.a, self.b, out=out)  # noqa: E731
            stats = time_cuda(fn, bench.warmup_iterations, bench.repetitions, self._flush)
            return TrialResult("ok", wall_s=time.perf_counter() - t0, timing=stats)

        return self._guard(body, t0)
