"""GPU timing with CUDA events, warm-up, repeated samples and optional L2 flushing."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class TimingStats:
    samples_ms: tuple[float, ...]
    median_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    p25_ms: float
    p75_ms: float

    @property
    def n(self) -> int:
        return len(self.samples_ms)

    @property
    def cv(self) -> float:
        return self.std_ms / self.mean_ms if self.mean_ms > 0 else float("nan")

    def statistic(self, name: str) -> float:
        return {"median": self.median_ms, "mean": self.mean_ms, "min": self.min_ms}[name]

    def to_dict(self) -> dict:
        return asdict(self)


def summarize(samples_ms: Sequence[float]) -> TimingStats:
    arr = np.asarray(samples_ms, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot summarize an empty list of timing samples")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("timing samples must be finite and non-negative")
    return TimingStats(
        samples_ms=tuple(float(x) for x in arr),
        median_ms=float(np.median(arr)),
        mean_ms=float(arr.mean()),
        std_ms=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        min_ms=float(arr.min()),
        p25_ms=float(np.percentile(arr, 25)),
        p75_ms=float(np.percentile(arr, 75)),
    )


def make_flush_buffer(l2_cache_bytes: int):
    """A buffer at least twice the L2 size; zeroing it evicts the kernel's operands."""
    import torch

    nbytes = max(2 * int(l2_cache_bytes), 32 * 1024**2)
    return torch.empty(nbytes // 4, dtype=torch.int32, device="cuda")


def time_cuda(fn: Callable[[], object], warmup: int, repetitions: int, flush_buffer=None) -> TimingStats:
    """Time `fn` with CUDA events: one event pair per repetition, kernel only.

    Zeroing the flush buffer before each sample also keeps the GPU busy while the
    CPU enqueues the start event and kernel, so Python launch overhead does not
    leak into the measured interval for small kernels.
    """
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    for i in range(repetitions):
        if flush_buffer is not None:
            flush_buffer.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return summarize([s.elapsed_time(e) for s, e in zip(starts, ends)])
