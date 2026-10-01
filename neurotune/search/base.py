"""Common interface for search strategies (ask/tell)."""
from __future__ import annotations

import abc
from typing import Sequence

from .space import DeviceLimits, KernelConfig, Shape


class SearchStrategy(abc.ABC):
    """A strategy proposes one configuration at a time and is told the outcome.

    `observe(config, None)` reports a failed trial (compile error, incorrect
    result, out of resources ...). Replaying stored observations into a fresh
    strategy reproduces its state, which is how interrupted runs resume.
    """

    name: str = "base"
    last_prediction_ms: float | None = None

    @abc.abstractmethod
    def propose(self) -> KernelConfig | None:
        """Next configuration to benchmark, or None if the space is exhausted."""

    @abc.abstractmethod
    def observe(self, config: KernelConfig, latency_ms: float | None) -> None:
        """Record a measured latency (or None for a failed trial)."""


def make_strategy(name: str, candidates: Sequence[KernelConfig], *, shape: Shape, seed: int,
                  initial_trials: int, device: DeviceLimits, dtype_bytes: int, kappa: float = 1.0,
                  prior=None) -> SearchStrategy:
    if not candidates:
        raise ValueError(f"no valid configurations for shape {shape}")
    if name == "random":
        from .random_search import RandomSearch
        return RandomSearch(candidates, seed)
    if name == "learned":
        from .learned_search import LearnedSearch
        return LearnedSearch(candidates, shape=shape, device=device, dtype_bytes=dtype_bytes, seed=seed,
                             initial_trials=initial_trials, kappa=kappa, prior=prior)
    if name == "tpe":
        from .tpe_search import TPESearch
        return TPESearch(candidates, seed=seed, n_startup=initial_trials)
    raise ValueError(f"unknown strategy {name!r}")
