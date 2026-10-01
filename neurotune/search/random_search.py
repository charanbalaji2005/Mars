"""Uniform random search without replacement over statically valid configurations."""
from __future__ import annotations

from typing import Sequence

from .base import SearchStrategy
from .space import KernelConfig, seeded_order


class RandomSearch(SearchStrategy):
    name = "random"

    def __init__(self, candidates: Sequence[KernelConfig], seed: int):
        self._order = seeded_order(candidates, seed)
        self._pos = 0
        self._seen: set[str] = set()

    def propose(self) -> KernelConfig | None:
        self.last_prediction_ms = None
        while self._pos < len(self._order):
            cfg = self._order[self._pos]
            self._pos += 1
            if cfg.key() not in self._seen:
                return cfg
        return None

    def observe(self, config: KernelConfig, latency_ms: float | None) -> None:
        self._seen.add(config.key())
