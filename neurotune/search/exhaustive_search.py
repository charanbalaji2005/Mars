"""Exhaustive search strategy.

Iterates systematically through all statically valid configurations in the search
space. On small configuration spaces, this serves as an empirical oracle that finds
the global optimum L*, enabling exact regret evaluation for heuristic strategies.
"""
from __future__ import annotations

from typing import Sequence

from .base import SearchStrategy
from .space import KernelConfig, seeded_order


class ExhaustiveSearch(SearchStrategy):
    name = "exhaustive"

    def __init__(self, candidates: Sequence[KernelConfig], seed: int = 0):
        # Order deterministically with seed so different seeds explore different permutations
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
