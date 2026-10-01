"""Analytical latency model-guided search strategy.

Uses the analytical features (FLOPs, traffic, wave efficiency, padding efficiency)
fitted via Ridge regression as a baseline search heuristic. This provides a direct
scientific baseline to test whether the learned Random Forest model outperforms
a simpler, interpretable physics-inspired linear model under equivalent budgets.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..features.extraction import feature_matrix
from ..models.latency_predictor import AnalyticalBaseline
from .base import SearchStrategy
from .space import DeviceLimits, KernelConfig, Shape, seeded_order


class AnalyticalSearch(SearchStrategy):
    name = "analytical"

    def __init__(self, candidates: Sequence[KernelConfig], *, shape: Shape, device: DeviceLimits,
                 dtype_bytes: int, seed: int, initial_trials: int = 10):
        self.shape, self.seed = tuple(shape), seed
        self.initial_trials = initial_trials
        self._cands = seeded_order(candidates, seed)
        self._X = feature_matrix([(self.shape, c) for c in self._cands], device, dtype_bytes)
        self._index = {c.key(): i for i, c in enumerate(self._cands)}
        self._obs: dict[str, float | None] = {}

    def observe(self, config: KernelConfig, latency_ms: float | None) -> None:
        self._obs[config.key()] = latency_ms

    def _training_data(self) -> tuple[np.ndarray, np.ndarray]:
        ok = {k: v for k, v in self._obs.items() if v is not None and k in self._index}
        rows, ys = [], []
        for k, v in ok.items():
            rows.append(self._index[k])
            ys.append(v)
        return self._X[rows] if rows else np.zeros((0, self._X.shape[1])), np.asarray(ys)

    def propose(self) -> KernelConfig | None:
        self.last_prediction_ms = None
        unseen = [i for i, c in enumerate(self._cands) if c.key() not in self._obs]
        if not unseen:
            return None
        X, y = self._training_data()
        if len(self._obs) < self.initial_trials or len(y) < 2:
            return self._cands[unseen[0]]
        model = AnalyticalBaseline(alpha=1.0)
        model.fit(X, y)
        preds = model.predict_ms(self._X[unseen])
        j = int(np.argmin(preds))
        self.last_prediction_ms = float(preds[j])
        return self._cands[unseen[j]]
