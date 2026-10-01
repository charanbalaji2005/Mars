"""Model-guided search: random initial design, then a lower-confidence-bound rule.

After `initial_trials` observations (taken from the same seeded permutation as
random search, so both start identically), a random-forest latency model is
fitted to everything measured so far. The next candidate minimizes

    mu(log latency) - kappa * sigma(log latency)

over all unmeasured valid configurations, trading exploitation (low predicted
latency) against exploration (high model disagreement).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..features.extraction import feature_matrix
from ..models.latency_predictor import LatencyPredictor, Measurement
from .base import SearchStrategy
from .space import DeviceLimits, KernelConfig, Shape, seeded_order

FAILURE_PENALTY = math.log(2.0)  # failed trials are modelled as 2x the worst success


class LearnedSearch(SearchStrategy):
    name = "learned"

    def __init__(self, candidates: Sequence[KernelConfig], *, shape: Shape, device: DeviceLimits,
                 dtype_bytes: int, seed: int, initial_trials: int = 10, kappa: float = 1.0,
                 prior: Sequence[Measurement] | None = None, n_estimators: int = 100):
        self.shape, self.seed, self.kappa = tuple(shape), seed, kappa
        self.initial_trials = initial_trials
        self.n_estimators = n_estimators
        self._cands = seeded_order(candidates, seed)
        self._X = feature_matrix([(self.shape, c) for c in self._cands], device, dtype_bytes)
        self._index = {c.key(): i for i, c in enumerate(self._cands)}
        self._obs: dict[str, float | None] = {}
        # Prior measurements of *other* shapes only; using the target shape would leak the answer.
        prior = [m for m in (prior or []) if tuple(m.shape) != self.shape]
        self.n_prior = len(prior)
        if prior:
            self._prior_X = feature_matrix([(m.shape, m.config) for m in prior], device, dtype_bytes)
            self._prior_y = np.log([m.latency_ms for m in prior])
        else:
            self._prior_X = np.zeros((0, self._X.shape[1]))
            self._prior_y = np.zeros(0)

    def observe(self, config: KernelConfig, latency_ms: float | None) -> None:
        self._obs[config.key()] = latency_ms

    def _training_data(self) -> tuple[np.ndarray, np.ndarray]:
        ok = {k: v for k, v in self._obs.items() if v is not None and k in self._index}
        failed = [k for k, v in self._obs.items() if v is None and k in self._index]
        rows, ys = [], []
        for k, v in ok.items():
            rows.append(self._index[k])
            ys.append(math.log(v))
        if failed and ys:
            penalty = max(ys) + FAILURE_PENALTY
            for k in failed:
                rows.append(self._index[k])
                ys.append(penalty)
        X = np.vstack([self._prior_X, self._X[rows]]) if rows else self._prior_X
        y = np.concatenate([self._prior_y, np.asarray(ys)])
        return X, y

    def propose(self) -> KernelConfig | None:
        self.last_prediction_ms = None
        unseen = [i for i, c in enumerate(self._cands) if c.key() not in self._obs]
        if not unseen:
            return None
        X, y = self._training_data()
        if len(self._obs) < self.initial_trials or len(y) < 2:
            return self._cands[unseen[0]]  # next element of the seeded random permutation
        model = LatencyPredictor(n_estimators=self.n_estimators, min_samples_leaf=1,
                                 seed=self.seed + len(self._obs))
        model.fit(X, np.exp(y))
        mu, sigma = model.predict_log(self._X[unseen])
        j = int(np.argmin(mu - self.kappa * sigma))
        self.last_prediction_ms = float(np.exp(mu[j]))
        return self._cands[unseen[j]]
