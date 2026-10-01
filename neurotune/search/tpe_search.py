"""Tree-structured Parzen Estimator baseline (Optuna) over the same discrete space.

Statically invalid proposals are reported to Optuna as failed and re-asked; they
never reach the GPU and never consume benchmark budget. Proposals of already
measured configurations are answered from the cache, also without consuming budget.
"""
from __future__ import annotations

import math
import random
import warnings
from typing import Sequence

from .base import SearchStrategy
from .space import PARAM_NAMES, KernelConfig

MAX_ASK_ATTEMPTS = 500


class TPESearch(SearchStrategy):
    name = "tpe"

    def __init__(self, candidates: Sequence[KernelConfig], seed: int, n_startup: int = 10):
        import optuna
        from optuna.distributions import CategoricalDistribution

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self._optuna = optuna
        self._valid = {c.key(): c for c in candidates}
        self._dists = {name: CategoricalDistribution(sorted({getattr(c, name) for c in candidates}))
                       for name in PARAM_NAMES}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup, multivariate=True)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)
        self._pending: dict[str, object] = {}
        self._seen: dict[str, float | None] = {}
        self._rng = random.Random(seed)
        self.fallback_proposals = 0

    def propose(self) -> KernelConfig | None:
        from optuna.trial import TrialState

        self.last_prediction_ms = None
        if len(self._seen) >= len(self._valid):
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(MAX_ASK_ATTEMPTS):
                trial = self.study.ask(self._dists)
                cfg = KernelConfig(**{n: int(trial.params[n]) for n in PARAM_NAMES})
                key = cfg.key()
                if key not in self._valid:
                    self.study.tell(trial, state=TrialState.FAIL)
                    continue
                if key in self._seen:
                    value = self._seen[key]
                    if value is None:
                        self.study.tell(trial, state=TrialState.FAIL)
                    else:
                        self.study.tell(trial, value)
                    continue
                if key in self._pending:
                    self.study.tell(trial, state=TrialState.FAIL)
                    continue
                self._pending[key] = trial
                return cfg
        # TPE keeps proposing measured points: fall back to an unmeasured valid one.
        self.fallback_proposals += 1
        remaining = sorted(k for k in self._valid if k not in self._seen and k not in self._pending)
        return self._valid[self._rng.choice(remaining)] if remaining else None

    def observe(self, config: KernelConfig, latency_ms: float | None) -> None:
        from optuna.trial import TrialState, create_trial

        key = config.key()
        value = math.log(latency_ms) if latency_ms else None
        self._seen[key] = value
        trial = self._pending.pop(key, None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if trial is not None:
                if value is None:
                    self.study.tell(trial, state=TrialState.FAIL)
                else:
                    self.study.tell(trial, value)
            else:  # replayed history (resume) or a fallback proposal
                params = config.to_dict()
                if all(params[n] in self._dists[n].choices for n in PARAM_NAMES):
                    if value is None:
                        self.study.add_trial(create_trial(params=params, distributions=self._dists,
                                                          state=TrialState.FAIL))
                    else:
                        self.study.add_trial(create_trial(params=params, distributions=self._dists, value=value))
