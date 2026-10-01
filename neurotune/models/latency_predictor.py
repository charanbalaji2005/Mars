"""Latency prediction models and leakage-safe evaluation.

* `LatencyPredictor` — random forest on log-latency; the spread of per-tree
  predictions is used as an (approximate) uncertainty estimate for search.
* `AnalyticalBaseline` — ridge regression on operation count, estimated memory
  traffic and occupancy proxies only. A learned model has to beat this to be useful.

Leakage control: repeated measurements of the same (shape, config) are first
aggregated into one row, so a configuration can never appear in both the
training and the evaluation split. Generalization to unseen shapes is evaluated
with leave-one-shape-out splits.
"""
from __future__ import annotations

import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from ..features.extraction import ANALYTICAL_FEATURES, FEATURE_NAMES, feature_matrix
from ..search.space import DeviceLimits, KernelConfig, Shape


@dataclass(frozen=True)
class Measurement:
    shape: Shape
    config: KernelConfig
    latency_ms: float
    n_measurements: int = 1


def aggregate_measurements(rows: Iterable[tuple[Shape, KernelConfig, float]]) -> list[Measurement]:
    """Collapse repeated measurements of one (shape, config) into a single median row."""
    groups: dict[tuple[Shape, KernelConfig], list[float]] = defaultdict(list)
    for shape, cfg, latency in rows:
        if latency is not None and math.isfinite(latency) and latency > 0:
            groups[(tuple(shape), cfg)].append(float(latency))
    return [Measurement(shape, cfg, float(np.median(v)), len(v))
            for (shape, cfg), v in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1]))]


def design_matrix(measurements: Sequence[Measurement], device: DeviceLimits, dtype_bytes: int,
                  names: Sequence[str] = FEATURE_NAMES) -> tuple[np.ndarray, np.ndarray, list[Shape]]:
    X = feature_matrix([(m.shape, m.config) for m in measurements], device, dtype_bytes, names)
    y = np.array([m.latency_ms for m in measurements], dtype=np.float64)
    return X, y, [m.shape for m in measurements]


class LatencyPredictor:
    def __init__(self, n_estimators: int = 200, min_samples_leaf: int = 2,
                 max_features: float = 0.7, seed: int = 0):
        from sklearn.ensemble import RandomForestRegressor

        self.feature_names = FEATURE_NAMES
        self.model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                           max_features=max_features, random_state=seed, n_jobs=1)
        self.fitted = False

    def fit(self, X: np.ndarray, latency_ms: np.ndarray) -> "LatencyPredictor":
        if len(X) < 2:
            raise ValueError("need at least two measurements to fit a predictor")
        self.model.fit(X, np.log(latency_ms))
        self.fitted = True
        return self

    def predict_log(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and std of log-latency across trees."""
        per_tree = np.stack([tree.predict(X) for tree in self.model.estimators_])
        return per_tree.mean(axis=0), per_tree.std(axis=0)

    def predict_ms(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.model.predict(X))

    def feature_importances(self) -> dict[str, float]:
        return dict(sorted(zip(self.feature_names, map(float, self.model.feature_importances_)),
                           key=lambda kv: -kv[1]))

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": list(self.feature_names),
                     "metadata": metadata or {}}, path)

    @classmethod
    def load(cls, path: str | Path) -> tuple["LatencyPredictor", dict]:
        import joblib

        blob = joblib.load(path)
        if tuple(blob["feature_names"]) != FEATURE_NAMES:
            raise ValueError("saved model was trained with a different feature set; retrain it")
        obj = cls.__new__(cls)
        obj.feature_names, obj.model, obj.fitted = FEATURE_NAMES, blob["model"], True
        return obj, blob["metadata"]


class AnalyticalBaseline:
    """Ridge regression on log-latency using only analytical features."""

    def __init__(self, alpha: float = 1.0):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.columns = [FEATURE_NAMES.index(n) for n in ANALYTICAL_FEATURES]
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, X: np.ndarray, latency_ms: np.ndarray) -> "AnalyticalBaseline":
        self.model.fit(X[:, self.columns], np.log(latency_ms))
        return self

    def predict_ms(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.model.predict(X[:, self.columns]))


# --------------------------------------------------------------------------- metrics
def ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray, groups: Sequence[Shape]) -> dict[str, float]:
    """Error and ranking quality. Ranking metrics are computed within each shape,
    because the tuner only ever has to rank configurations for one workload."""
    from scipy.stats import spearmanr

    ape = np.abs(y_pred - y_true) / y_true
    by_group: dict[Shape, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[tuple(g)].append(i)
    spearmans, regrets, top5 = [], [], []
    for idx in by_group.values():
        idx = np.asarray(idx)
        if len(idx) < 3:
            continue
        t, p = y_true[idx], y_pred[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = spearmanr(t, p).statistic
        if np.isfinite(rho):
            spearmans.append(float(rho))
        regrets.append(float(t[np.argmin(p)] / t.min() - 1.0) * 100.0)
        top5.append(float(np.argmin(t) in set(np.argsort(p)[:5])))
    nan = float("nan")
    return {
        "n": int(len(y_true)),
        "mape_pct": float(ape.mean() * 100),
        "median_ape_pct": float(np.median(ape) * 100),
        "spearman_within_shape": float(np.mean(spearmans)) if spearmans else nan,
        "top1_regret_pct": float(np.mean(regrets)) if regrets else nan,
        "top5_hit_rate": float(np.mean(top5)) if top5 else nan,
    }


def kfold_splits(n: int, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Held-out configurations: folds over unique (shape, config) rows."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, min(n_folds, n))
    return [(np.setdiff1d(perm, f), np.sort(f)) for f in folds if len(f)]


def leave_one_shape_out_splits(groups: Sequence[Shape]) -> list[tuple[Shape, np.ndarray, np.ndarray]]:
    """Held-out shapes: the evaluation shape is never seen during training."""
    groups = [tuple(g) for g in groups]
    out = []
    for shape in sorted(set(groups)):
        test = np.array([i for i, g in enumerate(groups) if g == shape])
        train = np.array([i for i, g in enumerate(groups) if g != shape])
        out.append((shape, train, test))
    return out


ModelFactory = Callable[[], object]


def _oof(factory: ModelFactory, X, y, splits) -> np.ndarray:
    pred = np.full(len(y), np.nan)
    for train, test in splits:
        if len(train) < 2:
            continue
        model = factory()
        model.fit(X[train], y[train])  # type: ignore[attr-defined]
        pred[test] = model.predict_ms(X[test])  # type: ignore[attr-defined]
    return pred


def evaluate(measurements: Sequence[Measurement], device: DeviceLimits, dtype_bytes: int,
             seed: int = 0, n_folds: int = 5) -> dict:
    """Compare the learned predictor with the analytical baseline on two protocols."""
    if len(measurements) < 10:
        raise ValueError(f"need at least 10 unique measurements to evaluate, got {len(measurements)}")
    X, y, groups = design_matrix(measurements, device, dtype_bytes)
    models: dict[str, ModelFactory] = {
        "random_forest": lambda: LatencyPredictor(seed=seed),
        "analytical_ridge": lambda: AnalyticalBaseline(),
    }
    result: dict = {"n_measurements": len(measurements), "n_shapes": len(set(groups)), "protocols": {}}

    splits = kfold_splits(len(y), n_folds, seed)
    result["protocols"]["heldout_configs"] = {
        name: ranking_metrics(y, _oof(f, X, y, splits), groups) for name, f in models.items()}

    if len(set(groups)) >= 2:
        loso = leave_one_shape_out_splits(groups)
        proto: dict = {}
        for name, factory in models.items():
            pred = _oof(factory, X, y, [(tr, te) for _, tr, te in loso])
            proto[name] = ranking_metrics(y, pred, groups)
            proto[name]["per_shape"] = {
                "x".join(map(str, shape)): ranking_metrics(y[te], pred[te], [shape] * len(te))
                for shape, _, te in loso}
        result["protocols"]["heldout_shapes"] = proto
    else:
        result["protocols"]["heldout_shapes"] = {"skipped": "need at least two distinct shapes"}
    return result
