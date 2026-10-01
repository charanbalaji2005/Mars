import numpy as np
import pytest

from conftest import synthetic_latency_ms
from neurotune.models.latency_predictor import (LatencyPredictor, Measurement, aggregate_measurements,
                                                 design_matrix, evaluate, kfold_splits,
                                                 leave_one_shape_out_splits, ranking_metrics)
from neurotune.search.space import SearchSpace, seeded_order

SHAPES = [(128, 128, 128), (512, 512, 512), (257, 511, 129)]


def _measurements(limits, per_shape=40):
    out = []
    for shape in SHAPES:
        cands = seeded_order(SearchSpace().valid_configs(limits, 2, shape), 0)[:per_shape]
        out += [Measurement(shape, c, synthetic_latency_ms(shape, c, limits)) for c in cands]
    return out


def test_repeated_measurements_are_aggregated(limits):
    """Leakage guard: repeats of one (shape, config) become a single row."""
    cfg = SearchSpace().valid_configs(limits, 2)[0]
    rows = [((64, 64, 64), cfg, 1.0), ((64, 64, 64), cfg, 3.0), ((64, 64, 64), cfg, 2.0), ((64, 64, 64), cfg, None)]
    agg = aggregate_measurements(rows)
    assert len(agg) == 1 and agg[0].latency_ms == 2.0 and agg[0].n_measurements == 3


def test_splits_do_not_overlap():
    for train, test in kfold_splits(23, 5, 0):
        assert not set(train) & set(test) and len(train) + len(test) == 23
    groups = [(1, 1, 1)] * 5 + [(2, 2, 2)] * 4
    for shape, train, test in leave_one_shape_out_splits(groups):
        assert all(groups[i] == shape for i in test) and all(groups[i] != shape for i in train)


def test_predictor_fit_predict_uncertainty(limits):
    ms = _measurements(limits)
    X, y, _ = design_matrix(ms, limits, 2)
    model = LatencyPredictor(n_estimators=50).fit(X, y)
    mu, sigma = model.predict_log(X)
    assert mu.shape == sigma.shape == (len(ms),) and np.all(sigma >= 0)
    assert set(model.feature_importances()) and np.all(model.predict_ms(X) > 0)


def test_evaluate_reports_both_protocols(limits):
    result = evaluate(_measurements(limits, 25), limits, 2)
    for proto in ("heldout_configs", "heldout_shapes"):
        for model in ("random_forest", "analytical_ridge"):
            m = result["protocols"][proto][model]
            assert np.isfinite(m["mape_pct"]) and 0 <= m["top5_hit_rate"] <= 1
    assert len(result["protocols"]["heldout_shapes"]["random_forest"]["per_shape"]) == 3


def test_ranking_metrics_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = ranking_metrics(y, y.copy(), [(1, 1, 1)] * 4)
    assert m["mape_pct"] == 0 and m["spearman_within_shape"] == pytest.approx(1.0) and m["top1_regret_pct"] == 0


def test_save_load(tmp_path, limits):
    ms = _measurements(limits, 10)
    X, y, _ = design_matrix(ms, limits, 2)
    model = LatencyPredictor(n_estimators=10).fit(X, y)
    model.save(tmp_path / "m.joblib", {"note": "x"})
    loaded, meta = LatencyPredictor.load(tmp_path / "m.joblib")
    assert meta["note"] == "x" and np.allclose(loaded.predict_ms(X), model.predict_ms(X))
