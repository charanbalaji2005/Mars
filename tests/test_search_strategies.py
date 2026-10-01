import pytest

from conftest import SyntheticRunner, synthetic_latency_ms
from neurotune.search.base import SearchStrategy, make_strategy
from neurotune.search.optimizer import run_search
from neurotune.search.space import SearchSpace
from neurotune.storage.experiment_store import ExperimentStore

SHAPE = (257, 511, 129)


def _cands(limits):
    return SearchSpace().valid_configs(limits, 2, SHAPE)


def _drive(strategy, limits, n):
    seen = []
    for _ in range(n):
        cfg = strategy.propose()
        if cfg is None:
            break
        seen.append(cfg)
        strategy.observe(cfg, synthetic_latency_ms(SHAPE, cfg, limits))
    return seen


@pytest.mark.parametrize("name", ["random", "learned", "tpe", "exhaustive", "analytical"])
def test_strategies_propose_unique_valid_configs(name, limits):
    cands = _cands(limits)
    s = make_strategy(name, cands, shape=SHAPE, seed=3, initial_trials=4, device=limits, dtype_bytes=2)
    seen = _drive(s, limits, 15)
    assert len(seen) == 15 and len({c.key() for c in seen}) == 15
    assert all(c in set(cands) for c in seen)


def test_learned_initial_design_matches_random(limits):
    cands = _cands(limits)
    kw = dict(shape=SHAPE, seed=11, initial_trials=6, device=limits, dtype_bytes=2)
    r = _drive(make_strategy("random", cands, **kw), limits, 6)
    l = _drive(make_strategy("learned", cands, **kw), limits, 6)
    assert r == l


def test_learned_reports_predictions_after_initial(limits):
    s = make_strategy("learned", _cands(limits), shape=SHAPE, seed=0, initial_trials=3, device=limits, dtype_bytes=2)
    _drive(s, limits, 3)
    assert s.propose() is not None and s.last_prediction_ms and s.last_prediction_ms > 0


def test_random_exhausts_small_space(limits):
    cands = _cands(limits)[:5]
    s = make_strategy("random", cands, shape=SHAPE, seed=0, initial_trials=2, device=limits, dtype_bytes=2)
    assert len(_drive(s, limits, 10)) == 5 and s.propose() is None


class _Repeater(SearchStrategy):
    """Proposes each config twice to check duplicates are not charged to the budget."""
    name = "random"

    def __init__(self, cands):
        self.queue = [c for c in cands for _ in range(2)]

    def propose(self):
        return self.queue.pop(0) if self.queue else None

    def observe(self, config, latency_ms):
        pass


def test_budget_and_duplicates(tmp_path, limits):
    runner = SyntheticRunner(limits)
    with ExperimentStore(tmp_path / "db") as store:
        exp = store.create_experiment("optimize", "t", {}, {}, {})
        out = run_search(store=store, experiment_id=exp, runner=runner, strategy=_Repeater(_cands(limits)),
                         shape=SHAPE, dtype="fp16", seed=0, budget=6)
        assert out.trials_used == 6 and runner.calls == 6 and out.duplicates_skipped == 5
        assert len(store.trials(experiment_ids=[exp])) == 6


@pytest.mark.parametrize("name", ["random", "learned", "tpe", "exhaustive", "analytical"])
def test_resume_continues_to_budget(tmp_path, limits, name):
    cands = _cands(limits)
    kw = dict(shape=SHAPE, seed=5, initial_trials=3, device=limits, dtype_bytes=2)
    with ExperimentStore(tmp_path / "db") as store:
        exp = store.create_experiment("optimize", "t", {}, {}, {})
        first = run_search(store=store, experiment_id=exp, runner=SyntheticRunner(limits),
                           strategy=make_strategy(name, cands, **kw), shape=SHAPE, dtype="fp16", seed=5, budget=5)
        runner = SyntheticRunner(limits)
        second = run_search(store=store, experiment_id=exp, runner=runner,
                            strategy=make_strategy(name, cands, **kw), shape=SHAPE, dtype="fp16", seed=5, budget=9)
        assert first.trials_used == 5 and second.resumed_trials == 5 and second.trials_used == 9
        assert runner.calls == 4
        keys = [t.config_key for t in store.trials(experiment_ids=[exp])]
        assert len(keys) == len(set(keys)) == 9


def test_resume_random_is_deterministic(tmp_path, limits):
    """Interrupted-and-resumed random search measures exactly what an uninterrupted run would."""
    cands = _cands(limits)
    kw = dict(shape=SHAPE, seed=9, initial_trials=3, device=limits, dtype_bytes=2)
    with ExperimentStore(tmp_path / "db") as store:
        a = store.create_experiment("optimize", "a", {}, {}, {})
        run_search(store=store, experiment_id=a, runner=SyntheticRunner(limits),
                   strategy=make_strategy("random", cands, **kw), shape=SHAPE, dtype="fp16", seed=9, budget=8)
        b = store.create_experiment("optimize", "b", {}, {}, {})
        for budget in (3, 8):
            run_search(store=store, experiment_id=b, runner=SyntheticRunner(limits),
                       strategy=make_strategy("random", cands, **kw), shape=SHAPE, dtype="fp16", seed=9, budget=budget)
        key = lambda e: [t.config_key for t in sorted(store.trials(experiment_ids=[e]), key=lambda t: t.trial_index)]
        assert key(a) == key(b)
