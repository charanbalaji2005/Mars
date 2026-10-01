import pytest

from neurotune.errors import StoreError
from neurotune.storage.experiment_store import ExperimentStore, TrialRecord


def _rec(exp, key="bm64_bn64_bk32_g8_w4_s3", status="ok", ms=1.0, idx=0):
    return TrialRecord(experiment_id=exp, strategy="random", seed=0, trial_index=idx, shape=(64, 64, 64),
                       dtype="fp16", config_key=key, status=status,
                       config={"block_m": 64, "block_n": 64, "block_k": 32, "group_m": 8, "num_warps": 4,
                               "num_stages": 3},
                       median_ms=ms if status == "ok" else None, samples_ms=[ms, ms], device_name="gpu")


def test_roundtrip_and_queries(tmp_path):
    with ExperimentStore(tmp_path / "db.sqlite") as store:
        exp = store.create_experiment("optimize", "t", {"a": 1}, {"limits": {}}, {"py": "3"}, fingerprint="f")
        store.record_trial(_rec(exp))
        store.record_trial(_rec(exp, key="other", status="compile_error", idx=1))
        rows = store.trials(experiment_ids=[exp])
        assert [r.status for r in rows] == ["ok", "compile_error"]
        assert rows[0].samples_ms == [1.0, 1.0] and rows[0].kernel_config.block_m == 64
        assert store.trials(status="ok", kinds=["optimize"])[0].config_key == rows[0].config_key
        assert store.trials(kinds=["collect"]) == []
        assert store.best_known((64, 64, 64), "fp16", "gpu") == 1.0
        assert store.get_experiment(exp)["status"] == "running"
        store.set_status(exp, "interrupted", note="ctrl-c")
        assert store.get_experiment(exp)["notes"]["last_status_note"] == "ctrl-c"


def test_duplicates_rejected(tmp_path):
    with ExperimentStore(tmp_path / "db.sqlite") as store:
        exp = store.create_experiment("optimize", "t", {}, {}, {})
        store.record_trial(_rec(exp))
        with pytest.raises(StoreError, match="duplicate"):
            store.record_trial(_rec(exp, idx=5))


def test_missing_db_and_experiment(tmp_path):
    with pytest.raises(StoreError):
        ExperimentStore(tmp_path / "missing.db", create=False)
    with ExperimentStore(tmp_path / "db.sqlite") as store:
        with pytest.raises(StoreError):
            store.get_experiment("nope")
