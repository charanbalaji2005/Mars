"""End-to-end orchestration with the synthetic runner (no GPU): optimize, resume, report."""
import json

import pytest

from neurotune.cli import main
from neurotune.errors import StoreError
from neurotune.experiments import run_collect, run_optimize
from neurotune.models.training import train
from neurotune.reporting.report import generate_report
from neurotune.storage.experiment_store import ExperimentStore


def test_optimize_all_strategies_and_report(config, synthetic_ctx, tmp_path):
    with ExperimentStore(config.db_path) as store:
        exp, outcomes = run_optimize(config, store=store, ctx=synthetic_ctx,
                                     strategies=["random", "learned", "tpe"], budget=8)
        assert store.get_experiment(exp)["status"] == "complete"
        assert len(outcomes) == 2 * 3 * 2  # shapes x strategies x seeds
        assert all(o.trials_used == 8 for o in outcomes)
        strategies = {t.strategy for t in store.trials(experiment_ids=[exp])}
        assert strategies == {"random", "learned", "tpe", "default", "reference"}
        report = generate_report(store, [exp], tmp_path / "report")
    text = report.read_text()
    for section in ("## Environment", "## Protocol", "## Trial accounting", "ratio vs random", "break-even",
                    "## Limitations"):
        assert section in text
    summary = json.loads((tmp_path / "report" / "summary.json").read_text())
    assert set(summary["shapes"]) == {"128x128x128", "257x511x129"}
    assert (tmp_path / "report" / "trials.csv").stat().st_size > 0


def test_interrupt_marks_experiment_and_resume_completes(config, synthetic_ctx):
    runner = synthetic_ctx.runner
    original = runner.run
    calls = {"n": 0}

    def flaky(shape, cfg):
        calls["n"] += 1
        if calls["n"] == 10:
            raise KeyboardInterrupt
        return original(shape, cfg)

    runner.run = flaky
    with ExperimentStore(config.db_path) as store:
        with pytest.raises(KeyboardInterrupt):
            run_optimize(config, store=store, ctx=synthetic_ctx, strategies=["random"], budget=6)
        exp = store.list_experiments()[0]["id"]
        assert store.get_experiment(exp)["status"] == "interrupted"
        partial = len(store.trials(experiment_ids=[exp], strategy="random"))
        runner.run = original
        with pytest.raises(StoreError, match="different protocol"):
            run_optimize(config, store=store, ctx=synthetic_ctx, strategies=["random"], budget=7, resume_id=exp)
        run_optimize(config, store=store, ctx=synthetic_ctx, strategies=["random"], budget=6, resume_id=exp)
        assert store.get_experiment(exp)["status"] == "complete"
        total = len(store.trials(experiment_ids=[exp], strategy="random"))
        assert partial < total == 2 * 2 * 6


def test_collect_train_and_prior(config, synthetic_ctx, tmp_path):
    with ExperimentStore(config.db_path) as store:
        run_collect(config, store=store, ctx=synthetic_ctx)
        collected = store.trials(strategy="collect")
        assert len(collected) == 2 * 30
    summary = train(config.db_path, tmp_path / "models")
    assert (tmp_path / "models" / "latency_predictor.joblib").exists()
    assert "heldout_shapes" in summary["evaluation"]["protocols"]
    with ExperimentStore(config.db_path) as store:
        exp, outcomes = run_optimize(config, store=store, ctx=synthetic_ctx, strategies=["learned"], budget=6,
                                     prior_dataset=str(config.db_path), seeds=1)
        assert store.get_experiment(exp)["notes"]["protocol"]["prior_measurements"] > 0


def test_cli_list_report_space(config, synthetic_ctx, tmp_path, capsys):
    with ExperimentStore(config.db_path) as store:
        exp, _ = run_optimize(config, store=store, ctx=synthetic_ctx, strategies=["random"], budget=4, seeds=1)
    assert main(["list", "--db", str(config.db_path)]) == 0
    assert main(["report", "--experiment-id", exp, "--db", str(config.db_path),
                 "--output", str(tmp_path / "r")]) == 0
    assert main(["space", "--assume-rtx4050"]) == 0
    assert "valid=" in capsys.readouterr().out
    assert main(["report", "--experiment-id", "x", "--db", str(tmp_path / "missing.db")]) == 4
    assert main(["optimize", "--strategy", "random", "--config", "nope.yaml"]) == 2
