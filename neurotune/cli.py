"""NeuroTune command-line interface.

Exit codes: 0 success · 1 check/validation failed · 2 invalid usage or configuration
· 3 environment problem · 4 storage problem · 5 fatal GPU error · 130 interrupted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .errors import ConfigError, EnvironmentCheckError, FatalGPUError, NeuroTuneError, StoreError
from .logging_utils import get_logger, setup_logging

DEFAULT_CONFIG_PATH = "configs/experiments/pilot.yaml"
log = get_logger("cli")


def _shape(text: str) -> tuple[int, int, int]:
    try:
        parts = tuple(int(x) for x in text.lower().replace("x", ",").split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid shape {text!r}; use M,N,K or MxNxK") from None
    if len(parts) != 3 or min(parts) < 1:
        raise argparse.ArgumentTypeError(f"invalid shape {text!r}; use three positive integers")
    return parts  # type: ignore[return-value]


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neurotune", description="ML-guided Triton kernel autotuning research harness")
    p.add_argument("--version", action="version", version=f"neurotune {__version__}")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-json", action="store_true", help="emit JSON log lines on stderr")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="verify CUDA, Triton, and GPU compatibility")
    d.add_argument("--json", action="store_true", help="print results as JSON")

    v = sub.add_parser("validate", help="kernel correctness tests (no timing)")
    v.add_argument("--kernel", default="matmul", choices=["matmul"])
    v.add_argument("--suite", default="smoke", choices=["smoke", "full"])
    v.add_argument("--configs-per-shape", type=int, default=4, help="sampled configs in addition to the default")
    v.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="tolerances and dtype are read from here")

    b = sub.add_parser("benchmark", help="benchmark the default config and torch.matmul")
    b.add_argument("--config", default=DEFAULT_CONFIG_PATH)

    c = sub.add_parser("collect", help="collect a dataset of measured configurations")
    c.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    c.add_argument("--samples-per-shape", type=_positive)
    c.add_argument("--resume", metavar="EXPERIMENT_ID")

    t = sub.add_parser("train", help="train and evaluate the latency predictor")
    t.add_argument("--dataset", required=True, help="path to the experiment database (e.g. artifacts/matmul_trials.db)")
    t.add_argument("--output", default="artifacts/models")
    t.add_argument("--device", help="device name, if the dataset contains several")
    t.add_argument("--dtype", choices=["fp16", "bf16"])
    t.add_argument("--include-optimize", action="store_true", help="also train on trials from optimize runs")
    t.add_argument("--seed", type=int, default=0)

    o = sub.add_parser("optimize", help="run a search strategy under a fixed GPU-trial budget")
    o.add_argument("--strategy", required=True, choices=["random", "learned", "tpe", "all"])
    o.add_argument("--budget", type=_positive, help="GPU trials per (shape, seed); default: search.max_trials")
    o.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    o.add_argument("--seeds", type=_positive, help="independent repetitions; default: search.seeds")
    o.add_argument("--initial-trials", type=_positive)
    o.add_argument("--shape", type=_shape, action="append", dest="shapes", help="restrict to a shape (repeatable)")
    o.add_argument("--prior-dataset", help="database whose collect trials (other shapes only) warm-start learned search")
    o.add_argument("--resume", metavar="EXPERIMENT_ID")

    r = sub.add_parser("report", help="generate an experiment report")
    r.add_argument("--experiment-id", required=True, nargs="+")
    r.add_argument("--db", default="artifacts/matmul_trials.db")
    r.add_argument("--output", help="default: <db dir>/reports/<first id>")

    ls = sub.add_parser("list", help="list experiments in a database")
    ls.add_argument("--db", default="artifacts/matmul_trials.db")

    sp = sub.add_parser("space", help="inspect the search space and static pruning (no GPU needed)")
    sp.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    sp.add_argument("--assume-rtx4050", action="store_true", help="use nominal RTX 4050 Laptop limits")
    return p


# ------------------------------------------------------------------------ commands
def cmd_doctor(args) -> int:
    from .hardware.discovery import doctor

    checks = doctor()
    if args.json:
        print(json.dumps([c.__dict__ for c in checks], indent=2))
    else:
        marks = {"ok": "PASS", "warn": "WARN", "fail": "FAIL"}
        for c in checks:
            print(f"  {marks[c.status]}  {c.name:<14} {c.detail}")
    failed = [c for c in checks if c.status == "fail"]
    if failed:
        print(f"\n{len(failed)} check(s) failed.", file=sys.stderr)
        return 1
    print("\nEnvironment OK.")
    return 0


def cmd_validate(args) -> int:
    from .config import load_config
    from .experiments import gpu_context, run_validate
    from .workloads.matmul import SUITES

    config = load_config(args.config)
    ctx = gpu_context(config)
    results = run_validate(config, ctx=ctx, shapes=SUITES[args.suite], sampled_configs=args.configs_per_shape)
    failed = [r for r in results if r["status"] != "ok"]
    for r in results:
        mark = "PASS" if r["status"] == "ok" else "FAIL"
        print(f"  {mark}  {'x'.join(map(str, r['shape'])):<16} {r['config']:<32} "
              f"max_abs_err={r['max_abs_err'] if r['max_abs_err'] is not None else '—'}"
              + (f"  [{r['status']}] {r['error']}" if r["status"] != "ok" else ""))
    out = Path(config.experiment.output_dir) / "validation" / f"{args.kernel}_{args.suite}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": ctx.limits.to_dict(), "results": results}, indent=2))
    print(f"\n{len(results) - len(failed)}/{len(results)} passed · details: {out}")
    ctx.runner.close()
    return 1 if failed else 0


def _with_store(config):
    from .storage.experiment_store import ExperimentStore

    return ExperimentStore(config.db_path)


def cmd_benchmark(args) -> int:
    from .config import load_config
    from .experiments import gpu_context, run_benchmark

    config = load_config(args.config)
    ctx = gpu_context(config)
    with _with_store(config) as store:
        exp_id = run_benchmark(config, store=store, ctx=ctx)
        _print_baselines(store, exp_id, config.benchmark.report_statistic)
    print(f"\nexperiment: {exp_id}\nreport:     neurotune report --experiment-id {exp_id} --db {config.db_path}")
    return 0


def _print_baselines(store, exp_id, statistic) -> None:
    rows = store.trials(experiment_ids=[exp_id])
    print(f"\n  {'shape':<16} {'default Triton (ms)':>20} {'torch.matmul (ms)':>18}")
    for shape in sorted({r.shape for r in rows}):
        d = next((r for r in rows if r.shape == shape and r.strategy == "default"), None)
        t = next((r for r in rows if r.shape == shape and r.strategy == "reference"), None)
        dv = f"{d.latency(statistic):.4f}" if d and d.ok else (d.status if d else "—")
        tv = f"{t.latency(statistic):.4f}" if t and t.ok else (t.status if t else "—")
        print(f"  {'x'.join(map(str, shape)):<16} {dv:>20} {tv:>18}")


def cmd_collect(args) -> int:
    from .config import load_config
    from .experiments import gpu_context, run_collect

    config = load_config(args.config)
    ctx = gpu_context(config)
    with _with_store(config) as store:
        exp_id = run_collect(config, store=store, ctx=ctx, resume_id=args.resume,
                             samples_per_shape=args.samples_per_shape)
    print(f"\nexperiment: {exp_id}\nnext:       neurotune train --dataset {config.db_path}")
    return 0


def cmd_train(args) -> int:
    from .models.training import train

    summary = train(args.dataset, args.output, device_name=args.device, dtype=args.dtype,
                    include_optimize=args.include_optimize, seed=args.seed)
    print(f"dataset: {summary['unique_pairs']} unique (shape, config) pairs from {summary['raw_trials']} trials, "
          f"{summary['evaluation']['n_shapes']} shapes, device {summary['device']}")
    for proto, models in summary["evaluation"]["protocols"].items():
        print(f"\n  {proto}")
        if "skipped" in models:
            print(f"    skipped: {models['skipped']}")
            continue
        print(f"    {'model':<18} {'MAPE %':>8} {'Spearman':>9} {'top-1 regret %':>15} {'top-5 hit':>10}")
        for name, m in models.items():
            print(f"    {name:<18} {m['mape_pct']:>8.1f} {m['spearman_within_shape']:>9.3f} "
                  f"{m['top1_regret_pct']:>15.1f} {m['top5_hit_rate']:>10.2f}")
    print(f"\nmodel + evaluation written to {args.output}/")
    return 0


def cmd_optimize(args) -> int:
    from .config import load_config
    from .experiments import gpu_context, run_optimize

    config = load_config(args.config)
    strategies = list(config.search.strategies) if args.strategy == "all" else [args.strategy]
    budget = args.budget or config.search.max_trials
    ctx = gpu_context(config)
    with _with_store(config) as store:
        exp_id, outcomes = run_optimize(config, store=store, ctx=ctx, strategies=strategies, budget=budget,
                                        seeds=args.seeds, initial_trials=args.initial_trials, shapes=args.shapes,
                                        prior_dataset=args.prior_dataset, resume_id=args.resume)
    print(f"\n  {'shape':<16} {'strategy':<9} {'seed':>10} {'trials':>7} {'best ms':>10}  best config")
    for o in outcomes:
        best = f"{o.best_ms:.4f}" if o.best_ms is not None else "—"
        print(f"  {'x'.join(map(str, o.shape)):<16} {o.strategy:<9} {o.seed:>10} {o.trials_used:>7} {best:>10}  "
              f"{o.best_key or '—'}")
    print(f"\nexperiment: {exp_id}\nreport:     neurotune report --experiment-id {exp_id} --db {config.db_path}")
    return 0


def cmd_report(args) -> int:
    from .reporting.report import generate_report
    from .storage.experiment_store import ExperimentStore

    with ExperimentStore(args.db, create=False) as store:
        out = Path(args.output) if args.output else Path(args.db).parent / "reports" / args.experiment_id[0]
        path = generate_report(store, args.experiment_id, out)
    print(f"report written to {path} (plus trials.csv, summary.json{', convergence.png' if (out / 'convergence.png').exists() else ''})")
    return 0


def cmd_list(args) -> int:
    from .storage.experiment_store import ExperimentStore

    with ExperimentStore(args.db, create=False) as store:
        exps = store.list_experiments()
        if not exps:
            print("no experiments")
        for e in exps:
            n = len(store.trials(experiment_ids=[e["id"]]))
            print(f"  {e['id']:<38} {e['kind']:<10} {e['status']:<12} {n:>6} trials  {e['created_at']}")
    return 0


def cmd_space(args) -> int:
    from .config import load_config
    from .search.space import DTYPE_BYTES, DeviceLimits, load_space

    config = load_config(args.config)
    space = load_space(config.search.space_file)
    if args.assume_rtx4050:
        limits = DeviceLimits.assumed_rtx4050_laptop()
    else:
        from .hardware.discovery import device_limits
        try:
            limits = device_limits()
        except Exception as exc:
            raise EnvironmentCheckError(f"no GPU available ({exc}); pass --assume-rtx4050 to plan offline") from exc
    dtype_bytes = DTYPE_BYTES[config.workload.dtype]
    print(f"device: {limits.name}\nraw grid: {space.raw_size} configurations")
    print(f"hardware-valid: {len(space.valid_configs(limits, dtype_bytes))}")
    for shape in sorted(set(config.workload.shapes) | set(config.collect_shapes)):
        report = space.prune_report(limits, dtype_bytes, shape)
        pruned = ", ".join(f"{k}={v}" for k, v in report.items() if k not in ("raw_size", "valid"))
        print(f"  {'x'.join(map(str, shape)):<16} valid={report.get('valid', 0):<5} pruned: {pruned}")
    return 0


COMMANDS = {"doctor": cmd_doctor, "validate": cmd_validate, "benchmark": cmd_benchmark, "collect": cmd_collect,
            "train": cmd_train, "optimize": cmd_optimize, "report": cmd_report, "list": cmd_list, "space": cmd_space}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, json_console=args.log_json)
    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        print("\ninterrupted; partial results are saved and the experiment can be resumed with --resume",
              file=sys.stderr)
        return 130
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except EnvironmentCheckError as exc:
        print(f"environment error: {exc}", file=sys.stderr)
        return 3
    except StoreError as exc:
        print(f"storage error: {exc}", file=sys.stderr)
        return 4
    except FatalGPUError as exc:
        print(f"fatal GPU error (experiment marked failed; restart the process): {exc}", file=sys.stderr)
        return 5
    except NeuroTuneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
