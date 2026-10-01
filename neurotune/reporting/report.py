"""Experiment reports: Markdown summary, raw trial CSV, JSON summary, optional plot.

Reports state what was measured, under which protocol, including failures and
cases where a strategy did *not* help. "Best known" latency is the best measured
on this device in the whole database, not a proven optimum.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import STRATEGIES
from ..storage.experiment_store import ExperimentStore, TrialRecord

NAN = float("nan")


# ------------------------------------------------------------------------- helpers
def _fmt(x, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}g}"
    return str(x)


def _table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(_fmt(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _shape_str(shape) -> str:
    return "x".join(map(str, shape))


def bootstrap_ci(values: Sequence[float], n_resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if arr.size < 2:
        return NAN, NAN
    idx = np.random.default_rng(seed).integers(0, arr.size, size=(n_resamples, arr.size))
    stats = np.median(arr[idx], axis=1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def bootstrap_ratio_ci(a: Sequence[float], b: Sequence[float], n_resamples: int = 2000,
                       seed: int = 0) -> tuple[float, float, float]:
    """Ratio of medians a/b with a 95% percentile-bootstrap interval (independent resampling)."""
    a = np.asarray([v for v in a if v is not None], dtype=float)
    b = np.asarray([v for v in b if v is not None], dtype=float)
    if a.size == 0 or b.size == 0:
        return NAN, NAN, NAN
    point = float(np.median(a) / np.median(b))
    if a.size < 2 or b.size < 2:
        return point, NAN, NAN
    rng = np.random.default_rng(seed)
    ra = np.median(a[rng.integers(0, a.size, size=(n_resamples, a.size))], axis=1)
    rb = np.median(b[rng.integers(0, b.size, size=(n_resamples, b.size))], axis=1)
    ratio = ra / rb
    return point, float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))


def trajectory(trials: Sequence[TrialRecord], statistic: str) -> list[float | None]:
    best, out = None, []
    for t in sorted(trials, key=lambda r: r.trial_index):
        lat = t.latency(statistic) if t.ok else None
        if lat is not None and (best is None or lat < best):
            best = lat
        out.append(best)
    return out


def trials_to_target(traj: Sequence[float | None], target: float | None) -> int | None:
    if target is None:
        return None
    for i, v in enumerate(traj, start=1):
        if v is not None and v <= target:
            return i
    return None


def _median(values) -> float:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.median(vals)) if vals else NAN


# -------------------------------------------------------------------------- report
def generate_report(store: ExperimentStore, experiment_ids: Sequence[str], output_dir: str | Path) -> Path:
    exps = [store.get_experiment(i) for i in experiment_ids]
    trials = store.trials(experiment_ids=list(experiment_ids))
    cfg = exps[0]["config"]
    statistic = cfg["benchmark"]["report_statistic"]
    ev, rep = cfg["evaluation"], cfg["reporting"]
    within = float(ev["within_pct"])
    n_boot = int(ev.get("bootstrap_resamples", 2000))
    dtype = cfg["workload"]["dtype"]
    device = exps[0]["hardware"].get("limits", {}).get("name")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    md: list[str] = [f"# NeuroTune report — {', '.join(experiment_ids)}", ""]
    incomplete = [e for e in exps if e["status"] != "complete"]
    if incomplete:
        md += ["> **Incomplete data.** " + "; ".join(
            f"`{e['id']}` is `{e['status']}`" + (" (process ended without cleanup)" if e["status"] == "running" else "")
            for e in incomplete) + ". Results below cover only the trials that were recorded.", ""]
    fingerprints = {e["config_fingerprint"] for e in exps}
    if len(fingerprints) > 1:
        md += ["> **Warning:** these experiments used different configurations; comparisons may not be like-for-like.", ""]

    md += ["## Experiments", "", _table(["id", "kind", "status", "created (UTC)", "trials"],
           [[e["id"], e["kind"], e["status"], e["created_at"],
             sum(1 for t in trials if t.experiment_id == e["id"])] for e in exps]), ""]

    # Environment
    hw, sw = exps[0]["hardware"], exps[0]["software"]
    lim = hw.get("limits", {})
    md += ["## Environment", "",
           _table(["item", "value"], [
               ["device", lim.get("name")],
               ["compute capability", ".".join(map(str, lim.get("compute_capability", [])))],
               ["SMs", lim.get("sm_count")],
               ["shared memory / block (KiB)", (lim.get("max_shared_mem_per_block") or 0) // 1024],
               ["L2 (MiB)", (lim.get("l2_cache_bytes") or 0) // 2**20],
               *[[k, v] for k, v in sw.items()],
           ]), ""]
    smi = hw.get("nvidia_smi_at_start")
    if smi:
        md += ["nvidia-smi at start: " + ", ".join(f"{k}={v}" for k, v in smi[0].items()), ""]

    # Protocol
    protocol = next((e["notes"].get("protocol") for e in exps if e["notes"].get("protocol")), None)
    md += ["## Protocol", "",
           f"- Workload: `{cfg['workload']['kernel']}`, {dtype} inputs, {cfg['workload']['accumulation']} accumulation.",
           f"- Timing: {cfg['benchmark']['warmup_iterations']} warm-up iterations, "
           f"{cfg['benchmark']['repetitions']} timed repetitions, statistic = **{statistic}**, "
           f"L2 flush = {cfg['benchmark'].get('flush_l2', True)}, "
           f"validated before timing = {cfg['benchmark']['validate_before_timing']}.",
           f"- Correctness: rtol={cfg['correctness']['rtol']}, atol={cfg['correctness']['atol']} "
           f"vs. an FP32 reference (TF32 disabled); NaN/Inf rejected = {cfg['correctness']['reject_nan_inf']}.",
           f"- Pre-registered primary metric: **{ev['primary_metric']}**; target = within {within}% "
           "of the best latency measured on this device."]
    if protocol:
        md += [f"- Search: strategies {protocol['strategies']}, budget {protocol['budget']} GPU trials per "
               f"(shape, strategy, seed), {protocol['seeds']} seeds, {protocol['initial_trials']} initial "
               f"random trials, kappa={protocol['kappa']}, prior measurements={protocol['prior_measurements']}.",
               "- Budget counts every configuration sent to the GPU, including failed ones. Statically "
               "invalid and duplicate proposals cost nothing."]
    md.append("")

    # Trial accounting
    by_status = defaultdict(Counter)
    for t in trials:
        by_status[t.strategy][t.status] += 1
    statuses = sorted({t.status for t in trials})
    md += ["## Trial accounting", "", _table(["strategy", *statuses, "total"],
           [[s, *[c.get(st, 0) for st in statuses], sum(c.values())] for s, c in sorted(by_status.items())]), ""]
    errors = Counter((t.status, (t.error or "")[:160]) for t in trials if not t.ok)
    if errors:
        md += ["Most common failures:", ""]
        md += [f"- `{st}` ×{n}: {msg or '(no message)'}" for (st, msg), n in errors.most_common(8)]
        md.append("")

    shapes = sorted({t.shape for t in trials})
    summary: dict = {"experiments": experiment_ids, "statistic": statistic, "within_pct": within, "shapes": {}}
    convergence: dict = {}

    md += ["## Results by shape", ""]
    for shape in shapes:
        st = [t for t in trials if t.shape == shape]
        default_ms = _median(t.latency(statistic) for t in st if t.strategy == "default" and t.ok)
        torch_ms = _median(t.latency(statistic) for t in st if t.strategy == "reference" and t.ok)
        best_known = store.best_known(shape, dtype, device, statistic)
        target = best_known * (1 + within / 100) if best_known else None
        ss: dict = {"default_ms": default_ms, "torch_matmul_ms": torch_ms, "best_known_ms": best_known,
                    "strategies": {}}
        md += [f"### {_shape_str(shape)}", "",
               f"Default Triton config: {_fmt(default_ms)} ms · torch.matmul: {_fmt(torch_ms)} ms · "
               f"best known on this device: {_fmt(best_known)} ms · target (≤ +{within}%): {_fmt(target)} ms", ""]

        # Search strategies
        per_strategy: dict[str, dict] = {}
        for strat in [s for s in STRATEGIES if any(t.strategy == s for t in st)]:
            runs = defaultdict(list)
            for t in st:
                if t.strategy == strat:
                    runs[t.seed].append(t)
            bests, hits, overheads, trajs = [], [], [], {}
            for seed, rs in sorted(runs.items()):
                traj = trajectory(rs, statistic)
                trajs[seed] = traj
                bests.append(traj[-1] if traj else None)
                hits.append(trials_to_target(traj, target))
                overheads.append(sum(r.wall_s or 0.0 for r in rs))
            per_strategy[strat] = {"bests": bests, "hits": hits, "trajectories": trajs,
                                   "overhead_s": overheads, "trials": [r for rs in runs.values() for r in rs]}
            convergence.setdefault(shape, {})[strat] = trajs

        if per_strategy:
            rows = []
            random_bests = per_strategy.get("random", {}).get("bests")
            for strat, d in per_strategy.items():
                med = _median(d["bests"])
                lo, hi = bootstrap_ci(d["bests"], n_boot)
                reached = [h for h in d["hits"] if h is not None]
                ratio = bootstrap_ratio_ci(d["bests"], random_bests, n_boot) if (random_bests and strat != "random") \
                    else (NAN, NAN, NAN)
                rows.append([strat, len(d["bests"]), med, f"[{_fmt(lo)}, {_fmt(hi)}]",
                             f"{_fmt(ratio[0])} [{_fmt(ratio[1])}, {_fmt(ratio[2])}]" if strat != "random" else "1 (ref)",
                             f"{len(reached)}/{len(d['hits'])}",
                             _median(reached) if reached else None,
                             default_ms / med if rep.get("include_speedup", True) and med == med else None,
                             torch_ms / med if rep.get("include_speedup", True) and med == med else None])
                ss["strategies"][strat] = {
                    "best_ms_per_seed": d["bests"], "median_best_ms": med, "ci95_median": [lo, hi],
                    "ratio_vs_random": list(ratio), "trials_to_target_per_seed": d["hits"],
                    "tuning_overhead_s_per_seed": d["overhead_s"]}
            md += [_table(["strategy", "seeds", f"best {statistic} ms (median over seeds)", "95% CI",
                           "ratio vs random [95% CI]", "reached target", "trials to target (median)",
                           "speedup vs default", "torch.matmul time / best (>1 = faster than torch)"], rows), "",
                   "Ratio < 1 means lower latency than random search. With few seeds the intervals are wide; "
                   "overlapping intervals mean the data do not show a difference.", ""]

            # Best configs and their timing distributions
            rows = []
            for strat, d in per_strategy.items():
                ok = [t for t in d["trials"] if t.ok]
                if not ok:
                    rows.append([strat, "no successful trial", None, None, None, None, None])
                    continue
                b = min(ok, key=lambda t: t.latency(statistic))
                cv = (b.std_ms / b.mean_ms) if b.mean_ms else None
                rows.append([strat, f"`{b.config_key}`", b.median_ms, f"{_fmt(b.p25_ms)}–{_fmt(b.p75_ms)}", cv,
                             b.n_samples, b.first_call_s if rep.get("include_compile_time", True) else None])
            md += [_table(["strategy", "best config (all seeds)", "median ms", "IQR ms", "CV", "samples",
                           "first call s (incl. JIT)"], rows), ""]

            # Overhead / break-even
            rows = []
            for strat, d in per_strategy.items():
                over = float(np.mean(d["overhead_s"])) if d["overhead_s"] else NAN
                med = _median(d["bests"])
                saved_ms = default_ms - med if (default_ms == default_ms and med == med) else NAN
                breakeven = over / (saved_ms / 1000) if saved_ms == saved_ms and saved_ms > 0 else None
                rows.append([strat, over, saved_ms, breakeven if breakeven else "never (no saving)"])
            md += [_table(["strategy", "tuning wall time per run (s)", "saving per call vs default (ms)",
                           "break-even calls"], rows), ""]

            # Prediction error
            if rep.get("include_prediction_error", True) and "learned" in per_strategy:
                pairs = [(t.predicted_ms, t.latency(statistic)) for t in per_strategy["learned"]["trials"]
                         if t.ok and t.predicted_ms]
                if pairs:
                    ape = [abs(p - m) / m * 100 for p, m in pairs]
                    md += [f"Learned-search predictions for the configurations it chose: n={len(pairs)}, "
                           f"MAPE={_fmt(float(np.mean(ape)))}%, median APE={_fmt(float(np.median(ape)))}%. "
                           "These are errors on *selected* candidates, which are biased toward optimistic predictions.", ""]
                    ss["learned_prediction_mape_pct"] = float(np.mean(ape))

        # Dataset collection summary
        coll = [t for t in st if t.strategy == "collect"]
        if coll:
            ok = [t for t in coll if t.ok]
            lats = sorted(t.latency(statistic) for t in ok)
            best = min(ok, key=lambda t: t.latency(statistic)) if ok else None
            md += [_table(["collected", "ok", "best ms", "median ms", "worst ms", "worst/best", "best config"],
                          [[len(coll), len(ok), lats[0] if lats else None, _median(lats),
                            lats[-1] if lats else None, (lats[-1] / lats[0]) if lats else None,
                            f"`{best.config_key}`" if best else None]]), ""]
            ss["collect"] = {"n": len(coll), "ok": len(ok), "best_ms": lats[0] if lats else None}
        summary["shapes"][_shape_str(shape)] = ss

    # Notes for the reader
    md += ["## Limitations", "",
           "- Results are for one device, driver, and software stack (listed above); they do not transfer automatically.",
           "- \"Best known\" is the best configuration *measured*, not a proven optimum of the space.",
           "- Each configuration is timed in one session; run-to-run drift (clocks, temperature, power limits on "
           "laptops) is not removed. Compare nvidia-smi snapshots at start and end, and rerun to check stability.",
           "- Static pruning (shared memory, register pressure, oversized blocks) uses heuristics and may exclude "
           "configurations that would have worked.",
           ""]

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(md))
    _write_csv(trials, output_dir / "trials.csv", include_samples=rep.get("include_raw_samples", True))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _plot(convergence, store, dtype, device, statistic, output_dir / "convergence.png")
    return report_path


def _write_csv(trials: Sequence[TrialRecord], path: Path, include_samples: bool) -> None:
    fields = ["experiment_id", "strategy", "seed", "trial_index", "m", "n", "k", "dtype", "config_key", "status",
              "error", "median_ms", "mean_ms", "std_ms", "min_ms", "p25_ms", "p75_ms", "n_samples",
              "first_call_s", "wall_s", "max_abs_err", "max_rel_err", "predicted_ms", "device_name", "created_at"]
    if include_samples:
        fields.append("samples_ms")
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for t in trials:
            row = {f: getattr(t, f, None) for f in fields if f not in ("m", "n", "k", "samples_ms")}
            row.update(m=t.shape[0], n=t.shape[1], k=t.shape[2])
            if include_samples:
                row["samples_ms"] = json.dumps(t.samples_ms) if t.samples_ms else ""
            writer.writerow(row)


def _plot(convergence: dict, store: ExperimentStore, dtype: str, device, statistic: str, path: Path) -> None:
    if not convergence:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    shapes = sorted(convergence)
    fig, axes = plt.subplots(1, len(shapes), figsize=(4.5 * len(shapes), 3.6), squeeze=False)
    for ax, shape in zip(axes[0], shapes):
        best = store.best_known(shape, dtype, device, statistic) or 1.0
        for strat, trajs in convergence[shape].items():
            length = max(len(t) for t in trajs.values())
            mat = np.full((len(trajs), length), np.nan)
            for i, t in enumerate(trajs.values()):
                vals = [np.nan if v is None else v / best for v in t]
                mat[i, :len(vals)] = vals
            with np.errstate(all="ignore"):
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    med = np.nanmedian(mat, axis=0)
                    lo, hi = np.nanmin(mat, axis=0), np.nanmax(mat, axis=0)
            x = np.arange(1, length + 1)
            ax.plot(x, med, label=strat)
            ax.fill_between(x, lo, hi, alpha=0.15)
        ax.set_title(_shape_str(shape))
        ax.set_xlabel("GPU trials")
        ax.set_ylabel("best-so-far / best known")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
