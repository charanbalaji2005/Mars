# NeuroTune

ML-guided autotuning research harness for a parameterized Triton matrix-multiplication kernel.

NeuroTune measures Triton matmul configurations on your GPU, stores every trial (including
failures), trains a latency predictor, and compares a model-guided search against random
search and TPE under identical GPU-trial budgets. The goal is a falsifiable answer to one
question: *does learned search reach a good configuration with fewer benchmarks than random
search, on this hardware?* A "no" is a valid result.

## Status

| Part | State |
| --- | --- |
| Config validation, search space, features, predictor, search strategies, SQLite store, resume, reports, CLI | Implemented; covered by the CPU test suite (`pytest`, 55 tests) |
| Triton kernel, CUDA timing, correctness harness, device discovery | Implemented; **not yet run on a GPU**. Verify with `neurotune doctor` and `neurotune validate` before trusting any measurement |
| Dashboard, second kernel family, CUDA C++ backend | Deliberately deferred (see `dashboard/README.md`) |

No performance numbers are claimed. The tests use a synthetic latency function only to
exercise the search/storage/report logic.

## Requirements

- Linux or **WSL2** (Triton does not officially support native Windows).
- NVIDIA GPU with compute capability ≥ 7.0 (RTX 4050 = 8.9) and a recent driver.
- Python ≥ 3.10.
- PyTorch with CUDA. On Linux the PyTorch CUDA wheel already includes a matching Triton.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
# 1. PyTorch with CUDA — pick the command for your driver at https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu124
# 2. NeuroTune
pip install -e ".[dev,plots]"
# 3. Check everything
neurotune doctor
pytest            # GPU tests run automatically when CUDA is available
```

## Workflow

Run from the repository root. Each step maps to a milestone in the plan.

```bash
neurotune doctor                                            # M1: environment
neurotune validate --kernel matmul --suite smoke            # M1: correctness incl. non-aligned shapes
neurotune benchmark --config configs/experiments/pilot.yaml # M1: default config vs torch.matmul
neurotune collect   --config configs/experiments/pilot.yaml # M2: dataset (150 configs/shape)
neurotune train     --dataset artifacts/matmul_trials.db    # M3: predictor + held-out evaluation
neurotune optimize  --strategy all --budget 50              # M4: random vs learned vs TPE, 5 seeds
neurotune report    --experiment-id <optimize-id>           # M4/M5: report.md, trials.csv, summary.json, plot
```

`scripts/run_pilot.sh` runs the whole sequence. `neurotune list` shows experiment IDs.
`neurotune space --assume-rtx4050` prints the search space and pruning without a GPU.

Run strategies separately if you prefer (`--strategy random`, `learned`, `tpe`) and compare them
with `neurotune report --experiment-id ID1 ID2 ID3`.

### Interrupted runs

Every trial is committed atomically. Ctrl-C marks the experiment `interrupted`; a killed
process leaves it `running`, and reports flag both as incomplete. Continue with
`--resume EXPERIMENT_ID` and the same arguments. Resuming refuses to proceed if the config,
protocol or GPU changed.

### Transfer experiment

`configs/experiments/transfer_generalization.yaml` collects data on 8 training shapes and
evaluates search on 3 held-out shapes, optionally warm-starting learned search with
`--prior-dataset`. The learned strategy never uses prior data from the shape it is tuning.

## Methodology in brief

- **Budget** = configurations sent to the GPU, failed ones included. Statically invalid and
  duplicate proposals cost nothing. Same rule for every strategy.
- **Fairness**: random and learned search share the same seeded initial design.
- **Timing**: CUDA events, warm-up, 30 repetitions, L2 flush between samples, median reported,
  all raw samples stored. Inputs are validated against an FP32 reference (TF32 off) before timing.
- **Predictor evaluation**: repeated measurements are aggregated per (shape, config) before
  splitting, so nothing leaks. Two protocols: held-out configurations and leave-one-shape-out.
  The random forest is compared against an analytical ridge baseline (FLOPs, traffic, waves, padding).
- **Statistics**: per shape and strategy, the median best latency over seeds with a bootstrap
  95% CI, the ratio vs random search with a CI, how many seeds reached the target (within 5% of
  the best measured), tuning overhead and break-even call count.

Details: `docs/methodology.md`. Architecture: `docs/architecture.md`.

## Repository layout

```text
configs/            experiment configs and the search space
neurotune/
  cli.py            command-line interface
  config.py         config loading and validation (before any GPU work)
  experiments.py    orchestration: benchmark, collect, optimize, validate
  hardware/         device discovery, doctor checks, nvidia-smi snapshots
  workloads/        matmul workload and test suites
  kernels/          Triton kernel and PyTorch reference
  benchmark/        runner, timing, correctness
  features/         analytical features per (shape, config)
  models/           latency predictor, baseline, evaluation, training
  search/           space, random, learned (LCB), TPE, budgeted search loop
  storage/          SQLite experiment store
  reporting/        report generation
tests/              CPU tests + GPU correctness tests
docs/               methodology and architecture
scripts/            pipeline script
```

## Exit codes

0 success · 1 check failed · 2 bad config/usage · 3 environment · 4 storage · 5 fatal GPU error · 130 interrupted.

## Laptop GPU tips (RTX 4050)

Plug in to AC power and use the high-performance power mode; laptop GPUs change clocks with
temperature and power limits. Each experiment records `nvidia-smi` clocks, temperature and
power state at start and end, so compare them when results look unstable. `first_call_s`
includes JIT compilation only on a Triton cache miss; set `TRITON_CACHE_DIR` to an empty
directory to measure cold compilation.
