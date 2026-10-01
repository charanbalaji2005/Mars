# Architecture

```text
CLI (cli.py)
  └─ experiments.py ── session(): status running → complete | interrupted | failed
        ├─ config.py            validated, frozen ExperimentConfig (+ fingerprint for resume)
        ├─ hardware/discovery   DeviceLimits, metadata, doctor
        ├─ search/space         grid + static pruning → candidate list per shape
        ├─ search/optimizer     budgeted ask/tell loop, duplicate guard, replay-based resume
        │     └─ search/{random_search, learned_search, tpe_search}
        │            └─ models/latency_predictor ← features/extraction
        ├─ benchmark/runner     TrialRunner protocol; TritonMatmulRunner
        │     ├─ kernels/triton_matmul, kernels/reference
        │     ├─ benchmark/timing, benchmark/correctness
        │     └─ workloads/matmul
        ├─ storage/experiment_store   SQLite: experiments + trials, one transaction per trial
        └─ reporting/report     report.md, trials.csv, summary.json, convergence.png
```

## Key interfaces

- `TrialRunner.run(shape, config) -> TrialResult`: the only thing the search loop needs from
  the GPU. Tests substitute a synthetic runner.
- `SearchStrategy.propose() / observe(config, latency_or_None)`: strategies are rebuilt by
  replaying stored observations, which is how resume works without serializing models.

## Adding a kernel family

1. Add `kernels/<name>.py` and a reference implementation.
2. Add a workload in `workloads/` and a search space (YAML or a `SearchSpace` subclass).
3. Add features in `features/extraction.py` (or a family-specific module).
4. Implement a `TrialRunner` for it and register the kernel name in `config.KERNELS`.

## Data model

`experiments(id, kind, status, config_json, hardware_json, software_json, notes_json)` and
`trials(experiment_id, strategy, seed, trial_index, m, n, k, dtype, config_key, status, timing…,
samples_json, predicted_ms, device_name)`, unique on
`(experiment_id, strategy, seed, m, n, k, dtype, config_key)`.
Strategies `default` and `reference` hold the per-experiment baselines.
