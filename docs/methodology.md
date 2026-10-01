# Methodology

## Research questions

- **A. Sample efficiency.** Does learned search reach a configuration within X% of the best
  measured latency using fewer GPU trials than random search?
- **B. Generalization.** Does a predictor trained on some shapes rank configurations usefully
  on unseen shapes?
- **C. Transfer.** Does prior data from other shapes reduce the trials needed on a new shape?

Fix the primary metric (`evaluation.primary_metric`), the target (`within_pct`), the budget,
and the number of seeds **before** the final run, and do not change them after seeing results.

## Search space and static pruning

Parameters: `block_m`, `block_n` ∈ {16..256}, `block_k` ∈ {16..128}, `group_m` ∈ {1,4,8},
`num_warps` ∈ {2,4,8}, `num_stages` ∈ {2..5}: 3,600 grid points. Before any GPU time,
configurations are removed if they exceed shared memory (A+B tiles × stages), imply more than
256 fp32 accumulators per thread, under-fill warps, or use blocks larger than the next power of
two of the dimension. Counts per reason are stored with each `collect` experiment. These are
heuristics and can exclude configurations that would work.

## Measurement protocol

1. Inputs: N(0,1) values generated on CPU from a seed derived from (seed, shape), identical for
   every configuration.
2. First call: timed separately (includes JIT on a cache miss); output buffer pre-filled with NaN
   so unwritten elements fail validation.
3. Validation: `|out − ref| ≤ atol + rtol·|ref|` against FP32 `torch.matmul` with TF32 disabled.
4. Timing: warm-up, then one CUDA event pair per repetition around the kernel only, with an L2
   flush (zeroing a buffer ≥ 2× L2) before each sample. The flush also hides Python launch
   latency, which would otherwise inflate tiny kernels.
5. Stored: all raw samples, median/mean/std/min/IQR, errors, wall time, device name.

Failures are classified (`compile_error`, `out_of_resources`, `oom`, `incorrect`,
`runtime_error`) and recorded. Errors that corrupt the CUDA context (illegal memory access)
stop the experiment and mark it `failed`.

## Strategies

- **random**: uniform without replacement over valid configurations.
- **learned**: same seeded initial design as random, then a random forest on log-latency;
  choose the unmeasured candidate minimizing `μ − κσ`, where σ is the spread across trees.
  Failed trials are modeled as 2× the worst success.
- **tpe**: Optuna's multivariate TPE on the same categorical space.

Budget accounting is shared: a unit is one configuration sent to the GPU.

## Predictor evaluation

Measurements are aggregated to one row per (shape, config) before any split. Protocols:
5-fold over configurations (held-out configurations) and leave-one-shape-out (unseen shapes).
Metrics: MAPE, within-shape Spearman correlation, top-1 regret (how much slower the
predicted-best configuration is than the measured best), top-5 hit rate. The analytical
ridge baseline uses only FLOPs, estimated traffic, waves and padding efficiency.

## Reporting and interpretation

Per shape and strategy: median best latency over seeds (bootstrap 95% CI), ratio vs random
(bootstrap CI), seeds reaching the target and trials needed, speedup vs the default config
and vs `torch.matmul`, tuning overhead and break-even calls. With 5 seeds intervals are wide:
claim a difference only when the ratio interval excludes 1, and report shapes where the
learned strategy loses.

## Known threats to validity

Single device; thermal and power drift on laptops; "best known" is the best measured, not
the optimum; selection bias in prediction error on chosen candidates; the Triton disk cache
hides compile cost on repeated runs.
