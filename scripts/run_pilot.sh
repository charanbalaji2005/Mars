#!/usr/bin/env bash
# Full pilot pipeline on one GPU. Stops at the first failing step.
set -euo pipefail
CONFIG=${1:-configs/experiments/pilot.yaml}
DB=artifacts/matmul_trials.db

neurotune doctor
neurotune validate --kernel matmul --suite smoke --config "$CONFIG"
neurotune benchmark --config "$CONFIG"
neurotune collect --config "$CONFIG"
neurotune train --dataset "$DB"
neurotune optimize --strategy all --config "$CONFIG"
neurotune list --db "$DB"
echo
echo "Generate the report with:  neurotune report --experiment-id <optimize-...> --db $DB"
