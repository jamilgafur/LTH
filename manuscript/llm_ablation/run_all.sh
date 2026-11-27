#!/usr/bin/env bash
# run_all.sh -- paper sweep: every model x query set x region ablation.
# Usage: bash run_all.sh [device]
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../.. && pwd):$(pwd)"

PY=${PYTHON:-python3}
DEVICE=${1:-auto}

for MODEL in toy-moe mixtral-8x7b olmoe-1b-7b; do
  for QS in domains; do
    # whole experts removed everywhere, ranked by domain correlation
    for KEEP in 0.75 0.5 0.25; do
      $PY run_experiment.py --model "$MODEL" --query-set "$QS" \
        --ablation experts --keep-fraction "$KEEP" \
        --rank-by max_corr --protect-domains all \
        --device "$DEVICE" --skip-existing
    done
    # contiguous layer spans collapsed (Transfer-style)
    for K in 1 2 4; do
      $PY run_experiment.py --model "$MODEL" --query-set "$QS" \
        --ablation layers --collapse-layers "$K" \
        --rank-by max_corr --protect-domains all \
        --device "$DEVICE" --skip-existing
    done
    # multi-layer x multi-expert regions
    for KEEP in 0.5 0.25; do
      $PY run_experiment.py --model "$MODEL" --query-set "$QS" \
        --ablation region --collapse-layers 2 --keep-fraction "$KEEP" \
        --rank-by max_corr --protect-domains all \
        --device "$DEVICE" --skip-existing
    done
  done
done

$PY plot_results.py
$PY dashboard.py --runs runs
