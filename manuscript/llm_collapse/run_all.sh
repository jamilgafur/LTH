#!/usr/bin/env bash
# run_all.sh -- paper sweep: every model x collapse width x merge policy.
# Usage: bash run_all.sh [device]
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../.. && pwd):$(pwd)"

PY=${PYTHON:-python3}
DEVICE=${1:-auto}

for MODEL in toy-lm opt-350m pythia-410m; do
  # Transfer-style: group layers by similar activation variance and collapse
  # every identified span before one retraining phase
  for TOL in 0.1 0.25 0.5; do
    $PY run_experiment.py --model "$MODEL" \
      --span-policy variance --variance-tol "$TOL" \
      --merge mean --finetune-steps 50 --device "$DEVICE" --skip-existing
  done
  # baselines: one span of width 2 picked by relevance / at random
  for POLICY in relevance random; do
    $PY run_experiment.py --model "$MODEL" \
      --span-policy "$POLICY" --collapse-width 2 \
      --merge mean --finetune-steps 50 --device "$DEVICE" --skip-existing
  done
done

$PY plot_results.py
