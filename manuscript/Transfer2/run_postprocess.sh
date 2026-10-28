#!/bin/bash

# Define the models and datasets to process
models=("VGG16" "RegNetX_400MF" "XceptionNet" "InceptionNet" "MobileNet" "ConvNeXt")
datasets=("tinyimagenet" "Cifar100" "Cifar10")

echo "=== Submitting CKA Post-Processing Jobs ==="

# Epoch configs as "EPOCHS PRETRAIN" pairs (matches temp1/temp2 arg order)
configs=("100 300" "200 200" "300 100")

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for quant in "False" "True"; do
            for cfg in "${configs[@]}"; do
                read -r EPOCHS PRETRAIN <<< "$cfg"

                # Target CSV as main.py auto-names it
                QTAG=""; [ "$quant" == "True" ] && QTAG="_quant"
                CSV="cka_results_${model}_${dataset}${QTAG}_pre${PRETRAIN}_post${EPOCHS}.csv"

                # Skip only when a non-CPU row already exists; CPU-only rows (bad
                # node) and missing files get (re)submitted.
                if python3 - "$CSV" <<'PY'
import csv, sys
try:
    rows = list(csv.reader(open(sys.argv[1])))
    hdr = rows[0]; gi = hdr.index("gpu_name")
    sys.exit(0 if any(r[gi].strip().upper() != "CPU" for r in rows[1:] if len(r) > gi) else 1)
except FileNotFoundError:
    sys.exit(1)
PY
                then
                    echo "SKIP (GPU row exists): $CSV"
                    continue
                fi

                # Submit the post-processing job using SGE GPU flags
                command="qsub -q all.q -l ngpus=1,gpu=TRUE -v MODEL=\"$model\",DATASET=\"$dataset\",QUANT=\"$quant\",EPOCHS=\"$EPOCHS\",PRETRAIN=\"$PRETRAIN\" submit_postprocess.pbs"

                echo "Executing: $command"
                eval "$command"

            done
        done
    done
done

echo "All jobs submitted to the queue."
