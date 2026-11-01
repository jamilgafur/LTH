#!/bin/bash

# Define the models and datasets to process
models=("VGG16" "RegNetX_400MF" "XceptionNet" "InceptionNet" "MobileNet" "ConvNeXt")
datasets=("tinyimagenet" "Cifar100" "Cifar10")

echo "=== Submitting CKA Post-Processing Jobs ==="

# Epoch configs as "EPOCHS PRETRAIN" pairs (matches temp1/temp2 arg order)
configs=("100 300")
#"200 200" "300 100")

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for quant in "False" "True"; do
            for cfg in "${configs[@]}"; do
                read -r EPOCHS PRETRAIN <<< "$cfg"

                # Target CSV as main.py auto-names it
                QTAG=""; [ "$quant" == "True" ] && QTAG="_quant"
                CSV="cka_results_${model}_${dataset}${QTAG}_pre${PRETRAIN}_post${EPOCHS}.csv"

                # Skip when a real non-CPU measurement row already exists, or when
                # a 'no_collapsed_regions' verdict row exists AND the collapsed
                # artifact is still untrained (nothing to measure; rerun once the
                # artifact lands). CPU-only rows (bad node), 1-byte placeholders
                # and missing files get (re)submitted.
                if python3 - "$CSV" "$model" "$dataset" "$EPOCHS" "$PRETRAIN" <<'PY'
import csv, glob, sys
# NB: --quant measures the same fp32 collapsed artifact converted to fp16, so the
# existence check is quant-independent (final_..._All_Combined_whole.pt).
csv_path, model, dataset, epochs, pretrain = sys.argv[1:6]
try:
    with open(csv_path) as f:
        rows = [r for r in csv.reader(f) if r]
    hdr = rows[0]
    gi = hdr.index("gpu_name")
    si = hdr.index("status") if "status" in hdr else -1
    real_gpu = any(
        len(r) > gi and r[gi].strip() and r[gi].strip().upper() != "CPU"
        and (si < 0 or len(r) <= si or r[si] != "no_collapsed_regions")
        for r in rows[1:])
    verdict = any(si >= 0 and len(r) > si and r[si] == "no_collapsed_regions"
                  for r in rows[1:])
    artifact = glob.glob(
        f"../Tranfer/{model}_{dataset}_*epochs{epochs}_pretrain{pretrain}"
        f"/checkpoints/final_JF_Dynamic_Region_All_Combined_whole.pt") or glob.glob(
        f"../Tranfer/{model}_{dataset}_*epochs{epochs}_pretrain{pretrain}"
        f"/checkpoints/final_JF_Set_*_whole.pt")
    sys.exit(0 if real_gpu or (verdict and not artifact) else 1)
except Exception:
    sys.exit(1)   # unreadable / no column / empty placeholder => run
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
