#!/bin/bash

# Define the models and datasets to process
models=("VGG16" "RegNetX_400MF" "XceptionNet" "InceptionNet" "MobileNet" "ConvNeXt")
datasets=("tinyimagenet" "Cifar100" "Cifar10")

# Measurement reps per job (env-overridable, e.g. REPS=3 ./run_postprocess.sh).
# Existing CSVs with fewer than REPS real-GPU rows get resubmitted.
REPS="${REPS:-10}"

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

                # Skip only when the CSV already holds >= REPS real (non-CPU)
                # measurement rows in the current schema (the saturated-energy /
                # loader-floor column set), or when a 'no_collapsed_regions'
                # verdict row exists AND the collapsed artifact is still untrained
                # (nothing to measure; rerun once the artifact lands). CPU-only
                # rows (bad node), old-schema rows, 1-byte placeholders and
                # missing files get (re)submitted.
                if python3 - "$CSV" "$model" "$dataset" "$EPOCHS" "$PRETRAIN" "$REPS" <<'PY'
import csv, glob, sys
# NB: --quant measures the same fp32 collapsed artifact converted to fp16, so the
# existence check is quant-independent (final_..._All_Combined_whole.pt).
csv_path, model, dataset, epochs, pretrain, reps_needed = sys.argv[1:7]
try:
    with open(csv_path) as f:
        rows = [r for r in csv.reader(f) if r]
    hdr = rows[0]
    gi = hdr.index("gpu_name")
    si = hdr.index("status") if "status" in hdr else -1
    real_rows = [r for r in rows[1:]
                 if len(r) > gi and r[gi].strip() and r[gi].strip().upper() != "CPU"
                 and (si < 0 or len(r) <= si or r[si] != "no_collapsed_regions")]
    verdict = any(si >= 0 and len(r) > si and r[si] == "no_collapsed_regions"
                  for r in rows[1:])
    # Schema gate: rows measured before the saturated-energy / loader-floor
    # columns existed do not count as done, however many reps they have.
    need = ("j_per_img_saturated_collapsed", "loader_joules", "n_power_samples_collapsed")
    schema_ok = all(c in hdr for c in need)
    artifact = glob.glob(
        f"../Tranfer/{model}_{dataset}_*epochs{epochs}_pretrain{pretrain}"
        f"/checkpoints/final_JF_Dynamic_Region_All_Combined_whole.pt") or glob.glob(
        f"../Tranfer/{model}_{dataset}_*epochs{epochs}_pretrain{pretrain}"
        f"/checkpoints/final_JF_Set_*_whole.pt")
    sys.exit(0 if (len(real_rows) >= int(reps_needed) and schema_ok) or (verdict and not artifact) else 1)
except Exception:
    sys.exit(1)   # unreadable / no column / empty placeholder => run
PY
                then
                    echo "SKIP (>= $REPS GPU rows exist): $CSV"
                    continue
                fi

                # Submit the post-processing job using SGE GPU flags
                command="qsub -q all.q -l ngpus=1,gpu=TRUE -v MODEL=\"$model\",DATASET=\"$dataset\",QUANT=\"$quant\",EPOCHS=\"$EPOCHS\",PRETRAIN=\"$PRETRAIN\",REPS=\"$REPS\" submit_postprocess.pbs"

                echo "Executing: $command"
                eval "$command"

            done
        done
    done
done

echo "All jobs submitted to the queue."
