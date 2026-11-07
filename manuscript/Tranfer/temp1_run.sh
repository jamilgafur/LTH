#!/bin/bash

# --- Input Validation ---
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <discovery_epochs> <hpc_compute_epochs> [scorer]"
    echo "  scorer: local_dev (default) | percentile | cv | change | cka | all"
    echo "  'all' is discovery-only: one probe -> per-scorer region maps + ablation JSON."
    echo "Example: $0 50 100 change"
    exit 1
fi

EPOCHS=$1
PRETRAIN=$2
SCORER=${3:-local_dev}

case "$SCORER" in
    local_dev|percentile|cv|change|cka|all) ;;
    *) echo "Error: unknown scorer '$SCORER' (expected local_dev|percentile|cv|change|cka|all)"; exit 1 ;;
esac

# 1. Keep previous SGE stdout/stderr (~/submit*) — they are the only evidence
#    when the node SIGKILLs a job (qacct is unavailable here).

models=("VGG16" "RegNetX_400MF" "XceptionNet" "InceptionNet" "MobileNet" "ConvNeXt")
#models=("MobileNet")
#"ConvNeXt" "MobileNet" "InceptionNet" "XceptionNet" "RegNetX_400MF" "VGG16")
datasets=("tinyimagenet" "Cifar100" "Cifar10")
quant=("False" "True")

echo "=== Submitting Stage 1: Discovery Jobs ==="
echo "    Discovery Budget (Epochs): $EPOCHS | HPC Target (Pretrain): $PRETRAIN | Scorer: $SCORER"

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for quant_flag in "${quant[@]}"; do
            for flag in "JF"; do

                # --- Skip completed discovery: the deliverable is the region-map JSON ---
                if [ "$SCORER" = "local_dev" ]; then
                    json_done="${model}_${dataset}_epochs${EPOCHS}_pretrain${PRETRAIN}_${flag}_discovered_regions.json"
                    suffixed_done="${model}_${dataset}_epochs${EPOCHS}_pretrain${PRETRAIN}_scorer${SCORER}_${flag}_discovered_regions.json"
                elif [ "$SCORER" = "all" ]; then
                    # 'all' never writes a _scorerall_ map; its deliverable is the ablation JSON
                    json_done="${model}_${dataset}_epochs${EPOCHS}_pretrain${PRETRAIN}_scorer_ablation.json"
                    suffixed_done=""
                else
                    json_done="${model}_${dataset}_epochs${EPOCHS}_pretrain${PRETRAIN}_scorer${SCORER}_${flag}_discovered_regions.json"
                    suffixed_done=""
                fi
                qtag=""; [ "$quant_flag" = "True" ] && qtag="_quant"
                map_done=false
                [ -f "$json_done" ] && map_done=true
                { [ -n "$suffixed_done" ] && [ -f "$suffixed_done" ]; } && map_done=true
                # Control is scorer-independent: any universe's final counts (Transfer2 borrows it)
                control_done=false
                compgen -G "${model}_${dataset}_*_epochs${EPOCHS}_pretrain${PRETRAIN}/checkpoints/final_JF_Control${qtag}.pt" > /dev/null && control_done=true
                if $map_done && $control_done; then
                    echo "[SKIP] $model/$dataset discovery map already exists: $json_done"
                    continue
                fi

                # Submit discovery job. Will train/inherit Control if needed, then dump JSON
                command="qsub -q UI-GPU -l ngpus=1,gpu=TRUE -v MODEL=\"$model\",DATASET=\"$dataset\",EXPERIMENT=\"discover\",FLAG=\"$flag\",QUANT=\"$quant_flag\",EPOCHS=\"$EPOCHS\",PRETRAIN=\"$PRETRAIN\",SCORER=\"$SCORER\" submit_job.pbs"
                
                echo "Executing: $command"
                eval "$command"
                
            done
        done
    done
done
