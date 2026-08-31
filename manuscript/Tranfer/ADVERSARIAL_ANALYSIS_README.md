# Adversarial Robustness Analysis Integration

This document explains how to integrate the adversarial analysis pipeline with the transfer learning framework.

## Overview

The adversarial analysis pipeline evaluates the robustness of pruned (collapsed) vs. original (uncollapsed) models using multiple attack methods. It supports:

- **Multiple attacks**: PGD, FGSM, IFGSM, BIM, APGD, CW, DeepFool, AutoAttack, Square
- **Transferability measurement**: How well attacks transfer between different model architectures and pruning states
- **Parallel HPC execution**: Distributed attack generation across multiple GPUs
- **Comprehensive visualizations**: Attack success rates, transferability heatmaps, attack comparisons

It also supports a single-source workflow: generate one adversarial dataset from one source model variant and evaluate transfer against all other models with the same dataset.

## Quick Start

### Option 1: Local Execution (Single Machine)

```bash
# Full pipeline: generate attacks → analyze transferability → plot results
python manuscript/Tranfer/adversarial_analysis.py --mode full

# Generate attacks only (fastest for multiple runs)
python manuscript/Tranfer/adversarial_analysis.py --mode generate

# Single-source generation (example)
python manuscript/Tranfer/adversarial_analysis.py --mode generate \
  --model InceptionNet --dataset Cifar10 --attack PGD --kind Finetuned \
  --output-dir adversarial_results/inception_c10_ft_pgd

# Analyze transferability (requires pre-generated attacks)
python manuscript/Tranfer/adversarial_analysis.py --mode analyze

# Analyze transfer from single-source dataset against all same-dataset targets
python manuscript/Tranfer/adversarial_analysis.py --mode analyze \
  --model InceptionNet --dataset Cifar10 --attack PGD --kind Finetuned \
  --output-dir adversarial_results/inception_c10_ft_pgd

# Generate plots (requires summary.csv and transferability.csv)
python manuscript/Tranfer/adversarial_analysis.py --mode plot
```

### Option 2: HPC Parallel Execution

```bash
# Make scripts executable
chmod +x manuscript/Tranfer/adversarial_hpc_orchestrate.sh
chmod +x manuscript/Tranfer/adversarial_hpc_submit.pbs

# Submit parallel generation jobs
./manuscript/Tranfer/adversarial_hpc_orchestrate.sh generate adversarial_results

# Submit a single-source generation job (model/dataset/attack/kind)
./manuscript/Tranfer/adversarial_hpc_orchestrate.sh generate \
  adversarial_results/inception_c10_ft_pgd InceptionNet Cifar10 PGD Finetuned

# Monitor progress
qstat

# Once generation completes, run analysis and plotting
./manuscript/Tranfer/adversarial_hpc_orchestrate.sh analyze adversarial_results
./manuscript/Tranfer/adversarial_hpc_orchestrate.sh plot adversarial_results

# Single-source analyze and plot
./manuscript/Tranfer/adversarial_hpc_orchestrate.sh analyze \
  adversarial_results/inception_c10_ft_pgd InceptionNet Cifar10 PGD Finetuned
./manuscript/Tranfer/adversarial_hpc_orchestrate.sh plot adversarial_results/inception_c10_ft_pgd
```

## Stage Scripts (temp3/temp4)

Use these two stage scripts after `temp1_run.sh` and `temp2_run.sh` complete.

```bash
chmod +x manuscript/Tranfer/temp3.sh
chmod +x manuscript/Tranfer/temp4.sh

# Stage 3: submit adversarial attack generation jobs
bash manuscript/Tranfer/temp3.sh 100 300

# Stage 4: submit transferability analysis, then plotting
bash manuscript/Tranfer/temp4.sh 100 300
```

Both scripts derive the output folder from epochs and pretrain:
- `adversarial_results_ep<discovery_epochs>_pre<pretrain_epochs>`

Examples:

```bash
# Single-source run (one model/dataset/attack/kind)
bash manuscript/Tranfer/temp3.sh 100 300 InceptionNet Cifar10 PGD Finetuned
bash manuscript/Tranfer/temp4.sh 100 300 InceptionNet Cifar10 PGD Finetuned
```

Arguments for both scripts:

```text
bash temp3.sh <discovery_epochs> <pretrain_epochs> [model] [dataset] [attack] [kind]
bash temp4.sh <discovery_epochs> <pretrain_epochs> [model] [dataset] [attack] [kind]

Optional filters default to ALL when omitted.
```

## Filtering and Selective Execution

```bash
# Analyze specific model
python manuscript/Tranfer/adversarial_analysis.py --mode generate --model VGG16

# Analyze specific dataset
python manuscript/Tranfer/adversarial_analysis.py --mode generate --dataset Cifar10

# Analyze specific attack
python manuscript/Tranfer/adversarial_analysis.py --mode generate --attack PGD

# Analyze specific source kind
python manuscript/Tranfer/adversarial_analysis.py --mode generate --kind Finetuned

# Combine filters
python manuscript/Tranfer/adversarial_analysis.py --mode generate \
  --model VGG16 --dataset Cifar10 --attack FGSM --kind Original
```

## Output Structure

```
adversarial_results/
├── summary.csv                              # Direct attack/susceptibility metrics
├── transferability.csv                      # Cross-model transferability (long-form)
├── attack_success_rates.png                 # Overall attack success rate by attack
├── attack_success_rates_Finetuned.png       # Results for collapsed models
├── attack_success_rates_Original.png        # Results for uncollapsed models
├── attack_comparison_Cifar10.png            # Attack effectiveness comparison
├── attack_comparison_Cifar100.png
├── direct_attack_success_heatmap_Cifar10.png
├── collapsed_vs_original_delta_Cifar10.png
├── transferability_heatmap_Cifar10_PGD.png  # Cross-architecture transferability
├── transferability_heatmap_full_Cifar10_PGD.png
├── transferability_matrix_full_Cifar10_PGD.csv
├── collapsed_vs_original_explainability_summary.csv  # Pruning-vs-original attack + SHAP summary
├── shap_original_vs_collapsed_summary.csv            # SHAP similarity metrics for original vs pruned models
├── shap_class_examples_<dataset>_<model>.png/.svg     # Per-class original vs collapsed SHAP comparison grid
├── shap_class_examples_<dataset>_<model>.csv         # Per-class comparison metrics and class metadata
├── shap_class_examples_<dataset>_<model>.npz         # Saved inputs, SHAP maps, and delta maps
├── transferability_heatmap_Cifar10_FGSM.png
├── InceptionNet_Cifar10_Finetuned_PGD_adv.pt # Single-source attack bundle
└── ...
```

### Saved Adversarial Dataset Bundle

Each generated `*_adv.pt` file now stores:

- `clean_images`
- `adversarial_images`
- `true_labels`
- `source_clean_predictions`
- `source_adversarial_predictions`
- metadata: `source_model`, `dataset`, `kind`, `attack`

### CSV Schema

**summary.csv** - Direct attack success rates:
```
model,dataset,kind,attack,model_label,clean_acc,adv_acc,robust_accuracy,clean_error_rate,adv_error_rate,attack_success_rate,accuracy_drop,relative_accuracy_drop,robustness_ratio
VGG16,Cifar10,Original,PGD,VGG16 (Original),0.9141,0.3456,0.3456,0.0859,0.6544,0.6544,0.5685,0.6219,0.3781
VGG16,Cifar10,Finetuned,PGD,VGG16 (Finetuned),0.9141,0.4123,0.4123,0.0859,0.5877,0.5877,0.5018,0.5489,0.4511
```

**transferability.csv** - Cross-model transferability:
```
source_model,source_kind,source_label,source_attack,target_model,target_kind,target_label,dataset,transfer_acc,transfer_success_rate,source_attack_success_rate,normalized_transfer_rate,same_architecture,same_kind,pair_type
VGG16,Original,VGG16 (Original),PGD,RegNetX_400MF,Original,RegNetX_400MF (Original),Cifar10,0.4532,0.5468,0.6544,0.8357,False,True,cross_arch_same_kind
VGG16,Original,VGG16 (Original),PGD,RegNetX_400MF,Finetuned,RegNetX_400MF (Finetuned),Cifar10,0.5123,0.4877,0.6544,0.7452,False,False,cross_arch_cross_kind
```

## Integration with transfer.py

The adversarial analysis uses the same checkpoint discovery mechanism as `transfer.py`:

```python
# transfer.py already defines:
CHECKPOINT_BASES    # Base directories for checkpoints
CHECKPOINT_FILES    # Filename mappings for finetuned and original models

# adversarial_analysis.py imports these and extends them:
from transfer import CHECKPOINT_BASES, CHECKPOINT_FILES
```

To run adversarial analysis as part of your transfer learning pipeline:

```python
# In transfer.py or a wrapper script:
import subprocess
import sys

def run_adversarial_analysis(output_dir="adversarial_results", mode="full"):
    """Run adversarial robustness evaluation on trained models."""
    cmd = [
        sys.executable,
        "manuscript/Tranfer/adversarial_analysis.py",
        "--mode", mode,
        "--output-dir", output_dir
    ]
    subprocess.run(cmd, check=True)

# Call after transfer learning experiments
if __name__ == "__main__":
    # ... your transfer learning code ...
    
    # Run adversarial analysis
    run_adversarial_analysis()
```

## Supported Attacks

| Attack | Type | Description |
|--------|------|-------------|
| **PGD** | Iterative | Projected Gradient Descent; strong multi-step attack |
| **FGSM** | Single-step | Fast Gradient Sign Method; single-step baseline |
| **IFGSM** | Iterative | Iterative FGSM variant |
| **BIM** | Iterative | Basic Iterative Method |
| **APGD** | Iterative | AutoAttack's PGD variant; state-of-the-art |
| **CW** | Iterative | Carlini-Wagner L2 attack |
| **DeepFool** | Iterative | Minimum-perturbation attack |
| **Square** | Query-based | Query-based black-box attack |
| **AutoAttack** | Ensemble | Strong adaptive ensemble attack |

## Interpretation of Results

### Attack Success Rates (summary.csv)
- **attack_success_rate = 1 - adv_acc**: Direct fooling rate on source model
- **accuracy_drop = clean_acc - adv_acc**: Absolute robustness loss
- **relative_accuracy_drop = (clean_acc - adv_acc) / clean_acc**: Clean-normalized vulnerability
- **robustness_ratio = adv_acc / clean_acc**: Fraction of clean performance retained

Compare Original vs. Finetuned for collapse effects and compare attacks for threat severity.

### Transferability (transferability.csv)
- **transfer_success_rate = 1 - transfer_acc**: Target fooling rate
- **normalized_transfer_rate = transfer_success_rate / source_attack_success_rate**
  - Near 1.0: transferred nearly as well as it attacked the source model
  - Near 0.0: weak transfer
- **pair_type** provides grouped comparisons:
  - `self_same_kind`
  - `same_arch_cross_kind`
  - `cross_arch_same_kind`
  - `cross_arch_cross_kind`

Use `transferability_heatmap_full_*.png` and `transferability_matrix_full_*.csv` for the full source-kind to target-kind matrix (for example, InceptionNet Finetuned source to all Cifar10 targets).

### Explainability comparison (collapsed_vs_original_explainability_summary.csv)
- **mean_shap_cosine_similarity**: Mean cosine similarity between original and collapsed SHAP reference vectors
- **mean_shap_pearson_r / mean_shap_spearman_r**: Linear and rank correlations between explanation profiles
- **mean_shap_topk_jaccard**: Overlap of the most important attribution features after pruning
- **mean_shap_l1_mean_abs_diff / mean_shap_l2_distance**: Attribution shift magnitude between original and collapsed models
- **mean_delta_attack_success_rate**: Attack-success change used in the paper-facing pruning comparison table

### Per-class SHAP examples (shap_class_examples_<dataset>_<model>.*)
- A compact 4-column grid shows the representative input image, original-model SHAP map, collapsed-model SHAP map, and delta map for one example per class.
- The matching CSV stores per-class metadata and the saved `.npz` stores the raw input and attribution arrays used for the figure.

## Performance Tuning

### For Local Machines
```bash
# Reduce batch size if running out of GPU memory
# (would require modifying the script)

# Run one attack at a time
python manuscript/Tranfer/adversarial_analysis.py --mode generate --attack PGD
```

### For HPC Clusters
The HPC orchestration script automatically submits one job per (model, dataset, attack) combination. Typical timing:

- **Generate phase**: ~5-10 minutes per job (varies by attack and dataset size)
- **Analyze phase**: ~2-5 minutes (depends on number of generated attacks)
- **Plot phase**: ~1-2 minutes

### Memory Requirements
- GPU: ~4-8 GB per job (sufficient for typical V100/A100 nodes)
- CPU RAM: ~8 GB
- Disk: ~50 MB per (model, dataset, attack) combination

## Troubleshooting

### torchattacks not found
```bash
pip install torchattacks
```

### Checkpoints not found
Ensure `transfer.py` has been run to populate:
- `../structured_study/pruning_checkpoints/` directory

### Out of GPU memory during attack generation
Some attacks (especially AutoAttack) are memory-intensive. Consider:
- Reducing batch size in the code
- Running attacks sequentially instead of full mode
- Using smaller datasets first (Cifar10 before Cifar100)

### HPC Jobs not submitting
Check that:
- PBS system is available (`which qsub`)
- Correct queue name in `adversarial_hpc_submit.pbs`
- GPU allocation request matches your cluster configuration
- Logs directory is writable

## Advanced: Custom Attack Parameters

Edit `adversarial_analysis.py` to adjust attack parameters:

```python
# In generate_attacks_phase():
epsilon = 0.03      # Perturbation budget (modify per attack type)
steps = 40          # Iterations for iterative attacks
attacks = [...]     # Add/remove attacks

# In instantiate_attack():
# Adjust epsilon, steps, and attack-specific parameters
```

## Citation & References

If you use this analysis, please cite the relevant attack papers:
- PGD/FGSM: Madry et al., 2018
- CW: Carlini & Wagner, 2016
- AutoAttack/APGD: Croce & Hein, 2020
- DeepFool: Moosavi-Dezfooli et al., 2016
