# Adversarial Robustness Analysis Integration

This document explains how to integrate the adversarial analysis pipeline with the transfer learning framework.

## Overview

The adversarial analysis pipeline evaluates the robustness of pruned (collapsed) vs. original (uncollapsed) models using multiple attack methods. It supports:

- **Multiple attacks**: PGD, FGSM, IFGSM, BIM, APGD, CW, DeepFool, AutoAttack, Square
- **Transferability measurement**: How well attacks transfer between different model architectures and pruning states
- **Parallel HPC execution**: Distributed attack generation across multiple GPUs
- **Comprehensive visualizations**: Attack success rates, transferability heatmaps, attack comparisons

## Quick Start

### Option 1: Local Execution (Single Machine)

```bash
# Full pipeline: generate attacks → analyze transferability → plot results
python manuscript/Tranfer/adversarial_analysis.py --mode full

# Generate attacks only (fastest for multiple runs)
python manuscript/Tranfer/adversarial_analysis.py --mode generate

# Analyze transferability (requires pre-generated attacks)
python manuscript/Tranfer/adversarial_analysis.py --mode analyze

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

# Monitor progress
qstat

# Once generation completes, run analysis and plotting
./manuscript/Tranfer/adversarial_hpc_orchestrate.sh analyze adversarial_results
./manuscript/Tranfer/adversarial_hpc_orchestrate.sh plot adversarial_results
```

## Filtering and Selective Execution

```bash
# Analyze specific model
python manuscript/Tranfer/adversarial_analysis.py --mode generate --model VGG16

# Analyze specific dataset
python manuscript/Tranfer/adversarial_analysis.py --mode generate --dataset Cifar10

# Analyze specific attack
python manuscript/Tranfer/adversarial_analysis.py --mode generate --attack PGD

# Combine filters
python manuscript/Tranfer/adversarial_analysis.py --mode generate \
    --model VGG16 --dataset Cifar10 --attack FGSM
```

## Output Structure

```
adversarial_results/
├── summary.csv                              # Direct attack results
├── transferability.csv                      # Cross-model transferability
├── attack_success_rates.png                 # Overall accuracy drop by attack
├── attack_success_rates_Finetuned.png       # Results for collapsed models
├── attack_success_rates_Original.png        # Results for uncollapsed models
├── attack_comparison_Cifar10.png            # Attack effectiveness comparison
├── attack_comparison_Cifar100.png
├── transferability_heatmap_Cifar10_PGD.png  # Cross-architecture transferability
├── transferability_heatmap_Cifar10_FGSM.png
└── ...
```

### CSV Schema

**summary.csv** - Direct attack success rates:
```
model,dataset,kind,attack,clean_acc,adv_acc,accuracy_drop
VGG16,Cifar10,Original,PGD,0.9141,0.3456,0.5685
VGG16,Cifar10,Finetuned,PGD,0.9141,0.4123,0.4918
```

**transferability.csv** - Cross-model transferability:
```
source_model,source_kind,source_attack,target_model,target_kind,dataset,transfer_acc
VGG16,Original,PGD,RegNetX_400MF,Original,Cifar10,0.4532
VGG16,Original,PGD,RegNetX_400MF,Finetuned,Cifar10,0.5123
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
- **accuracy_drop = clean_acc - adv_acc**: Measures vulnerability
  - Larger drop → more vulnerable
  - Compare Original vs. Finetuned to see if pruning affects robustness
  - Compare attacks to see which are most effective

### Transferability (transferability.csv)
- **transfer_acc**: How well attacks from one model transfer to another
  - High transferability → attacks are general/model-agnostic
  - Low transferability → attacks exploit model-specific vulnerabilities
  - Compare Original→Finetuned vs. Finetuned→Original to study collapse effects

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
