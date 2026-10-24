# PyPrune Structural Collapse / Transfer Pipeline

Two-stage framework for discovering structurally collapsible layer regions in
pretrained CNNs, then collapsing and fine-tuning them to measure
accuracy-vs-efficiency tradeoffs. Produces journal-quality figures and LaTeX
tables from the results.

## Pipeline Overview

```
temp1_run.sh  ──►  transfer.py --experiment discover        [Stage 1]
                        │  trains Control baseline (skipped if checkpoint exists)
                        │  probes layer variances, discovers collapsible regions
                        │  writes heuristic plots → outputs/discovery/
                        ▼  emits <model>_<dataset>_..._discovered_regions.json

temp2_run.sh  ──►  transfer.py --experiment <name>          [Stage 2]
                        │  parallel collapse + fine-tune jobs (--quant optional)
                        │  checkpoints + metrics per experiment
                        ▼
                 Journal_plot.py (auto-run by submit_job.pbs)
                        └── paper figures + LaTeX tables
```

Stage 2 jobs train independently and in parallel. `submit_job.pbs` appends a
`Journal_plot.py` run after every `transfer.py` invocation, so figures refresh
incrementally as jobs complete.

## Files

| File | Purpose |
|---|---|
| `transfer.py` | Main entry point — Stage 1 discovery + Stage 2 experiment driver |
| `experiments.py` | Training engine: checkpoint/resume logic, per-job metric JSON shards, merging |
| `trainer.py` | Single-epoch training loop |
| `diagnostic.py` | Per-experiment diagnostics (params/FLOPs, activations, memory decomposition, collapse prediction, loss curves) |
| `plots.py` | Collapse-score bar charts and fig/table helper functions |
| `utils.py` | Dataset loading, heuristic target summary plot |
| `Journal_plot.py` | Paper figures (fig1–fig9) + master summary JSON |
| `plot_tradeoff.py` | Composite tradeoff figure + `table_*.tex` LaTeX tables (requires Python ≥ 3.12) |
| `paths.py` | **Single source of truth for all output locations** — do not hardcode paths elsewhere |
| `filemanager.py` | Checkpoint naming, model loading from checkpoint |
| `run_all.sh` | Legacy batch submission (hardcoded experiment lists) |
| `temp1_run.sh` | Stage 1 submission — one discovery job per model × dataset |
| `temp2_run.sh` | Stage 2 submission — one job per discovered experiment × quant flag |
| `submit_job.pbs` | Job template: activates env, runs `transfer.py` then `Journal_plot.py` |

## Usage

### Batch (SGE cluster)

```bash
# Stage 1: discovery — trains Control baseline only if no checkpoint exists
bash temp1_run.sh <discovery_epochs> <pretrain_epochs> [scorer]

# Stage 2: parallel experiments (wait for Stage 1 jobs to finish first)
bash temp2_run.sh <discovery_epochs> <pretrain_epochs> [all|subset]
```

- `scorer`: `local_dev` (default) | `percentile` | `cv` | `change` | `cka` | `all`
- `subset` mode runs **only** `Control_Continuted` and
  `Dynamic_Region_All_Combined` instead of every discovered region.

Edit the `models=(...)` / `datasets=(...)` / `quant=(...)` arrays at the top of
each script to select the sweep. Model choices: `VGG16`, `RegNetX_400MF`,
`InceptionNet`, `XceptionNet`, `MobileNet`, `ConvNeXt`. Datasets: `Cifar10`,
`Cifar100`, `tinyimagenet`, `imagenet`. Pretrained starting checkpoints are
resolved from `../structured_study/pruning_checkpoints/`.

### Single run (manual)

```bash
# Discovery
python transfer.py --model MobileNet --dataset Cifar10 \
    --experiment discover --epochs 100 --pretrain 300

# One experiment from the discovered map
python transfer.py --model MobileNet --dataset Cifar10 \
    --experiment Dynamic_Region_All_Combined --epochs 100 --pretrain 300 [--quant]

# Regenerate paper figures/tables
python Journal_plot.py --epoch 100 --pretrain 300
```

Defaults: `--model RegNetX_400MF --dataset Cifar10 --pretrain 300 --epochs 100`.

## Key Concepts

- **Control baseline** — fully trained uncompressed reference model, produced
  during discovery. All collapse experiments report deltas against it.
- **Discovered regions** — layer ranges flagged as collapsible by the Stage 1
  variance probe; stored in
  `<model>_<dataset>_epochs{E}_pretrain{P}_JF_discovered_regions.json`.
  `temp2_run.sh` requires this file to exist.
- **Scorers** — region-selection heuristics (`local_dev` output variance,
  `percentile`/`cv` calibrated variance, `change` input→output deviation,
  `cka` representational similarity, `all` = per-scorer ablation).
- **Checkpoint semantics** — an experiment with a `final_*.pt` in its
  `checkpoints/` directory is considered complete: training is skipped
  entirely and only metrics are reloaded. Intermediate `*_epoch{N}.pt`
  checkpoints enable exact resume (model/optimizer/scheduler/RNG state).
- **`--quant`** — quantization-aware fine-tuning; runs are suffixed `_quant`
  everywhere (checkpoints, metrics, figures).

## Output Layout

All generated **plots, figures, and tables** live under `outputs/`
(defined in `paths.py`):

```
outputs/
├── discovery/{model}/{dataset}/epochs{E}_pretrain{P}/
│       Stage-1 heuristic plots (Variance/Redundancy/KL/CollapseScore),
│       Layer_Statistics CSVs, layer_variances.json
├── diagnostics/{model}_{dataset}_..._epochs{E}_pretrain{P}/
│       Per-experiment diagnostic SVGs/CSVs and accuracy/loss curves
├── figures/ep{E}_pre{P}/
│       fig1 individual_plots · fig2 methodology · fig3 heuristic_validation
│       fig4 search_space · fig5 hardware_efficiency · fig6 learning_curves
│       fig7 convergence · fig8 accuracy_vs_params · fig9 flops_vs_memory
│       + journal_tradeoff_analysis.png
├── tables/
│       table_hardware_efficiency.tex · table_quantization.tex ·
│       table_control_metrics.tex        (from standalone plot_tradeoff.py)
└── tables/ep{E}_pre{P}/
        master_results_summary.json      (from Journal_plot.py)
```

**Training state lives outside `outputs/`** — this is deliberate:

| Location | Contents |
|---|---|
| `{model}_{dataset}_{ckpt}_epochs{E}_pretrain{P}/` | Per-experiment `checkpoints/` and `metrics/merged_metrics.json` |
| `runs/metrics/` | Per-job metric JSON shards (timestamped, SLURM-safe) |
| CWD root | `*_discovered_regions.json` region maps |

**Cleanup guidance:** to wipe results while keeping trained weights, delete
`outputs/`, `runs/`, and any stale root-level `.tex`/`.png` — but **keep the
per-experiment folders** (they hold the weights). Everything else regenerates
on rerun: discovery plots from Stage 1, figures/tables from `Journal_plot.py`.
Note that diagnostics for already-completed experiments do *not* regenerate
(final-checkpoint skip short-circuits the diagnostics step).

## Requirements

- Python ≥ 3.12 (earlier versions work, but `plot_tradeoff.py` uses 3.12+
  f-string syntax and will be silently skipped by `Journal_plot.py`)
- `torch`, `torchvision`, `fvcore`, `torchinfo`, `pandas`, `seaborn`,
  `matplotlib`, `numpy`
- CUDA-capable GPU (falls back to CPU, impractical for training)
- SGE `qsub` for the batch scripts (queues `all.q` / `UI-GPU`); environment
  expected at `~/env/lth_env/` (see `submit_job.pbs`)
