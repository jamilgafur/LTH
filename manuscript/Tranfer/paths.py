# paths.py
"""Centralized output directory structure for all plots, figures, and tables.

Single source of truth for where generated artifacts are written. Training
state (checkpoints, metrics JSON) is intentionally NOT managed here.

Layout:
    outputs/
    ├── discovery/{model}/{dataset}/epochs{E}_pretrain{P}/   # Stage-1 heuristic plots
    ├── diagnostics/{model}_{dataset}_..._epochs{E}_pretrain{P}/  # per-experiment diagnostics
    ├── figures/ep{E}_pre{P}/                               # journal figures
    └── tables/ep{E}_pre{P}/                                # .tex/.csv manuscript tables
"""
from pathlib import Path

OUTPUTS_ROOT = Path("./outputs")
DISCOVERY_ROOT = OUTPUTS_ROOT / "discovery"
DIAGNOSTICS_ROOT = OUTPUTS_ROOT / "diagnostics"
FIGURES_ROOT = OUTPUTS_ROOT / "figures"
TABLES_ROOT = OUTPUTS_ROOT / "tables"


def discovery_run_dir(model, dataset, epochs, pretrain):
    """Stage-1 discovery output for one model/dataset/epoch-budget combo."""
    d = DISCOVERY_ROOT / model / dataset / f"epochs{epochs}_pretrain{pretrain}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def discovery_model_dir(model):
    """Model-level discovery output (e.g. heuristic target summaries)."""
    d = DISCOVERY_ROOT / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_diagnostics_dir(save_path):
    """Per-experiment diagnostic plots/CSVs (mirrors the experiment folder name)."""
    d = DIAGNOSTICS_ROOT / Path(save_path).name
    d.mkdir(parents=True, exist_ok=True)
    return d


def figures_dir(epochs=None, pretrain=None):
    """Journal figures root, optionally scoped to an epoch/pretrain budget."""
    d = FIGURES_ROOT
    if epochs is not None and pretrain is not None:
        d = d / f"ep{epochs}_pre{pretrain}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tables_dir(epochs=None, pretrain=None):
    """Manuscript tables root, optionally scoped to an epoch/pretrain budget."""
    d = TABLES_ROOT
    if epochs is not None and pretrain is not None:
        d = d / f"ep{epochs}_pre{pretrain}"
    d.mkdir(parents=True, exist_ok=True)
    return d
