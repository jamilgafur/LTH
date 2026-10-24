# pyPrune.pruneMethods — Heritage Pruning Framework

[← pyPrune package](../README.md) · [Top-level README](../../README.md)

> **Heritage status.** These classes implement the original iterative magnitude pruning (IMP)
> framework this repository grew out of (pretrain → prune → fine-tune with checkpointing). The
> current dissertation pipelines (Tranfer/Transfer2) do **not** consume them; Transfer2's IMP
> baseline re-implements the same algorithm per experiment for matched-sparsity comparison.
> Kept for reference and reproduction of earlier work.

## Classes

| File | Class | Role |
|---|---|---|
| `Trainer.py` | `BaseTrainer` | Abstract trainer: owns model, loaders, optimizer, scheduler; standard epoch loop, checkpointing, metrics logging |
| `Pruner.py` | `BasePruner(BaseTrainer, ABC)` | Adds the pruning lifecycle: pretrain `pretrain_epochs`, iterate over `steps` (per-step sparsity), `finetune_epochs` recovery after each step, per-step checkpoint + `pruning_metrics.json` dump under `save_dir` (default `pruning_checkpoints/`) |
| `IterativePruner.py` | `IterativePruner(BasePruner)` | Concrete IMP: each step applies the strategy at the scheduled sparsity, then fine-tunes |
| `EarlyStopPruner.py` | `EarlyStopPruner(BasePruner)` | IMP with early stopping on validation recovery to save compute |

A `PruningStrategy` (see [strategies/](../strategies/README.md)) decides *which* weights are
zeroed each step; the default is `MagnitudePruningStrategy`. Prunable layers default to
`(nn.Conv2d, nn.Linear)` via `pyPrune.utils.get_pruneable_*` introspection helpers.

## Relationship to the collapse work

| | Heritage pruners (this module) | Collapse engine |
|---|---|---|
| Granularity | individual weights (unstructured masks) | whole layers/blocks (structural removal) |
| Compute after compression | unchanged (zeros still multiply) | genuinely reduced (surrogate layer) |
| Selection signal | weight magnitude / Hessian diagonal | activation change / variance / CKA |
| Current consumer | — (archived demos) | Tranfer + Transfer2 pipelines |

Transfer2 still trains an IMP baseline (`torch.nn.utils.prune` global-unstructured L1,
compounded schedule) so this lineage is measured head-to-head against collapse at matched
sparsity — see [Transfer2 docs](../../manuscript/Transfer2/README.md).

Where to go next: [strategies/](../strategies/README.md) ·
[pyPrune overview](../README.md)
