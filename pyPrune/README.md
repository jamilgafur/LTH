# pyPrune — Library Documentation

[← Back to top-level README](../README.md)

Python package containing the structural collapse engine, model zoo, data loaders, and the
heritage iterative-pruning framework this project grew out of. Installed as a normal package
(`pip install -e .`).

## Table of Contents

- [Package layout](#package-layout)
- [Public API: pyPrune.collapse](#public-api-pyprunecollapse)
- [pyPrune.models](#pyprunemodels)
- [pyPrune.utils](#pypruneutils)
- [Heritage pruning framework](#heritage-pruning-framework)
- [Where to go next](#where-to-go-next)

## Package Layout

| Module | Purpose | Docs |
|---|---|---|
| [`collapse/`](collapse/) | Structural layer collapse + redundancy-driven region discovery (the core) | [collapse/README.md](collapse/README.md) |
| [`models/`](models/) | 9 architectures (VGG16 → ConvNeXt) with dataset presets and `one_batch` probes | [models/README.md](models/README.md) |
| [`pruneMethods/`](pruneMethods/) | Heritage pruner/trainer classes (IMP lineage) | [pruneMethods/README.md](pruneMethods/README.md) |
| [`strategies/`](strategies/) | Heritage saliency strategies (Magnitude, Optimal Brain Damage) | [strategies/README.md](strategies/README.md) |
| `utils.py` | Data loaders (CIFAR-10/100, MNIST, FashionMNIST, STL-10, Caltech101, TinyImageNet, ImageNet), seed/memory helpers, pruneable-module queries | see below |

## Public API: `pyPrune.collapse`

Everything below is re-exported from `pyPrune.collapse` (see its
[README](collapse/README.md) for full internals):

| Symbol | Role |
|---|---|
| `collapse_only` | Collapse `(start, end)` layer spans into surrogate blocks (1×1 conv + adaptive pool), returning the modified model. Supports whole-model or from-checkpoint entry, dry-run, and `init='random'|'lsq'` surrogate initialization. |
| `discover_regions` | Score layers and return collapse-region configs (experiment map). Five scorers: `local_dev` (default), `percentile`, `cv`, `change`, `cka`. |
| `collect_change_stats` | One-pass probe over K real batches measuring **variance + representational change (L2) + CKA** per layer *and* per block via online sufficient statistics. |
| `build_metric_series` | Build the per-layer metric series for `change`/`cka` scorers (block-primary mapping). |
| `calibrate_layer_variances` | K-batch activation-variance calibration returning `(mean, std)` per layer. |
| `get_layer_variances` | Legacy single-input variance probe. |
| `get_dynamic_experiment_config` | Deprecated alias of `discover_regions`. |
| `wrap_pools_safe` / `_wrap_pools_safe` | Wrap pooling layers with underflow-safe wrappers (required for small-input geometry, e.g. InceptionNet on CIFAR). |
| `patch_skip_connections` | Patch residual/custom forwards to tolerate shape drift after collapse (registry-based, picklable). |
| `disable_inplace_relu` | Replace in-place ReLUs with out-of-place ones (hook safety). |
| `SmartIdentity` | Pass-through module replacing excised layers; attribute/index transparent. |
| `_SafePool` | The underflow-safe pooling wrapper (warn-once diagnostics). |
| `get_layer`, `count_trainable_params`, `layer_stats` | Graph utilities. |

## `pyPrune.models`

`from pyPrune.models.Vgg16 import VGG16`-style modules provide nine architectures with
consistent constructor kwargs (dataset-driven `num_classes`, input shape inference, and a
`one_batch` probe tensor used by the pipelines for shape discovery). Inventory and quirks:
[models/README.md](models/README.md).

## `pyPrune.utils`

| Function | Notes |
|---|---|
| `load_cifar10`, `load_cifar100` | batch 512, workers 4, `pin_memory=True`, `persistent_workers`; dataset-standard normalization |
| `load_mnist`, `load_fashionmnist`, `load_stl10`, `load_caltech101`, `load_tiny_imagenet`, `load_imagenet` | smaller defaults; TinyImageNet path is configurable in-source |
| `get_pruneable_modules`, `get_pruneable_named_modules`, `get_pruneable_named_parameters` | introspection helpers used by the heritage pruners |
| `set_seed`, `clean_memory`, `plot_loss_accuracy_sparsity`, `lr_lambda`, `exponential_decay_list` | training utilities |

## Heritage Pruning Framework

[`pruneMethods/`](pruneMethods/README.md) (`BasePruner`, `IterativePruner`, `EarlyStopPruner`,
`BaseTrainer`) plus [`strategies/`](strategies/README.md) (`MagnitudePruningStrategy`,
`OptimalBrainDamageStrategy`) implement the classic pretrain → prune → fine-tune loop with
checkpointing. The dissertation pipelines do **not** use them directly; Transfer2's IMP
baseline re-implements the same algorithm per-experiment for matched-sparsity comparison.
They remain for reference and reproduction of earlier work.

## Where to go next

- Collapse engine internals → [collapse/README.md](collapse/README.md)
- Training on the discovery/collapse pipeline → [manuscript/Tranfer/README.md](../manuscript/Tranfer/README.md)
- Measuring collapsed models → [manuscript/Transfer2/README.md](../manuscript/Transfer2/README.md)
