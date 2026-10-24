# pyPrune.strategies — Heritage Saliency Strategies

[← pyPrune package](../README.md) · [Top-level README](../../README.md)

> **Heritage status.** Saliency strategies used by the [heritage pruners](../pruneMethods/README.md).
> Superseded as a research direction by the activation-based selection in
> [`pyPrune.collapse`](../collapse/README.md); kept for reference and earlier-work reproduction.

## Strategy interface (`PruningStrategy.py`)

Abstract `apply(...)` consuming the model and prunable-module list, producing masks that the
pruner applies to zero the selected weights each step.

## Implementations

| File | Class | Saliency criterion |
|---|---|---|
| `MagnitudePruningStrategy.py` | `MagnitudePruningStrategy` | L1 norm of weights, unrolled across all prunable layers and thresholded globally (`_unroll` flattens per-device). The classic IMP/LTH criterion. |
| `OptimalBrainDamageStrategy.py` | `OptimalBrainDamageStrategy` | LeCun-style second-order saliency: diagonal Hessian estimated over training batches (`compute_diagonal_hessian`), weight saliency `½·H_ii·w_i²` (`compute_saliency`). |

## Contrast with collapse-time selection

Magnitude/OBD saliency scores are *parameter-local* (how big/curved is this weight?). The
collapse engine's criteria are *activation-relational* (does this layer's output differ from its
input across real data samples — L2 deviation / CKA — or vary at all across samples). The latter
is what enables removing **entire regions** and realizing genuine FLOPs/latency savings rather
than masked zeros.

Where to go next: [pruneMethods/](../pruneMethods/README.md) ·
[collapse engine](../collapse/README.md)
