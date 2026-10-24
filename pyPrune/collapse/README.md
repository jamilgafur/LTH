# pyPrune.collapse — Structural Layer Collapse Engine

[← pyPrune package](../README.md) · [Top-level README](../../README.md)

Model-agnostic engine that finds **redundant regions** in a trained CNN and replaces them with a
single learned **surrogate layer**. This is the core contribution of the repository; this document
describes the engine in full — the theory, the algorithms, and every module's internals.

## Table of Contents

- [Concept](#concept)
- [Package layout & data flow](#package-layout--data-flow)
- [Stage A — Probing the network](#stage-a--probing-the-network)
- [Stage B — Region discovery](#stage-b--region-discovery)
- [Stage C — Collapse](#stage-c--collapse)
- [Safety machinery](#safety-machinery)
- [Usage](#usage)
- [Guarantees & invariants](#guarantees--invariants)
- [Limitations](#limitations)
- [Module reference](#module-reference)

## Concept

A trained network contains layers that do little useful computation. This engine was built around a
precise version of that intuition:

> **Hypothesis:** if a layer (or block) barely changes its input — the activation after the layer
> is nearly the activation before it — the transformation is redundant, and the region can be
> replaced by a single layer with minimal accuracy loss after retraining.

The pipeline has three stages, all implemented model-agnostically (no architecture class-name
checks anywhere; everything keys off module attributes, shapes, and name paths):

```
probe  ──►  discover  ──►  collapse
activations   choose regions   replace with surrogate
```

Historically the probe measured only **output activation variance**; the engine also supports a
direct implementation of the hypothesis above — **representational change** (relative L2
deviation of output vs. input) and **CKA(input, output)** as redundancy criteria.

## Package Layout & Data Flow

| File | Role |
|---|---|
| [`core.py`](core.py) | `collapse_only` entry point + the 5-step `_collapse_block` algorithm |
| [`discovery.py`](discovery.py) | All probes (variance / change / CKA) + `discover_regions` scorer/expansion/greedy machinery |
| [`replace.py`](replace.py) | Surrogate block construction, LSQ initialization, graph surgery |
| [`identity.py`](identity.py) | `SmartIdentity` pass-through, `_SafePool` underflow wrapper |
| [`patching.py`](patching.py) | Registry-based forward patching for residual/parallel blocks (picklable) |
| [`graph_utils.py`](graph_utils.py) | Execution-order probing, I/O capture hooks, LCA/name helpers |

Typical data flow per experiment (driven by `manuscript/Tranfer`):

1. **Probe** the trained model with K real batches → per-layer & per-block statistics.
2. **Discover**: a scorer flags "active" (redundant) layers → contiguous runs are expanded under
   structural (LCA) constraints → candidate regions are trial-collapsed and scored by parameter
   and FLOP savings → greedy selection of a non-overlapping region set.
3. **Collapse**: each chosen region `[start..end]` is replaced by `Conv2d(1×1) → AdaptiveAvgPool2d`
   (with normalization/activation surrogates when needed), its weights LSQ- or random-initialized,
   and the graph is patched where residual semantics changed.
4. The collapsed model is fine-tuned downstream (outside this package).

## Stage A — Probing the Network

Three probe functions (all in `discovery.py`) exist for different granularity/cost trade-offs.

### `get_layer_variances(model, dummy_input)` — legacy single-input probe

One forward on a provided tensor (historically `torch.randn`). Hooks on `Conv2d`/`Linear` outputs;
variance computed over spatial dims for 4D maps (with a 1×1-spatial fallback over channels) or
globally otherwise. Fast but single-sample; the Tranfer default-scorer path still uses it.

### `calibrate_layer_variances(model, batches, k=16, device=None)` — calibrated variance

Runs up to `k` real batches; per-layer output variance accumulated with **Welford's online
algorithm** → `{layer: (mean_variance, std_across_batches)}`. The `std` powers the `cv` scorer's
drift gating: a layer whose variance is consistently low across batches is a stronger redundancy
candidate than one whose low mean is a fluke of one batch.

### `collect_change_stats(model, batches, k=16, device=None, max_points=4096, feat_cap=2048)` — combined probe

One pass measuring **three** statistics per unit, where a *unit* is:

- a **layer**: every `nn.Conv2d` / `nn.Linear`, and
- a **block**: every container module (has children, is not the root, contains ≥ 2 parametric
  descendants) — ResNet blocks, Inception blocks, stages… detected purely by containment.

For each unit, forward hooks capture the input `x` and output `y` per batch and accumulate
statistics **online** (no retained activations — memory is flat in `k` and batch size):

| Statistic | Definition | Accumulator |
|---|---|---|
| `var_mean`, `var_std` | Output activation variance (Welford, same formula as `calibrate_layer_variances`) | count, mean, M2 |
| `change` | Relative L2 deviation `E[ ‖y − x̂‖₂ / (‖x‖₂ + ε) ]`. **Same shape:** direct per-sample deviation. **Shape-mismatched** (stride/channel change): *aligned* deviation — ridge-fit the best linear projection `x̂ = xW` via sufficient statistics (`XᵀX`, `XᵀY` in float64 over subsampled spatial-position samples, spatial dims adaptive-pooled to their pairwise min), then report `sqrt(SSE / ‖Y‖²)`: the fraction of the block's behavior the cheapest linear map cannot explain. | scalar sums, or `XᵀX`/`XᵀY` matrices + `‖Y‖²` |
| `cka` | Linear CKA(in, out) on per-sample features (spatial-mean for 4D, identity for 2D): `‖X_cᵀY_c‖²_F / (‖X_cᵀX_c‖_F · ‖Y_cᵀY_c‖_F)`, feature-centered from online first/second moments (`XᵀX`, `YᵀY`, `XᵀY`, `Σx`, `Σy`, `n` in float64). Shape-agnostic. | 2 matrices + 3 vectors + count |

Return value (JSON-serializable):

```python
{name: {"kind": "layer"|"block", "var_mean": float, "var_std": float,
        "change": float | None, "cka": float | None,
        "mismatch": bool,        # True when the aligned (projected) change path was used
        "n_samples": int}}
```

`change`/`cka` are `None` when unmeasurable (hook failure, unsupported tensor rank, degenerate
statistics); `None` values are never flagged by the scorers. CKA matrix accumulation is skipped
for feature dims above `feat_cap` (memory). Determinism: subsampling of aligned-path spatial
points is strided (`[::step]`), not random.

### Why "change" and "variance" are different hypotheses

- The variance criteria answer: *"does this layer's output vary across data samples?"* A layer
  that strongly transforms its input **and** shrinks variance (e.g. aggressive normalization)
  scores as "redundant" under variance even though it is working hard.
- The change/CKA criteria answer the dissertation hypothesis directly: *"did the representation
  move?"* A near-identity residual block whose input has high variance is invisible to variance
  scorers and is precisely what change/CKA catch.

Both claim sets are computable in one capture pass, so the scorers are directly comparable with
data held constant (`--scorer all` in the Tranfer pipeline).

## Stage B — Region Discovery

`discover_regions(model, cnn_layers, variances, input_shape=(1,3,224,224), window_size=3, veto_fraction=0.25, scorer='local_dev', active_fraction=0.25)`

Returns an **experiment config**: a dict of region definitions consumed downstream by
`collapse_only` and the Tranfer job scripts. The `variances` argument is a polymorphic payload:
a flat list of per-layer values (legacy/variance), a `{name: (mean, std)}` calibration mapping
(`cv`), or the `collect_change_stats` dict (`change`, `cka`).

### Scorers (flag extraction, Phase 1)

`active_fraction` (default 25%) controls how many layers are flagged; "Active" = redundant
candidate. **`change` and `cka` are opposite-direction by design and failure to respect that is
the one way to collapse the most-active blocks — this is tested.**

| `scorer` | Payload | Rule | Status |
|---|---|---|---|
| `local_dev` | per-layer variance list | Rolling window-mean (`window_size=3`) of neighbors' variances; flag layer when its variance is **below** its neighborhood (`h = (v−v̄)/v̄ < 0`, clipped to [−1,1]) | **Default** (reproduces legacy behavior bit-for-bit) |
| `percentile` | variance list | flag layers with variance ≤ the `active_fraction` percentile (absolute bottom fraction) | opt-in |
| `cv` | `{name: (mean, std)}` | per-layer coefficient of variation `|std/mean|`; flag the **bottom** fraction (consistently-low variance) | opt-in |
| `change` | `collect_change_stats` dict | per-layer change series (see below); flag the **bottom** fraction — layers that change their input least | hypothesis-driven |
| `cka` | `collect_change_stats` dict | per-layer CKA series; flag the **top** fraction — layers whose output is most similar to their input | hypothesis-driven |

**Block-primary series (`build_metric_series`).** For `change`/`cka`, a layer takes the metric of
its **innermost measured block** when it lives inside one, else its own value — collapsible units
are blocks, so blocks set the score; the per-layer values remain in the stats payload for the
depth-profile figures. Missing/`None` units receive `+inf` (change) or `0.0` (cka) — never
auto-flagged.

### Candidate construction (Phase 2)

1. **Scan & classify.** Each layer gets a status: `Veto` (inside the first `veto_fraction` of
   layers — the stem is protected), `Rejected` (wrong module type / scorer-gated classifier
   names like `classifier`/`fc`/`aux`), `Passive` (non-parametric bridge such as ReLU/BN/pool —
   allowed to *extend* a chain but not start one), or `Active`.
2. **Raw runs.** Contiguous `Active` (+ trailing `Passive`) runs become raw candidates; runs
   with < 2 parametric layers are discarded (a ≥ 2 parametric-layer requirement applies at every
   later stage too).
3. **LCA expansion (structural validity).** For each raw candidate, compute the Lowest Common
   Ancestor of its endpoints' module paths; if the LCA container isn't a collapsible unit
   (`Sequential`/`ModuleList`/root/`*Block`/`*Stage`/… via `is_valid_lca`), expand one layer
   left/right (into Active/Passive only) until it is, or mark the run `Trapped` (dropped).
4. **Trim.** Non-parametric endpoints are shaved so regions start/end on `Conv2d`/`Linear`.

### Candidate evaluation & selection (Phase 3)

Each candidate is **trial-collapsed on a deep copy** (`collapse_only`) and accepted if it reduces
both parameters and FLOPs (fvcore; falls back gracefully when fvcore/base-FLOPs are
unavailable). Failing candidates are **BFS-shaved** (split and retested) to salvage sub-regions.
Accepted regions become `Set_0..N`; a greedy integration pass then unions non-overlapping sets
into `Dynamic_Region_All_Combined` (the headline collapse target).

### Output schema

```json
{
  "Set_0": ["layer.start", "layer.end"],
  "Set_1": ["..."],
  "Dynamic_Region_All_Combined": [["s0","e0"],["s1","e1"]],
  "Status_Map": {"layer": "Active|Passive|Gap|Rejected|Veto|..."},
  "Control_Continuted": null
}
```

`Status_Map` records the per-layer classification for diagnostics/figures;
`Control_Continuted` is the stage-2 control-arm placeholder (`null`, filled by convention).

## Stage C — Collapse

### `collapse_only(...)`

```python
collapse_only(model=None, model_weights_1=None, compression_set=None,
              model_class=None, model_kwargs=None, input_shape=(1,3,32,32),
              device='cpu', safe_param_reduction=True, handle_skips=True,
              debug=True, dry_run=False, init='random') -> nn.Module
```

- Accepts an instantiated model **or** (`model_weights_1` + `model_class` + `model_kwargs`).
- `compression_set`: a label → `(start_layer, end_layer)` mapping or a sequence of such pairs
  (e.g. the `Set_i` entries from discovery).
- `dry_run=True`: perform every step except splicing the surrogate into the parent container
  (used by discovery's candidate trials).
- `init='random'` (default) or `init='lsq'` (ridge least-squares surrogate init).
- `handle_skips`: patch residual/skip-connection forwards afterwards (see `patching.py`).
- `safe_param_reduction`: guard rails that keep the surrogate smaller than the excised block.

Each region goes through `_collapse_block`, a strict 5-step procedure:

**Step 1 — Locate.** Map `start/end` names into their Lowest Common Ancestor container
(`_locate_and_prepare_block`). Cross-branch spans that cannot be mapped raise (never silently
produce a broken graph).

**Step 2 — Capture.** Hook the LCA boundaries and capture the block's incoming activation `x`
and outgoing activation `y_out` (`_capture_preblock_activation` →
`_simulate_input_output_hooks`; currently a single dummy forward — see
[Limitations](#limitations)). **Topology guard:** collapse is defined for 4D NCHW feature maps;
anything else raises a descriptive error instead of corrupting the model.

**Step 3 — Probe downstream.** Find the next `Linear` *in true execution order*
(`execution_order()` — a hooked forward that ranks modules by first-call order, robust to DFS
name-order lies in branched architectures; name-order fallback on probe failure).

**Step 4 — Analyze.** `_analyze_block_output` chooses the feature-map strategy: target spatial
dims, whether a corrective pool is needed before the downstream classifier, which norm/act
surrogates the block needs (content-based, not name-based: `_detect_norm_act` inspects the
captured block's module attributes).

**Step 5 — Replace & fuse.** `_build_and_replace_block` constructs the surrogate and splices it
into the LCA container via `_replace_layers`. Downstream shape compatibility is verified
(`_validate_downstream`); a corrective `_insert_corrective_pool` is added when the classifier's
expected spatial size no longer matches.

### The surrogate block (`replace.py`)

Base anatomy: `Conv2d(1×1, in_c → out_c) → AdaptiveAvgPool2d(target_hw)`. Depending on the
excised block's detected norm/activation signature, the surrogate may interleave
normalization/activation modules — including `_SurrogateLayerNorm2d` for channels-first
(ConvNeXt-style) blocks. The 1×1 keeps the surrogate parametric (trainable projection) while
being orders of magnitude cheaper than the region it replaces.

**Initialization.**
- `init='random'`: standard random conv init (default).
- `init='lsq'`: `_init_surrogate_lsq` solves a **ridge least-squares regression** so that
  `conv(x) ≈ y_out` under the captured activations: `W = (Y Xᵀ)(X Xᵀ + λI)⁻¹` per output group
  (grouped/depthwise convs handled per-group; spatial locations subsampled to `max_points`
  for cost). Empirically ~6× lower output error than random init under the same probe
  statistics. Note: batch-norm recalibration of the *rest* of the model is deliberately
  **not** performed — measured to hurt the fit badly.

## Safety Machinery

- **`identity.py::SmartIdentity`** — a fully transparent pass-through (`forward` returns `x`;
  `__getattr__`/`__getitem__` delegate to the wrapped module's interface where applicable). Used
  as the placeholder for excised excisions and as probe markers.
- **`identity.py::_SafePool`** — wraps `MaxPool2d`/`AvgPool2d`; when the kernel exceeds the
  input's spatial extent (real geometry in stride-heavy nets at CIFAR resolution, e.g.
  InceptionNet's `maxpool4` on 2×2 maps), it adapts via `AdaptiveAvgPool2d` instead of crashing.
  Diagnostics are **warn-once per instance** to keep training logs sane.
- **`patching.py`** — collapsing changes tensor shapes flowing around residual/parallel blocks;
  naive class-name patching was rejected in favor of an **attribute-signature registry**:
  `_SIGNATURES = ((predicate, patch_cls), …)` matches modules by structure (`block`+`shortcut`
  ⇒ `_ResNetForwardPatch`; `dwconv`+`pwconv1/2` ⇒ `_ConvNeXtForwardPatch`; `branch1..4` ⇒
  `_InceptionForwardPatch`; `skip`+`rep` ⇒ `_XceptionForwardPatch`). Non-registry modules with
  custom forwards containing `SmartIdentity` descendants get `_SafeModuleForward`
  (try-original, fall back to identity). All patch classes are top-level and **picklable** —
  required so whole-model artifacts can be saved/loaded with `torch.save`/`torch.load`.
  Patching is idempotent (guarded by `_orig_forward`).
- **`graph_utils.py`** — `execution_order` (hook-based true call order), `_simulate_input_output_hooks`
  (single-forward I/O capture), `get_layer` (dotted-path access), `_update_container`,
  `disable_inplace_relu`, `count_trainable_params`, `layer_stats`.

## Usage

### Discover regions from data (change-based)

```python
from pyPrune.collapse import collect_change_stats, discover_regions

stats = collect_change_stats(model, train_loader, k=16, device="cuda")
layers = [n for n, s in stats.items() if s["kind"] == "layer"]
config = discover_regions(model=model, cnn_layers=layers, variances=stats,
                          input_shape=(1, 3, 32, 32), scorer="change")
print(config["Dynamic_Region_All_Combined"])
```

### Collapse chosen regions

```python
from pyPrune.collapse import collapse_only

collapsed = collapse_only(model=model,
                          compression_set=config["Dynamic_Region_All_Combined"],
                          input_shape=(1, 3, 32, 32), device="cuda", init="lsq")
```

### Variance-based (legacy-compatible) path

```python
from pyPrune.collapse import calibrate_layer_variances, discover_regions

var = calibrate_layer_variances(model, train_loader, k=16)
config = discover_regions(model=model, cnn_layers=list(var.keys()), variances=var,
                          input_shape=(1, 3, 32, 32), scorer="cv")
```

## Guarantees & Invariants

- **Model-agnostic**: no architecture class-name checks; detection is by attributes/shapes.
- **Topology guard**: refuses to produce non-4D-invalid surrogates.
- **Idempotent patching & wrapping**: repeated calls cannot stack patches/wrappers.
- **Picklable artifacts**: collapsed models round-trip through `torch.save`/`torch.load`.
- **Back-compat**: default scorer and legacy probes reproduce legacy discovered regions.
- **Fail loud, not wrong**: cross-branch spans, hook failures, unsupported tensors raise.

## Limitations

- **Surrogate fit probe**: `_capture_preblock_activation` currently captures on a single dummy
  forward, so `init='lsq'` fits Gaussian-noise statistics rather than data statistics. Directions
  to close this gap (data-fitted LSQ via `collect_change_stats`-style batch capture) are known.
- **Trial-collapse cost**: discovery's candidate evaluation deep-copies and collapses the model
  per candidate (accurate but expensive on large nets).
- Block detection is containment-based and 4D-CNN-oriented; transformers/sequence models are out
  of scope by design (topology guard enforces this loudly).

## Module reference

| Symbol | File | One-liner |
|---|---|---|
| `collapse_only` | core.py | multi-region collapse entry point |
| `_collapse_block` | core.py | the 5-step single-region algorithm |
| `discover_regions` | discovery.py | scorer → candidates → greedy region config |
| `collect_change_stats` | discovery.py | one-pass variance+change+CKA probe |
| `calibrate_layer_variances` | discovery.py | K-batch variance (mean, std) probe |
| `get_layer_variances` | discovery.py | legacy single-input variance probe |
| `build_metric_series` | discovery.py | block-primary per-layer metric series |
| `_build_and_replace_block` | replace.py | surrogate construction + graph splice |
| `_init_surrogate_lsq` | replace.py | ridge LSQ surrogate weight init |
| `_SurrogateLayerNorm2d` | replace.py | channels-first LayerNorm surrogate |
| `_validate_downstream` / `_insert_corrective_pool` | replace.py | downstream shape compatibility |
| `SmartIdentity` / `_SafePool` / `wrap_pools_safe` | identity.py | transparency + pool underflow safety |
| `patch_skip_connections` | patching.py | registry-based forward patching |
| `execution_order` / `get_layer` | graph_utils.py | call-order probing + path access |

---

Next: how the training pipeline drives this engine → [manuscript/Tranfer/README.md](../../manuscript/Tranfer/README.md)
