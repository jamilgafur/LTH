# pyPrune.collapse — Structural Layer Collapse: Methodology & Engine Specification

[← pyPrune package](../README.md) · [Top-level README](../../README.md)

Model-agnostic engine that finds **redundant regions** in a trained CNN and replaces each with a
single learned **surrogate layer**. This document is a self-contained specification of the method:
it defines the problem formally, gives every metric and algorithm with its exact mathematics and
pseudocode, and registers every hyperparameter — everything needed to write (and defend) a
dissertation methodology section or to reimplement the method.

## Table of Contents

- [1. Problem Formulation](#1-problem-formulation)
- [2. Notation](#2-notation)
- [3. Activation Probing (Measurement Stage)](#3-activation-probing-measurement-stage)
- [4. Scoring Rules (Redundancy Criteria)](#4-scoring-rules-redundancy-criteria)
- [5. Region Discovery](#5-region-discovery)
- [6. Structural Collapse](#6-structural-collapse)
- [7. Safety Machinery](#7-safety-machinery)
- [8. Hyperparameter Registry](#8-hyperparameter-registry)
- [9. Computational Complexity](#9-computational-complexity)
- [10. Reproducibility & Determinism](#10-reproducibility--determinism)
- [11. Limitations & Design Honesty](#11-limitations--design-honesty)
- [12. Usage](#12-usage)
- [13. Guarantees & Invariants](#13-guarantees--invariants)
- [14. Module Reference](#14-module-reference)

## 1. Problem Formulation

View a trained convolutional network as a composition of $L$ maps
$f = f_L \circ f_{L-1} \circ \cdots \circ f_1$, where the model's *parametric layers*
(`nn.Conv2d`, `nn.Linear`) are indexed $1..L$ in `named_modules` order. Two kinds of **units**
are considered (§3.1):

- **layers** — the individual parametric modules, and
- **blocks** — container modules (e.g. residual blocks, inception branches, stages) identified
  purely by module-tree containment.

A **region** $R = [s, e]$ is a contiguous span of units (in a valid container — §5) replaced by a
single **surrogate block** $\phi_R$: a cheap parametric map replicating the region's interface
(channel and spatial shape of its input/output tensors). The goal:

> Select a set of non-overlapping regions $\{R_j\}$ maximizing parameter and FLOP reduction,
> subject to post-finetune accuracy retention.

**Redundancy hypothesis.** A unit is *redundant* if it barely transforms its input:
$y \approx x$. Three operationalizations of "barely transforms" are supported (variance,
relative L2 change, CKA — §3 and §4), and the dissertation's central empirical question is which
criterion selects regions whose collapse costs the least accuracy.

## 2. Notation

| Symbol | Meaning |
|---|---|
| $x, y$ | input / output activations of a unit over one batch, $N\!\times\!C\!\times\!H\!\times\!W$ (NCHW) or $N\!\times\!D$ |
| $v_i$ | probe statistic of layer $i$ (variance list) |
| $\bar{v}_i$ | rolling neighborhood mean of $v$ around $i$ |
| $h_i$ | signed clipped deviation $(v_i-\bar v_i)/\bar v_i \in [-1,1]$ |
| $P_q(S)$ | $q$-th percentile of series $S$ |
| $X^{\top}X, X^{\top}Y$ | float64 sufficient statistics (aligned change / CKA) |
| $\mathrm{tr}(\cdot)$, $\lVert\cdot\rVert_F$ | trace, Frobenius norm |
| $\lambda$, $\varepsilon$ | ridge strength and numerical epsilon (§8) |
| $W_g$ | LSQ-fitted weight of group $g$ of the surrogate convolution |
| LCA | Lowest Common Ancestor of two dotted module paths (§5.3, §6.1) |

## 3. Activation Probing (Measurement Stage)

All scoring begins with a probe that turns a trained network + data into per-unit statistics.

### 3.1 Measured units

`collect_change_stats(model, batches, k=16, device=None, max_points=4096, feat_cap=2048)`
registers forward hooks over:

1. **Layers**: every module of type `Conv2d`/`Linear` (name from `model.named_modules()`), and
2. **Blocks**: every named container $m$ that (a) is not the root (`name != ""`), (b) is not
   itself parametric, (c) has children, and (d) contains **≥ 2 parametric descendants**
   (`sum(1 for sub in m.modules() if isinstance(sub, (Conv2d, Linear))) >= 2`).

Criterion (d) means blocks are detected by *containment only* — ResNet blocks, Inception
branches, ConvNeXt blocks and stage containers are all found without architecture-specific
code. A per-unit accumulator `_UnitStats` is created for each; hook inputs must be a single
tensor (any hook observation failure permanently marks the unit `update_failed` — measured
statistics become `None`, never fabricated).

### 3.2 Algorithm 1 — One-pass probe with online sufficient statistics

Activations are **never retained**: each batch's contribution is folded into sufficient
statistics, so memory is $O(d^2)$ per unit (feature dims below the caps) and flat in $K$ and
batch size.

```
Input:  model, batch iterable, K (max batches)
For each unit u: hook(u) captures (x = hook input, y = hook output)
for batch b in batches (until K consumed), model.eval(), no_grad:
    forward model(x_b)                      # all hooks fire
    per unit u, from its captured (x, y):
      (i)  VARIANCE  - Welford over batches:
             v_b = layer-variance(y)          # see 3.3
             n <- n+1;  d <- v_b - g
             g <- g + d/n;  M2 <- M2 + d*(v_b - g)
      (ii) CHANGE    - one of two paths (committed on FIRST batch):
             if shape(x) == shape(y):  mode <- 'l2'
                 rel += sum_samples ||y-x||_2 / (||x||_2 + eps)
             else:  mode <- 'aligned'
                 pool x, y to (min H, min W) via adaptive_avg_pool2d
                 flatten spatial positions -> (N*H*W, C), subsample <= max_points
                 XtX += x^T x (float64);  XtY += x^T y (float64);  ||Y||^2 += ||y||^2
      (iii) CKA      - per-sample features f(.) (spatial-mean for 4D):
             if feature dims <= feat_cap:
                 Sx, Sy, Fx^T Fx, Fx^T Fy, Fy^T Fy  (float64), n += N
Output: per-unit {kind, var_mean, var_std, change, cka, mismatch, n_samples}
```

If a unit committed to `'l2'` but a later batch's shapes drift, its path restarts as
`'aligned'` (accumulators reset) — shape drift mid-probe is handled, not crashed.

### 3.3 Per-batch output variance

`_layer_variance(y)`, unbiased=False (population):

- 4D with $H{,}W > 1$: $\mathrm{mean}_{n,c}\ \mathrm{Var}_{h,w}(y)$ (variance over spatial dims,
  averaged over batch × channel);
- 4D with collapsed spatial dims: $\mathrm{mean}_{n}\ \mathrm{Var}_{c}(y)$ if $C>1$, else 0;
- otherwise: global $\mathrm{Var}(y)$.

Over the K batches this yields `var_mean` $= \bar g$ and
`var_std` $= \sqrt{M2/n}$ (population std **across batches** — a drift measure, not a
within-batch std).

### 3.4 Representational change (relative L2 deviation)

**Same-shape path** ($x$ and $y$ identical shape):

$$\text{change} = \frac{1}{n}\sum_{\text{samples } i} \frac{\lVert y_i - x_i \rVert_2}{\lVert x_i \rVert_2 + \varepsilon}, \qquad \varepsilon = 10^{-12}$$

which is 0 for an exact identity map and grows with the transformation applied.

**Aligned path** (shape-mismatched units — e.g. strided or channel-changing blocks). Direct
subtraction is undefined, so the unit's output is compared against the *best linear
re-expression of its input*. With $X^{\top}X \in \mathbb{R}^{C_{in}\times C_{in}}$,
$X^{\top}Y \in \mathbb{R}^{C_{in}\times C_{out}}$ accumulated in float64 over (deterministically
strided) subsampled spatial positions:

$$W = \big(X^{\top}X + \lambda I\big)^{-1} X^{\top}Y, \qquad
\lambda = 10^{-4}\cdot\big(\overline{\mathrm{diag}(X^{\top}X)} + \varepsilon\big)$$

$$\text{SSE} = \max\Big(\lVert Y\rVert^{2} - 2\,\mathrm{tr}(W^{\top}X^{\top}Y) + \mathrm{tr}(W^{\top}X^{\top}X W),\ 0\Big), \qquad
\text{change} = \sqrt{\frac{\text{SSE}}{\lVert Y\rVert^{2} + \varepsilon}}$$

Interpretation: the fraction of the block's input→output behavior that **the cheapest possible
linear map cannot explain**. Units where the best linear projection already reproduces $y$
score ≈ 0 (collapse candidates); units doing genuinely nonlinear work score high.

### 3.5 Linear CKA (in vs. out)

Centered cross-covariance from online moments, with $n$ the total sample count and
$S_{XY} = F_x^{\top}F_y - \tfrac1n \Sigma f_x \Sigma f_y^\top$ (likewise $S_{XX}, S_{YY}$):

$$\text{CKA}(x, y) = \frac{\lVert S_{XY} \rVert_F^{2}}{\lVert S_{XX} \rVert_F \cdot \lVert S_{YY} \rVert_F + \varepsilon}, \qquad n \ge 2$$

CKA ∈ [0,1]; ≈ 1 means output features are a linear re-expression (rotation/scale included) of
input features — again the near-identity signature. Skipped (→ `None`) when either feature
dimension exceeds `feat_cap` (2048) — the $O(d^2)$ matrices would be too large — or $n<2$.

### 3.6 Probe output schema

```python
{name: {"kind": "layer"|"block",
        "var_mean": float, "var_std": float,
        "change": float | None,   # 3.4
        "cka":    float | None,   # 3.5
        "mismatch": bool,         # True iff the aligned path was used
        "n_samples": int}}        # l2: total samples; aligned: spatial points
```

`None` values are **never flagged** by any scorer (§4 sentinels).

### 3.7 Legacy probes (kept for compatibility)

| Probe | What it gives | Used by |
|---|---|---|
| `get_layer_variances(model, dummy_input)` | one forward, per-layer output variance (single sample) | `local_dev`, `percentile` |
| `calibrate_layer_variances(model, batches, k=16)` | K-batch Welford variance → `{name: (mean, std)}` | `cv` (pipeline derives it from the combined probe's variance channels) |

## 4. Scoring Rules (Redundancy Criteria)

`discover_regions(model, cnn_layers, variances, input_shape, window_size=3, veto_fraction=0.25, scorer='local_dev', active_fraction=0.25)`
first converts statistics into a per-layer boolean `Active` flag. The `variances` argument is a
**polymorphic payload**: flat list (variance scorers), `{name: (mean, std)}` (`cv`), or the
`collect_change_stats` dict (`change`, `cka`).

### 4.1 Block-primary metric series (`build_metric_series`)

Collapsible units are blocks, so for `change`/`cka` the per-layer series is built
**block-primary**: a layer's value is the metric of its **innermost measured block**
(longest path-prefix match among block names) when it lies inside one, else its own value.
Missing/`None` units receive the *never-flagged* sentinel: `+∞` for `change` (flagging is a
bottom threshold) and `0.0` for `cka` (top threshold). Per-layer values remain in the stats
payload for depth-profile figures.

### 4.2 The five scorers

$a$ = `active_fraction` (default 0.25). "Flag" = marked `Active` (redundant candidate).

| `scorer` | Payload | Rule (exact) | Direction |
|---|---|---|---|
| `local_dev` *(default)* | flat variance list | $\bar v_i = \mathrm{mean}\{v_j : \max(0,i{-}3) \le j \le \min(N{-}1, i{+}3)\}$ (window of ≤ 7 layers); $h_i = \mathrm{clip}\big(\tfrac{v_i-\bar v_i}{\bar v_i}, -1, 1\big)$; flag iff $h_i < 0$ and not NaN | local relative dip |
| `percentile` | flat variance list | flag $v_i \le P_{100a}(\{v\})$ | bottom $a$ (absolute) |
| `cv` | `{name: (μ, σ)}` | $\text{cv}_i = \lvert\sigma_i/\mu_i\rvert$ if $\mu_i>0$ else $+\infty$; flag $\text{cv}_i \le P_{100a}$. **Raises** `ValueError` if payload lacks std tuples | consistently low variance |
| `change` | stats dict → series §4.1 | flag $\text{change}_i \le P_{100a}$ | **bottom** $a$: least transformative |
| `cka` | stats dict → series §4.1 | flag $\text{cka}_i \ge P_{100(1-a)}$ | **top** $a$: most input-similar |

The opposing directions of `change` (flag small) and `cka` (flag large) are **by design** —
both encode near-identity behavior; inverting either flags the *most*-active units (this is
tested).

### 4.3 Hypothesis content of the criteria

- *Variance* criteria ask: "does this layer's output vary across data samples?" A layer that
  strongly transforms its input **and** shrinks variance scores as redundant under variance
  even while doing real work.
- *Change/CKA* criteria ask: "did the representation move?" A near-identity residual block
  whose input has high variance is invisible to variance scorers and is exactly what
  change/CKA catch.

Because `collect_change_stats` computes all channels in one pass, the criteria are directly
comparable with data held constant (`--scorer all` in the Tranfer pipeline runs three scorers
on one probe).

## 5. Region Discovery

Discovery proceeds in four phases: flag (§4) → raw runs → structural expansion → evaluated
selection.

### 5.1 Algorithm 2 — Scan & status classification

Each of the $N$ parametric-and-bridge layers receives a status:

```
veto_idx = int(N * veto_fraction)          # first 25% of layers protected (stem)
for i, layer in enumerate(cnn_layers):
    is_classifier  = any(t in name.lower() for t in {'classifier','fc','aux'})
    is_parametric  = isinstance(mod, (Conv2d, Linear))
    is_valid_target = isinstance(mod, (Conv2d, Linear, ReLU, BatchNorm2d))
                      and not is_classifier
    if   i < veto_idx:        status = Veto       # protected, can never enter a region
    elif not is_valid_target: status = Rejected   # hard barrier
    elif not active_flags[i]: status = Passive if not is_parametric else Rejected
    else:                     status = Active
```

- **Passive** (non-parametric, inactive: ReLU / BN / pool) layers are *bridges*: they may
  extend an open run but can never start one. This keeps e.g. `conv → bn → relu → conv` chains
  from being fractured by scorer hiccups on the non-parametric members.
- Runs are finalized at `Veto`/`Rejected` layers and accepted as **raw candidates** iff they
  contain **≥ 2 parametric layers**; shorter fragments are marked `Gap` (diagnostics only).

### 5.2 Why an LCA is needed

A region must be collapsible: its endpoints must live in a container whose forward executes
them as a sliceable sequence. A span whose endpoints diverge across parallel branches (e.g.
two Inception paths) has no sequential replacement. The Lowest Common Ancestor of the
endpoints' module paths decides this.

### 5.3 Algorithm 3 — LCA computation & structural expansion

**LCA of two dotted paths** (`get_lca_path`; identical logic at collapse time, §6.1):

```
get_lca_path(a, b):
    if a == b:  return parent(a)              # drop last component
    walk split(a, '.'), split(b, '.') zipped, keeping the common prefix
    return join(common_prefix)                # may be "" = root model
```

**Validity** (`is_valid_lca`) — the LCA module is collapsible iff any of:
`None` · `nn.Sequential`/`nn.ModuleList` · the root-model class (allows region spans to bridge
stages) · class name containing any of `{'Inception','Block','Stage','Layer','Residual','AnyNetX'}`
· or a leaf (no children). Everything else with children is invalid. *(Containment/attribute
rules first; the name-substring whitelist is a documented legacy heuristic — see §11.)*

**Expansion loop** — while the current LCA is invalid or the span has < 2 parametric layers:

1. extend **right** one layer if the next layer is `Active`/`Passive`, else
2. extend **left** one layer if the previous layer is `Active`/`Passive` and index ≥ `veto_idx`, else
3. neither possible → the candidate is **Trapped** → its layers are marked `Gap` and dropped.

Expansion only ever *grows* a region into scorer-approved or bridge layers, so the result is
always a valid subsequence of one container with ≥ 2 parametric layers.

### 5.4 Algorithm 4 — Trial-collapse evaluation with BFS shaving

```
base_params, base_flops <- measure(model)        # fvcore; flops check disabled if unavailable
queue <- R_candidates ; tested <- {} ; i <- 0
while queue:
    r <- queue.popleft()
    trim non-parametric layers from both ends of r      # regions start/end on Conv2d|Linear
    if parametric_count(r) < 2 or (r[0], r[-1]) in tested: continue
    tested.add((r[0], r[-1]))
    m <- deepcopy(model)
    try:
        m <- collapse_only(m, {test: (r[0], r[-1])}, input_shape, dry_run=False)
        m(dummy_input)                                  # forward must succeed
        new_params <- count_trainable_params(m)
        new_flops  <- FlopCountAnalysis(m, dummy_input) # or +inf on failure
        if new_params < base_params and (flop check off or new_flops < base_flops):
            accept as Set_i ;  i += 1                   # STRICT: both must decrease
        else:
            if |r| > 2:                                 # BFS shave: salvage sub-regions
                enqueue r[1:]  if parametric_count >= 2
                enqueue r[:-1] if parametric_count >= 2
    except Exception: log and continue                  # candidate crashes are non-fatal
    finally: release test model, gc, CUDA cache
```

Each accepted region is therefore *empirically certified*: a full collapse + forward proves
the graph still executes, and strict decrease proves the surrogate is cheaper in params and
(when measurable) FLOPs. Failed shapes are bisected (`r[1:]`, `r[:-1]`) so still-viable inner
sub-regions are salvaged. Test cost dominates discovery (§9).

### 5.5 Phase 4 — Integration

The headline artifact `Dynamic_Region_All_Combined` is the list of **all** accepted Sets
(forced verbatim when ≥ 2 Sets exist). An older combinatorial variant — re-collapsing
combinations to hunt for diminishing returns — exists commented out in the source; the active
semantics is "union of every certified region". (Earlier documentation describing a "greedy
union of non-overlapping sets" referred to the retired variant.)

### 5.6 Output schema

```json
{
  "Set_0": ["layer.start", "layer.end"],
  "Set_1": ["..."],
  "Dynamic_Region_All_Combined": [["s0","e0"],["s1","e1"]],
  "Status_Map": {"layer": "Active|Passive|Gap|Rejected|Veto|Rejected (Orphan)"},
  "Control_Continuted": null
}
```

`Set_*` = single certified spans; `Dynamic_Region_All_Combined` = the full union consumed by
stage 2; `Status_Map` = per-layer classification for diagnostics/figures;
`Control_Continuted` = stage-2 control-arm placeholder (convention, always `null` here).

## 6. Structural Collapse

`collapse_only(model=None, model_weights_1=None, compression_set=None, model_class=None, model_kwargs=None, input_shape=(1,3,32,32), device='cpu', safe_param_reduction=True, handle_skips=True, debug=True, dry_run=False, init='random')`

- Entry from an instantiated model **or** from (`model_weights_1` + `model_class` + `model_kwargs`).
- `compression_set`: `label → (start, end)` mapping, or a sequence of such pairs.
- `dry_run=True`: every step except the final splice (discovery's trials use `dry_run=False`,
  since acceptance requires an executable model).
- `init`: `'random'` (default) or `'lsq'` (§6.4).
- `handle_skips`: patch residual/parallel forwards afterwards (§7.3).
- `safe_param_reduction`: guard rails keeping the surrogate smaller than the excised block.

### 6.1 Algorithm 5 — `_collapse_block(start, end)` in five steps

**Step 1 — Locate (LCA resolution).** `_locate_and_prepare_block` computes the dotted-path LCA
(as in §5.3), then maps `start`/`end` to child indices of the LCA container by prefix matching
(`child_full_name == start` or `start.startswith(child_full_name + '.')`; swap if inverted).
Three adjustments to naive resolution:

- **Already collapsed**: if the container is a `SmartIdentity` (absorbed by an earlier region),
  the region is skipped (locator returns `None` → caller raises a `ValueError` naming
  cross-branch collapse as the likely cause).
- **Holistic escalation**: containers whose class is one of
  `{InceptionBlock, ConvNeXtBlock, XceptionBlock, DepthwiseSeparableConv}` — or, after index
  mapping, any container not in `{Sequential, SeparableConv2d, ModuleList, <root class>}` — are
  **escalated to their parent**: the whole block is replaced as one unit
  (`start_idx == end_idx` in the parent) rather than sliced internally.
- **Cross-branch rejection**: endpoints that cannot be mapped to children of one container
  return `None` → hard failure at collapse level (fail loud, never corrupt).

**Step 2 — Capture.** Forward hooks are placed on the true excision boundaries — the container
itself for non-sequential containers, else `{container}.{first}` and `{container}.{last}` —
and `_capture_preblock_activation` runs one forward on a dummy input, capturing the region's
incoming tensor $x$ and outgoing tensor $y_{out}$ (and the pre-collapse parameter count).
**Topology guard:** collapse is defined only for 4D NCHW feature maps; anything else raises a
descriptive `ValueError` instead of producing corrupted geometry.

**Step 3 — Downstream probe.** The next `Linear` is located in **true execution order**
(`execution_order()`: a hooked dummy forward ranking modules by first-call order), because DFS
`named_modules` order lies in branched architectures; name-order is the fallback if the probe
fails.

**Step 4 — Analysis.** `_analyze_block_output` records, from the *actual captured tensors*
(never from layer attributes): output shape/channels `(C_out, H_out, W_out)`; whether the block
contained pooling (informational); any shortcut conv's output channels (informational); and the
next linear's `in_features` heuristic.

**Step 5 — Surrogate synthesis & fusion** (§6.2–§6.4).

### 6.2 The surrogate block

`_build_and_replace_block` assembles `nn.Sequential(OrderedDict(...))`:

```
[conv] → [norm] → [act] → [adaptive_pool]
```

with every component chosen by inspecting the excised block's content:

**Convolution routing (geometry-inherited).** Let `c0` be the block's first `Conv2d`:

| Condition | Surrogate conv |
|---|---|
| `c0.groups == c0.in == c0.out` **and** captured `C_in == C_out` | **depthwise-grouped** `Conv2d(C, C, k, groups=C)` with `k`, stride, padding **inherited** from `c0` (name `conv_g1x1`) |
| otherwise | **pointwise** `Conv2d(C_in → C_out, 1×1)` (stride inherited, padding 0, name `conv_1x1`) |

All surrogates are `bias=False`; stride/padding inheritance prevents silent spatial-geometry
changes leaking into downstream layers.

**Normalization / activation inheritance** (`_detect_norm_act` takes the **last** norm and
activation found anywhere in the excised block — content-based, not name-based):

| Detected norm | Surrogate norm |
|---|---|
| None / `BatchNorm2d` | `BatchNorm2d(C_out)` |
| `LayerNorm` | `_SurrogateLayerNorm2d(C_out)` (channels-first LayerNorm — ConvNeXt style) |
| `GroupNorm(g)` | `GroupNorm(gcd(g, C_out), C_out)` |
| other | `InstanceNorm2d(C_out)` |

| Detected act | Surrogate act |
|---|---|
| None / `ReLU` | `ReLU(inplace=False)` |
| other class `T` | `T()` (falls back to `Identity` if uninstantiable) |

**Adaptive pool**: `AdaptiveAvgPool2d((H_out, W_out))` guarantees the surrogate's output spatial
size equals the excised block's regardless of conv geometry.

**Fusion**: `_replace_layers` splices the surrogate into the LCA container at
`[start_idx, end_idx]`; `_update_container` writes it back; the parameter delta is logged
(warning if positive — discovery already enforced strict decrease, §5.4). The exact
replacement slot is then validated (fallback: reverse search for surrogate-signature children:
`collapsed_*` names or `conv_1x1`/`conv_g1x1`/`conv_dw`/`adaptive_pool` keys).

**Downstream compatibility.** `_validate_downstream` inspects modules after the splice and
`_insert_corrective_pool` inserts a corrective pool before the downstream classifier when the
flattened `C×H×W` no longer matches its `in_features` — keeping the head intact without edits.

### 6.3 Function-preserving initialization (`init='lsq'`)

`_init_surrogate_lsq` fits **only the surrogate convolution** so that
$\text{conv}(x) \approx y_{out}$ under the captured block I/O (normalization is left at its
near-identity default so the fit is preserved). Per sample, `F.unfold` extracts input patches
$X_g \in \mathbb{R}^{(C_{in}/g)\cdot k^2 \times L}$ per group $g$, with matching output columns
$Y_g$; spatial locations are subsampled to ≤ 65536. Each group solves the regularized normal
equations:

$$A = X_g X_g^{\top}, \qquad A \leftarrow A + 10^{-4}\cdot\max\!\big(\overline{\mathrm{diag}(A)},\,10^{-8}\big)\,I, \qquad
W_g = \big(Y_g X_g^{\top}\big)\,A^{-1}$$

reshaped into `conv.weight`. Empirically ≈ 6× lower surrogate output error than random init on
the same probe statistics. Deliberately **not** performed: batch-norm recalibration of the rest
of the network (measured to hurt the fit badly). Shape mismatches between patch and output
grids raise a `RuntimeError` (fail loud).

### 6.4 What collapse guarantees (and does not)

Guaranteed by construction: the surrogate's output tensor shape equals the excised block's
(`adaptive_pool`), the graph remains executable (certified per candidate in §5.4), the
parameter count strictly decreases (acceptance criterion), and whole-model artifacts stay
picklable (§7.3). Not guaranteed: accuracy preservation before fine-tuning — that is what the
fine-tune stage and the LSQ init are for.

## 7. Safety Machinery

### 7.1 `SmartIdentity` (`identity.py`)

Transparent pass-through replacing excised modules: `forward(x) → x`; attribute and index
access absorb topology chains (`identity.submodule(x) ≡ identity(x)`) while explicitly
forbidding shape-affecting attributes (`in_channels`, `out_channels`, `in_features`,
`out_features`, `groups`, `shape`, `size`, `dim`, `ndim`) so misuse fails with
`AttributeError` instead of silently propagating wrong geometry.

### 7.2 `_SafePool` + `wrap_pools_safe` (`identity.py`)

Pooling underflow is real at CIFAR resolution (e.g. InceptionNet's 3×3 `maxpool4` meeting a
2×2 map after stride chains). `_SafePool` wraps `MaxPool2d`/`AvgPool2d` and, in `forward`:

1. kernel $(k_h, k_w)$ exceeds input $(H, W)$ (or degenerate input) →
   `adaptive_avg_pool2d(x, (max(1, min(H,k_h)), max(1, min(W,k_w))))`;
2. pool output collapses below $1\times1$ → return the input unchanged;
3. any runtime failure → bypass the pool, return input.

Diagnostics are **warn-once per wrapped instance** (training loops hit them every batch).
`wrap_pools_safe` applies this recursively; wrapping is idempotent.

### 7.3 `patching.py` — signature-registry forward patching

Collapsing changes tensor shapes flowing around residual/parallel blocks; their custom
forwards (`out += shortcut(x)`, `torch.cat(branches)`, …) must tolerate that. Instead of
class-name hardcoding, modules are matched by **attribute-signature predicates** in the
registry `_SIGNATURES`:

| Predicate (attributes present) | Patch |
|---|---|
| `block` + `shortcut` (ResNet-family) | `_ResNetForwardPatch` |
| `dwconv` + `pwconv1`/`pwconv2` (ConvNeXt) | `_ConvNeXtForwardPatch` |
| `branch1`..`branch4` (Inception) | `_InceptionForwardPatch` |
| `skip` + `rep` (Xception) | `_XceptionForwardPatch` |
| unmatched module with custom forward containing `SmartIdentity` descendants | `_SafeModuleForward` (try original, fall back to identity-safe path) |

All patch classes are top-level and **picklable** — required for whole-model
`torch.save`/`torch.load` artifacts — and patching is idempotent (guarded by `_orig_forward`).

## 8. Hyperparameter Registry

Every tunable, its default, and its definition site.

| Parameter | Default | Site | Meaning |
|---|---|---|---|
| `k` | **16** (pipeline) | `collect_change_stats` / `calibrate_layer_variances`; set by transfer.py | probe batches consumed |
| `max_points` | **4096** | `collect_change_stats` | aligned-path spatial rows per batch (strided subsample) |
| `feat_cap` | **2048** | `collect_change_stats` | CKA skipped above this feature dim |
| `window_size` | **3** | `discover_regions` | `local_dev` rolling half-width (window ≤ 2w+1 = 7 layers) |
| `veto_fraction` | **0.25** | `discover_regions` | first fraction of layers protected from regions |
| `active_fraction` | **0.25** | `discover_regions` | flagged fraction for threshold scorers |
| min parametric | **2** | discovery (scan, expansion, eval, shave) | minimum Conv2d/Linear per region |
| acceptance | strict ↓ | discovery evaluation | params strictly decrease **and** FLOPs strictly decrease (when measurable) |
| ridge $\lambda$ | **1e-4** × mean-diag | `_change_from_stats`, `_init_surrogate_lsq` | scale-free ridge for aligned projection / LSQ |
| LSQ `max_points` | **65536** | `_init_surrogate_lsq` | spatial locations per sample in the LSQ fit |
| $\varepsilon$ | **1e-12** | discovery internals | numerical stabilizer (L2 norms, CKA, SSE) |
| `input_shape` | pipeline `one_batch` shape | `collapse_only` / `discover_regions` args | dummy forward geometry |

## 9. Computational Complexity

- **Probe** (Algorithm 1): $K$ model-forwards plus $O(d^2)$ float64 memory per unit for the
  aligned-change and CKA matrices (feature dims ≤ `feat_cap`), **flat in $K$ and batch size**.
  Subsampling caps the aligned path at $O(\texttt{max\_points}\cdot(C_{in}^2 + C_{in}C_{out}))$
  accumulation per batch per unit.
- **Discovery** (§5): one base measurement (`params` + one fvcore pass), then per candidate a
  **full deepcopy + collapse + forward + fvcore analysis** — the dominant cost, linear in
  accepted+rejected candidates, with BFS shaving adding ≤ 2 children per rejection and
  `tested_regions` preventing re-evaluation. Region count is bounded by the layer count, so the
  worst case is quadratic in layer count (each candidate may shave down to singleton spans).
- **Collapse** (Algorithm 5): two dummy forwards (capture + execution-order probe) per region
  plus the splice — negligible versus training.

## 10. Reproducibility & Determinism

| Operation | Determinism |
|---|---|
| Aligned-change subsampling | **Deterministic** — fixed stride `[::step]`, no RNG |
| CKA / variance accumulation | Deterministic (pure algebra on captured tensors) |
| `local_dev` / threshold scorers | Deterministic given the variance list |
| LSQ subsampling | **Seed-dependent** — `torch.randperm` without a local generator; seed via `set_seed` for bit-exact surrogates |
| Hook firing order | Deterministic — follows `named_modules` DFS order per forward |
| Discovery candidate order | Deterministic — scan order + FIFO queue |
| Artifact picklability | Whole models round-trip through `torch.save`/`torch.load` (`weights_only=False`); all patch/surrogate classes are top-level |

Probe statistics are cached universe-wide (`runs/plots/.../change_stats.json`) so re-running
discovery for a different scorer on the same universe reuses identical measurements.

## 11. Limitations & Design Honesty

1. **LSQ fits Gaussian statistics.** `_capture_preblock_activation` captures on a single
   *dummy* forward, so `init='lsq'` fits the surrogate to noise distribution, not data.
   Closing this (data-driven capture using the §3 machinery) is known future work.
2. **Phase-4 union is forced, not greedy.** `Dynamic_Region_All_Combined` takes all certified
   Sets verbatim (§5.5); interaction effects between regions are not validated combinatorially
   (retired variant documented in source).
3. **Two name-based heuristics** exist despite the model-agnostic design goal: the
   `is_valid_lca` class-name whitelist (§5.3) and the escalation class list (§6.1). Both are
   fail-open (unknown classes escalate to parent or reject loudly), work on every architecture
   tested, and are listed here explicitly.
4. **4D convolutions only.** The topology guard (§6.1) refuses non-NCHW blocks; sequence
   models are out of scope by design.
5. **fvcore is optional.** When unavailable (or the base count fails), the FLOPs acceptance
   criterion is silently disabled — regions are then certified on executable-forward + params
   only.
6. **Containment-based block detection** (§3.1) can miss exotic in-place residual additions
   that do not appear as containers; patching (§7.3) is the compensating mechanism.

## 12. Usage

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

## 13. Guarantees & Invariants

- **Fail loud, not wrong**: cross-branch spans, hook failures, unsupported tensor ranks, and
  LSQ grid mismatches raise — they never silently corrupt the graph.
- **Certified regions only**: every discovered region survived a real collapse + forward and
  strictly reduced parameters (and FLOPs when measurable).
- **Idempotent patching & wrapping**: repeated `patch_skip_connections` / `wrap_pools_safe`
  calls cannot stack.
- **Picklable artifacts**: collapsed models round-trip whole through `torch.save`/`torch.load`.
- **Back-compat**: the default scorer (`local_dev`) and legacy probes reproduce legacy
  discovered regions; scorer directionality for `change`/`cka` is tested.
- **No fabricated statistics**: unmeasurable units report `None` and are never flagged.

## 14. Module Reference

| Symbol | File | Role |
|---|---|---|
| `collapse_only` | core.py | multi-region collapse entry point |
| `_collapse_block` | core.py | Algorithm 5: the 5-step single-region procedure |
| `discover_regions` | discovery.py | scorers → candidates → certified region config (§4–§5) |
| `collect_change_stats` | discovery.py | Algorithm 1: one-pass variance + change + CKA probe (§3) |
| `build_metric_series` | discovery.py | block-primary per-layer metric series (§4.1) |
| `calibrate_layer_variances` | discovery.py | K-batch Welford variance probe (§3.7) |
| `get_layer_variances` | discovery.py | legacy single-input variance probe (§3.7) |
| `get_dynamic_experiment_config` | discovery.py | deprecated alias of `discover_regions` |
| `_locate_and_prepare_block` | replace.py | collapse-time LCA: prefix algorithm + escalation (§6.1) |
| `_build_and_replace_block` | replace.py | surrogate assembly + graph splice (§6.2) |
| `_init_surrogate_lsq` | replace.py | ridge LSQ surrogate init (§6.3) |
| `_SurrogateLayerNorm2d` | replace.py | channels-first LayerNorm surrogate |
| `_detect_norm_act` | replace.py | content-based norm/act inheritance |
| `_validate_downstream` / `_insert_corrective_pool` | replace.py | classifier shape compatibility (§6.2) |
| `SmartIdentity` / `_SafePool` / `wrap_pools_safe` | identity.py | transparency + pool underflow safety (§7.1–§7.2) |
| `patch_skip_connections` | patching.py | signature-registry forward patching (§7.3) |
| `execution_order` / `get_layer` / `_update_container` | graph_utils.py | true call-order probing + path access |
| `disable_inplace_relu` | graph_utils.py | in-place ReLU → out-of-place (hook safety) |
| `count_trainable_params` / `layer_stats` | graph_utils.py | measurement utilities |

---

Next: how the training pipeline drives this engine → [manuscript/Tranfer/README.md](../../manuscript/Tranfer/README.md)
