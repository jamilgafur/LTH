# manuscript/Tranfer — Training Pipeline (Stage 1 + Stage 2)

[← Top-level README](../../README.md) · [pyPrune](../../pyPrune/README.md) ·
[Analysis: Transfer2](../Transfer2/README.md)

End-to-end pipeline that trains baselines, discovers redundant regions, collapses them, and
fine-tunes the compressed models — driven by [`pyPrune.collapse`](../../pyPrune/collapse/README.md)
and built for PBS/HPC execution.

## Table of Contents

- [Pipeline overview](#pipeline-overview)
- [CLI reference (transfer.py)](#cli-reference-transferpy)
- [Scorers](#scorers)
- [Universe & artifact conventions](#universe--artifact-conventions)
- [HPC scripts](#hpc-scripts)
- [Artifact inheritance](#artifact-inheritance)
- [Regions JSON schema](#regions-json-schema)
- [Where to go next](#where-to-go-next)

## Pipeline Overview

**Stage 1 — Discovery (`--experiment discover`)**, `run_discovery_stage`:

1. **Baseline**: inherit or train `final_JF_Control(_quant).pt` for `pretrain` epochs in the
   universe directory.
2. **Probe**:
   - default scorers (`local_dev`/`percentile`/`cv`) → single-input variance probe
     (`get_layer_variances`, cached as `runs/plots/.../layer_variances.json`);
   - `change`/`cka`/`all` → one combined real-data pass
     ([`collect_change_stats`](../../pyPrune/collapse/README.md#stage-a--probing-the-network),
     K=16 batches, cached as `runs/plots/.../change_stats.json` — scorer-independent, shared).
3. **Discover regions** per scorer and write the regions JSON (schema below). `--scorer all`
   runs `local_dev` + `change` + `cka` **from the same capture pass** and additionally writes a
   combined `*_scorer_ablation.json` (all three region maps + stats payload).
4. Emit diagnostics: heuristic analysis, and for change-based probes a depth-profile figure
   (`change_profile_<scorer>.{svg,png,csv}`).

**Stage 2 — HPC (`--experiment <name>`)**, `run_hpc_stage`: reads the universe's regions JSON,
re-inherits late-arriving donor artifacts, and runs `run_jf_experiment` for the requested
experiment key (`Control_Continuted`, `Set_i`, `Dynamic_Region_All_Combined`, …): collapse (for
region experiments) + `epochs` fine-tuning + metrics/figures + whole-model persistence.

## CLI Reference (`transfer.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `RegNetX_400MF` | `VGG16`, `RegNetX_400MF`, `InceptionNet`, `XceptionNet`, `MobileNet`, `ConvNeXt` |
| `--dataset` | `Cifar10` | Dataset name (e.g. `Cifar10`, `Cifar100`, `tinyimagenet`) |
| `--pretrain` | `300` | **Stage 1** baseline/Control epochs |
| `--epochs` | `100` | **Stage 2** post-collapse fine-tuning epochs |
| `--experiment` | `discover` | `discover` (Stage 1) or an experiment key from the regions JSON (Stage 2) |
| `--quant` | off | Quantization-aware training variants (`*_quant` checkpoint names) |
| `--scorer` | `local_dev` | `local_dev`, `percentile`, `cv`, `change`, `cka`, or `all` (discovery-only ablation mode) |

## Scorers

| Scorer | Criterion (flag rule) | Probe |
|---|---|---|
| `local_dev` (default) | output variance below rolling neighborhood mean | single-input variance |
| `percentile` | bottom 25% absolute variance | single-input variance |
| `cv` | bottom 25% variance drift (K-batch calibrated `(mean, std)`) | combined probe (variance channel) |
| `change` | bottom 25% representational change `‖out−in‖/‖in‖` | combined probe |
| `cka` | top 25% CKA(in, out) | combined probe |
| `all` | runs `local_dev`+`change`+`cka` on one probe; **discovery-only**, writes three namespaced region maps + `*_scorer_ablation.json` | combined probe |

## Universe & Artifact Conventions

- Universe dir: `{model}_{dataset}_{tag}_epochs{E}_pretrain{P}` — non-default scorers get a
  `_scorer{S}` infix after the tag. The default scorer keeps every legacy name byte-identical.
- Regions JSON: `{model}_{dataset}_epochs{E}_pretrain{P}[_scorer{S}]_JF_discovered_regions.json`.
- Checkpoints inside a universe: `final_{workflow}_{exp}.pt`,
  `structure_{workflow}_{exp}.pt` (collapsed architecture, pre-training),
  `final_{workflow}_{exp}_whole.pt` (trained collapsed model, whole-model pickle consumed
  directly by Transfer2).
- Probe caches in `runs/plots/{model}/{dataset}/epochs{E}_pretrain{P}/`:
  `layer_variances.json` (legacy) and `change_stats.json` (combined, shared across
  change/cka/all runs).

## HPC Scripts

| Script | Purpose | Usage |
|---|---|---|
| `temp1_run.sh` | Submit Stage-1 discovery jobs over model×dataset×quant loops | `./temp1_run.sh <epochs> <pretrain> [scorer]` — e.g. `./temp1_run.sh 300 100 all` |
| `temp2_run.sh` | Submit Stage-2 jobs per experiment read from the regions JSON | `./temp2_run.sh <epochs> <pretrain> [mode] [scorer]` — `mode`: `all` or `subset` (only `Control_Continuted` + `Dynamic_Region_All_Combined`); scorer `all` is rejected by design (discovery-only) |
| `submit_job.pbs` | PBS payload: env setup → `transfer.py … --scorer "$SCORER"` → `Journal_plot.py --epoch … --pretrain … --scorer …`. `SCORER` defaults to `local_dev`; log file names get a scorer tag for non-default scorers | submitted via `qsub -v …` by the wrappers |

## Artifact Inheritance

Scorer-independent artifacts are **copied, not retrained**, into scorer universes:

- `transfer.py::inherit_scorer_independent_artifacts` copies `final_JF_Control(_quant).pt` and
  `final_JF_Control_Continuted(_quant).pt` from the default universe (same epochs/pretrain),
  consulted at the start of **both** stages (Stage 2 re-checks to catch late-arriving donors).
- Transfer2 independently inherits its IMP/structural baselines (see
  [Transfer2 docs](../Transfer2/README.md#scorer-independent-baselines)).

## Regions JSON Schema

```json
{
  "Set_0": ["layer.start", "layer.end"],
  "Set_1": ["..."],
  "Dynamic_Region_All_Combined": [["s0", "e0"], ["s1", "e1"]],
  "Status_Map": {"layer.name": "Active|Passive|Gap|Rejected|Veto"},
  "Control_Continuted": null
}
```

`Set_*` values are single `[start, end]` collapsible spans;
`Dynamic_Region_All_Combined` is the greedy union of non-overlapping spans (the headline target);
`Status_Map`/`Control_Continuted` are diagnostics/convention keys that stage 2 skips as
non-experiments where applicable. See
[collapse/README.md § output schema](../../pyPrune/collapse/README.md#output-schema) for how
regions are produced.

## Where to go next

- Analysis of the produced artifacts → [manuscript/Transfer2/README.md](../Transfer2/README.md)
- Engine internals (probes, scorers, surrogates) → [pyPrune/collapse/README.md](../../pyPrune/collapse/README.md)
