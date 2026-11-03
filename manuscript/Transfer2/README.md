# manuscript/Transfer2 — Analysis Pipeline

[← Top-level README](../../README.md) · [Training: Tranfer](../Tranfer/README.md) ·
[pyPrune](../../pyPrune/README.md)

Post-hoc measurement pipeline: given a universe of trained artifacts, it loads the **original**
(post-finetune control), the **collapsed** whole-model artifact, and computed-matched
**unstructured IMP** and **structural L1** baselines, then measures accuracy, power/energy,
FLOPs, latency/throughput, memory, storage, CKA, adversarial robustness, calibration, OOD
detection, and corruption robustness — per checkpoint, GPU-sequentially to stay within VRAM.

## Table of Contents

- [Model roster spanned](#model-roster-spanned)
- [CLI reference](#cli-reference-mainpy)
- [Checkpoint discovery & matching](#checkpoint-discovery--matching)
- [Scorer-independent baselines](#scorer-independent-baselines)
- [Metric catalog](#metric-catalog)
- [Outputs](#outputs)
- [Where to go next](#where-to-go-next)

## Model Roster Spanned

Per processed `final_JF_Control.pt` ("before") checkpoint:

| Key | What it is | How obtained |
|---|---|---|
| `original` | control after continued fine-tuning | `final_JF_Control_Continuted.pt`, same universe |
| `collapsed` | variance/change/CKA-selected collapsed model | `final_JF_Dynamic_Region_All_Combined_whole.pt`, per-directory fallback `final_JF_Set_0_whole.pt` (whole-model pickle; a loud `FileNotFoundError` names the fix if absent) |
| `unstructured` | IMP at param-matched sparsity | trained on the fly if `final_JF_Unstructured_IMP_post{N}.pt` absent (or inherited, see below) |
| `structural` | L1 filter pruning at param-matched sparsity (torch-pruning DepGraph) | trained on the fly if `final_JF_Structural_L1_post{N}.pt` absent; skipped cleanly (columns `None`) if `torch-pruning` is not installed |

## CLI Reference (`main.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--model` | required | Architecture (e.g. `InceptionNet`) |
| `--dataset` | required | Dataset (e.g. `Cifar100`) |
| `--pre`, `--post` | `None` | Optional filters on `pretrain{M}`/`epochs{N}` tokens; **derived from checkpoint paths when omitted** |
| `--output` | `auto` | Results CSV; auto-named from model/dataset/(derived or override) epochs when unset |
| `--power-interval` | `1` | Power sampling interval (s, float) for the energy measurements; sub-second values down to 0.1 keep short runs well sampled (NVIDIA: `--loop-ms`) |
| `--analysis` | `standard` | `standard` = historical behavior; `full` adds adversarial ε-sweeps, calibration, OOD, and corruptions |
| `--corruptions-dir` | unset | CIFAR-C data root (stage via `download_corruptions.sh` on a login node); enables corruption evaluation under `--analysis full` |
| `--quant` | off | Analyze the fp16-quantized (PTQ) universe: loads the **same fp32 artifacts** as the default run and measures them converted to fp16 (autocast inference, fp16 storage estimate); no retraining involved. CSV outputs get a `_quant` infix so both universes coexist per directory. Legacy `*_quant.pt` artifacts (QAT-trained) are not read |
| `--reps` | `1` | Measurement repetitions for the accuracy/power/energy eval and the efficiency benchmark (sections 6–7); the CSV gets **one row per rep** (`rep` column) so plot-side medians can reject noisy power samples (use ~10). Deterministic steps (CKA, adversarial, calibration) run once. `run_postprocess.sh` jobs default to `REPS=10` (env-overridable) and resubmit CSVs with fewer than `REPS` real-GPU rows |

## Checkpoint Discovery & Matching

- Discovery globs `../Tranfer/{model}_{dataset}_*/checkpoints/` for the three buckets
  (`before`, `after/Continuted`, `collapsed *_whole.pt`) — with `--quant` the **same FP
  files** are discovered (PTQ conversion happens at measurement time). Epoch tokens are
  **not** baked into the glob, so multiple configs (`epochs100_pretrain300`,
  `epochs300_pretrain100`, …) coexist.
- Each before-checkpoint is matched to artifacts **in its own base directory**
  (`find_matching_checkpoint`) — epoch tokens come from `parse_epoch_tokens(path)`
  (`post ↔ epochs{N}`, `pre ↔ pretrain{M}`). The IMP training length and IMP/structural
  checkpoint names derive from the parsed `epochs` token (with `--post` as an explicit
  override only).
- Scorer universes (`{tag}_scorer{S}_epochs{E}_pretrain{P}`) are handled by the same rules —
  the scorer token is just part of the directory identity.
- **Missing-Control fallback:** default-scorer dirs that never ran the Control experiment
  borrow the `scorerall` sibling's `final_JF_Control{,_quant}.pt` as their `before`
  (the scorer affects only region discovery, not Control training); matching then looks in
  the default sibling dir for the collapsed/after artifacts, and derived baselines
  (IMP/structural) are saved into the default dir.
- **Optional `after`:** when `final_JF_Control_Continuted*.pt` does not exist yet (still
  training), the pipeline runs a baseline-only pass — sparsity, IMP/structural training,
  and the collapsed-variant metrics — writing a partial row with `None` for the
  `*_original*` / `cka_*` columns. Re-running after the artifact lands overwrites the same
  CSV with the full row; the expensive baselines are reused from disk.

## Scorer-Independent Baselines

IMP and Structural retrain from the universe's "before" model — which scorer universes inherit
from the default universe — making those artifacts scorer-independent.
`inherit_from_default_universe()` maps a missing scorer-universe checkpoint to its
default-universe sibling (regex strip of the `_scorer{S}` token) and copies it with a loud log
line, keeping baselines byte-identical across universes and avoiding per-universe retraining.
(Tranfer does the same for Control/Control_Continued — see
[Tranfer docs](../Tranfer/README.md#artifact-inheritance).)

## Metric Catalog

Result-dict columns (wide CSV), grouped by family:

**Identity**: `model`, `dataset`, `gpu_name`, `pre_epochs`, `post_epochs`, `model_before`,
`intermediate_epoch`.

**Accuracy / energy** (per variant `original`/`unstructured`/`collapsed`, plus `structural`
when the optional dep runs):
`acc_{variant}` (variants use names `acc_original_after`, `acc_unstructured`, `acc_collapsed`,
`acc_structural`), `power_watts_{variant}`, `energy_joules_{variant}` — measured with
`pyPrune.power.PowerTracker` during full test-set evaluation, after **one untracked warmup
pass** (warms the disk/page cache and autotuner so the tracked joules reflect compute, not
dataloader I/O). Joules are avg watts × elapsed over the tracked window; if the sampler
collects **zero samples** (unsupported driver flags, missing root, region too fast), the
cells are written **empty** and a `[WARN]` is logged — no reading is ever fabricated.
`gpu_util_{variant}` / `gpu_clock_mhz_{variant}` record average SM utilization/clock during the
tracked pass (None on CPU) — diagnostics for the "collapsed draws more watts" effect.
Util/clock samples are range-gated (util 0–100%, clock ≤3000 MHz, positional by query
order) so `[Not Supported]` placeholders and garbled telemetry can never land in the CSV.

**Window audit** (per variant): `eval_elapsed_s_{variant}`, `n_power_samples_{variant}` —
eval-window duration and power-sample depth, so every energy row carries the evidence of
what it measured. Config-level: `loader_time_s` / `loader_joules` / `loader_n_samples` from
`measure_loader_floor` — the data pipeline iterated once with **no forward pass**; the floor
every variant's eval window shares. Plot-side `loader_frac_*` / `loader_time_frac_*` columns
expose how much of a measured "energy" number is I/O vs compute.

**Saturated compute-only energy** (per variant, from `benchmark_model` with
`measure_energy=True`): forwards of one fixed GPU-resident batch — zero data pipeline in
the window — run until ~2 s of accumulated compute so the sampler sees enough samples.
`power_watts_saturated_{variant}`, `j_per_img_saturated_{variant}` (the FLOPs→joules
transferability numerator), `saturated_imgs_{variant}` and
`n_power_samples_saturated_{variant}`. Self-check: `[CHECK]` is logged when the saturated
rate deviates >2× from the loader-based `throughput_imgs` of the same run.

**Efficiency tetrad** (`benchmark_model`, CUDA-event timed, 20 batches):
`flops_{variant}` (fvcore measured; `flops_unstructured` keeps the legacy *theoretical*
dense-scaled value, `flops_unstructured_measured` reports the real number),
`latency_ms_{variant}`, `throughput_imgs_{variant}`, `peak_mem_mb_{variant}` (CUDA peak during
benchmark; `None` on CPU), `feature_maps_mb_{variant}` (activation bytes of one batch).

**Space**: `storage_mb_{variant}` (serialized `state_dict` size), `params_structural`,
`sparsity_structural` (structural-only; collapse sparsity is the matching target).

**Representation**: `cka_unstructured`, `cka_collapsed`, `cka_structural` (vs. original
features, sequential extraction).

**Adversarial** (`evaluate_adversarial_robustness`, FGSM + PGD-20 on one batch):
- `standard` mode: `adv_acc_FGSM_{model}`, `adv_acc_PGD_{model}` at ε=0.03 (legacy column names preserved);
- `full` mode: sweep ε ∈ {0.005, 0.01, 0.02, 0.03, 0.05} — ε=0.03 keeps the legacy names, other
  points get `adv_acc_{attack}_eps{ε}_{model}` columns **and** rows in the tidy curve CSV.

**Full-analysis extras** (`--analysis full` only, per variant):
`ece_{variant}` / `mce_{variant}` (15-bin calibration on the full test set),
`ood_auroc_{variant}` (max-softmax AUROC vs. SVHN for CIFAR datasets; rank-based, no sklearn)
and, with `--corruptions-dir`: `corrupt_mca_{variant}` (mean accuracy over 15 corruptions × 5
severities; raises a loud `FileNotFoundError` with staging instructions if the dir is missing).

## Outputs

| File | Format | Contents |
|---|---|---|
| `cka_results_{model}_{dataset}[ _pre{P}_post{E}].csv` | wide, one row per before-checkpoint | the full metric catalog above; combos whose discovery map has no collapsible regions at all (no `Dynamic_Region_All_Combined` and no `Set_*` spans) instead get a verdict row with `status="no_collapsed_regions"` and metric columns empty. Single-set maps are not verdicts — their `final_JF_Set_0_whole.pt` artifact is consumed by the same collapsed-artifact fallback that prefers `Dynamic_Region_All_Combined` |

**Never-clobber rule:** if every checkpoint in a run fails, the existing CSV is left
untouched and the failure is logged with a full traceback — an empty DataFrame is
never saved over real data. Marker/verdict rows are overwritten by a successful rerun
once the combo's collapsed artifact exists.
| `robustness_curves_{stem}.csv` | tidy long | `attack,epsilon,model,adv_acc` |
| `calibration_{stem}.csv` | tidy long | `model,bin_lo,bin_hi,n,acc,conf` (reliability diagrams) |
| `corruption_{stem}.csv` | tidy long | `model,corruption,severity,acc` |

`stem` = the results CSV's basename without extension, so per-config tidy files accumulate
alongside the wide table.

Sibling tooling:

| File | Purpose |
|---|---|
| `aggregate_results.py` | pools every `cka_results_*.csv` (any header schema, column-unioned), drops CPU rows when a GPU twin exists, adds within-row variant/original ratios + J/img, saturated-J/img ratios, window-dilution fractions (`loader_frac_*`), `energy_transfer` and the `regime` label, writes `aggregated_rows.csv` / `aggregated_median.csv` — the cross-GPU apples-to-apples tables |
| `diagnose_cka.py` | audits why a config's CKA is ~0: logs which layer each variant's feature extractor strips, plus drift-reference CKAs (extraction bug vs genuine divergence) |
| `run_postprocess.sh` | submits the config grid but **skips configs whose CSV already has ≥ REPS GPU (non-CPU) measurement rows in the current schema** (saturated-energy / loader-floor columns present — old-schema rows resubmit), or a `no_collapsed_regions` verdict row while the collapsed artifact is still untrained; CPU-only, marker-with-artifact-now-trained, old-schema and missing configs (re)run |

### Corruption data staging

Compute nodes have no internet; stage once on a login node:

```bash
./download_corruptions.sh ./corruption_data   # CIFAR-10-C + CIFAR-100-C (Zenodo)
python main.py --model InceptionNet --dataset Cifar100 \
    --analysis full --corruptions-dir ./corruption_data
```

## Where to go next

- Producing the artifacts analyzed here → [manuscript/Tranfer/README.md](../Tranfer/README.md)
- Collapse engine internals (what `collapsed` models are) →
  [pyPrune/collapse/README.md](../../pyPrune/collapse/README.md)
