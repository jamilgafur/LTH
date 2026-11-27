# llm_ablation — Domain-Aware Region Ablation for MoE LLMs

A modular example study for **ablating Mixture-of-Experts (MoE) large language
models**, built on the `pyPrune` package in `../../pyPrune`. It is the LLM
counterpart of the CNN structural-removal study in `../Tranfer`: there,
trained CNNs have unneeded contiguous layer blocks collapsed away; here, an
MoE LLM is queried with **domain-labelled queries**, its activations are
correlated with those domains, and the regions of the model that the target
workload does not need are **replaced out of the model** — at the finest, a
whole expert; at the coarsest, a contiguous span of transformer layers or a
multi-layer x multi-expert region. Nothing is ever removed inside an expert's
weights.

## Pipeline (one run = one paper data point)

```
1. LOAD       model_io.load_model()        load an MoE LLM (HF transformers)
2. QUERY      queries.load_domains()       run domain-labelled queries through
                                           the model (math, roman_history, ...)
3. STORE      activations.ActivationRecorder
              per query and per expert: top-k selection counts, gate probs,
              (classic layout) output magnitudes -> activations.pt
4. ANALYZE    analysis.domain_correlations()
              correlate each expert's per-query activation with each domain:
              "which experts fire for which domain" -> analysis.json
5. REPLACE    ablate.*                      remove unneeded regions:
              - ablate_experts()    whole experts (lowest domain correlation)
              - collapse_layers()   one contiguous layer span (Transfer-style)
              - collapse_region()   a layer span in which every layer drops
                                    its lowest-relevance experts
6. EVALUATE   evaluate.run_eval()          perplexity before/after, parameter
              footprint, routing stats, qualitative generations
7. VISUALIZE  dashboard.py                 self-contained dashboard.html over
              all runs + tidy CSV exports for paper figures
```

Expert relevance is the **Pearson (point-biserial) correlation** between an
expert's per-query usage and membership in the domains the deployment must
serve (`--protect-domains`). Experts correlated with, say, `roman_history` but
not with the protected `math` workload are removed first; `--rank-by usage`
or `gate_prob` are usage-based baselines.

## Install

```bash
cd manuscript/llm_ablation
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$(cd ../.. && pwd):$(pwd)"   # repo root for pyPrune
```

`pyPrune` is imported straight from the repo root via `PYTHONPATH`
(alternatively `pip install -e ../..`).

## Usage

One run (protect the math + code experts, remove everything else as regions):

```bash
python run_experiment.py \
    --model mixtral-8x7b --query-set domains \
    --ablation region --collapse-layers 2 --keep-fraction 0.5 \
    --rank-by max_corr --protect-domains math,code \
    --device cuda --dtype float16 --skip-existing
```

Offline smoke run (tiny random Mixtral-architecture model, CPU, no downloads):

```bash
python run_experiment.py --model toy-moe --query-set domains \
    --ablation region --collapse-layers 1 --keep-fraction 0.5 --device cpu
```

Paper sweep + summary table/figure:

```bash
bash run_all.sh cuda        # then: python plot_results.py --runs runs
```

Tests (both expert layouts, all ablation modes, fully offline):

```bash
python -m pytest tests/ -q
```

## Dashboard & visualization

`dashboard.py` turns every saved run into one **self-contained HTML page**
(`runs/dashboard.html`) — no server, no JS libraries, opens straight from
disk. It visualizes, per run:

- **the model** — layer x expert diagram with kept experts, removed experts
  and removed layers marked;
- **the input** — the domain-labelled queries used in the run;
- **the activations** — expert intensity heatmaps per layer (usage /
  gate_prob / output_norm) and a queries x experts heatmap for the selected
  layer ("which input lit up which expert");
- **the correlation** — domain x expert correlation heatmaps and a
  top-experts-per-domain table;
- **generations** — baseline vs. ablated completions.

```bash
python dashboard.py --runs runs            # -> runs/dashboard.html
```

`run_all.sh` builds it automatically after the sweep.

## Data exports for paper figures

Beyond the raw artifacts (`activations.pt`, `analysis.json`, ...), every run
gets tidy CSVs under `runs/exports/<run>/` ready for R/gnuplot/matplotlib:

- `expert_data.csv` — one row per (layer, expert): usage, gate_prob,
  output_norm, top_domain, max_corr, `corr_<domain>` per domain, kept flag
  (heatmaps, correlation bars, ablation tables);
- `per_query_usage.csv` — one row per (query, layer, expert): usage
  (query-activation figures).

Plus `runs/summary.csv` and `runs/delta_ppl_vs_params.png` from
`plot_results.py`.

## Poison / backdoor phrase removal

The same pipeline removes **poisoned behavior**: a backdoor that, say, prepends
*"you are a fish"* to the output whenever unknown trigger words appear in the
context. You do not need to know the triggers — only which outputs are
unwanted:

1. **Submit queries** across domains as usual (`--query-set domains`).
2. **Select the phrases** to remove: `data/poison_phrases.txt`, one phrase per
   line (updatable between runs; `--poison-phrases` points at another file,
   and a large set of known poisoned outputs works the same way).
3. Every query gets a **poison score** (`--poison-scorer`):
   `logprob` (default) — the model's own geometric-mean probability of
   *emitting* the phrase given the query (teacher-forced), sensitive even when
   greedy decoding would pick another token; `lexical` — difflib similarity of
   the generated output against the set. Both handle large signature sets and
   near-matches.
4. `analysis.score_correlations()` correlates each expert's per-query
   activations with those scores and **identifies the poisoned region** —
   the experts/layers whose activations track the poisoned outputs.
5. **Causal validation** (`causal.py`): the to-be-removed experts are
   *blocked* (activation patching — their routing contribution becomes zero)
   and the poison score is re-measured against random same-size expert sets.
   `metrics.json: poison_causal` reports `suspect_drop`, `control_drops`,
   `causal_ratio` and `beats_controls`: correlation says "fires together",
   this says "causes it".
6. **Trigger discovery** (`triggers.py`): occlusion attribution (remove each
   input token, re-measure phrase probability) over the flagged queries,
   clustered by the activations of the suspect experts → candidate trigger
   words per cluster (`triggers.json`). Moves you from "which region" to
   "which input pattern".
7. `--rank-by poison` then **collapses/replaces that region** (any of the
   `experts` / `layers` / `region` modes; the most poison-correlated region is
   removed instead of the least-relevant one).
8. The **triggered rate and mean similarity are re-measured after ablation**
   (`metrics.json: poison`), so you can show the backdoor is gone.

```bash
python run_experiment.py \
    --model mixtral-8x7b --query-set domains \
    --ablation region --collapse-layers 2 --keep-fraction 0.5 \
    --rank-by poison --poison-threshold 0.8 \
    --device cuda --dtype float16
```

Data saved for figures: `analysis.json: poison_correlations` (expert x poison
correlation), `metrics.json: poison` (before/after rates, per-query scores and
continuations), and `poison_corr` / `poison_score` columns in the CSV exports.

## Removal baselines (beat random)

`run_baselines.py` runs the identical configuration with different ranking
strategies at **equal sparsity** and writes one comparison table:

```bash
python run_baselines.py --model mixtral-8x7b --query-set domains \
    --ablation experts --keep-fraction 0.5 --rank-bys max_corr,usage,gate_prob,random
# poison study: --rank-bys poison,usage,gate_prob,random
# -> runs/baselines_<model>_<query_set>_<ablation>.csv
```

Columns: delta_ppl, params_after, poison_rate_after. `--rank-by random` is
also available directly in `run_experiment.py`.

## Activation cache

Runs that share (model, seed, dtype, queries, max_length) reuse recorded
activations from `.cache/activations/` (keyed by a hash of exactly those) —
sweeping ablation modes no longer re-records identical queries. Force
re-recording with `--no-cache`; the cache location is `--cache-dir`.

## Ablation modes

| mode | unit removed | selection |
|---|---|---|
| `experts` | whole experts in every layer | lowest relevance per layer, keeps `--keep-experts` / `--keep-fraction` |
| `layers` | one contiguous span of transformer layers | least-relevant span of `--collapse-layers` layers (or `--collapse-start`) |
| `region` | contiguous layer span x experts | span as above; within it each layer keeps only its top experts |

## Output layout

```
runs/<model>_<query_set>_<ablation>_.../
├── config.json              # full RunConfig for reproducibility
├── metrics.json             # baseline vs. ablated ppl, footprint, routing
├── activations.pt           # per-query usage records + router logit sample
├── activations_summary.json # aggregate + per-query usage/gate_prob/output_norm
├── analysis.json            # domain x expert correlations, top experts/domain
├── ablation_report.json     # exact regions removed (kept/dropped indices)
├── triggers.json            # trigger-token candidates per query cluster
├── samples_baseline.txt     # generations before ablation
├── samples_ablated.txt      # generations after ablation
└── ablated_model/           # with --save-model: save_pretrained() of the
                             # replacement model (the "replaced" MoE LLM)
runs/exports/<run>/expert_data.csv, per_query_usage.csv   # tidy figure data
runs/summary.csv, runs/delta_ppl_vs_params.png             # plot_results.py
runs/dashboard.html                                        # dashboard.py
```

## Extending (modularity for multiple runs / LLMs)

- **New MoE LLM** — add one entry to `MODEL_REGISTRY` in `config.py`. Anything
  `AutoModelForCausalLM` can load works as long as each MoE layer has an
  `experts` child plus a router of matching width; discovery in
  `model_io.find_moe_blocks` is structural, not model-class based. Verified
  against Mixtral-style blocks with both the classic per-expert-`nn.Linear`
  layout (transformers < 5) and the fused `[num_experts, ...]`-tensor layout
  (transformers >= 5).
- **New domain / query set** — add to `DOMAIN_QUERY_SETS` in `config.py`, or
  point `--query-set` at a `.txt` file with `[domain]` sections (see
  `data/queries/domains.txt`).
- **New region rule** — implement it in `ablate.py`; anything that shrinks
  the model and returns a dict report plugs into the pipeline.
- **New relevance signal** — add a function to `analysis.py` (or a method to
  `ActivationRecorder.importance_scores`) and a `--rank-by` choice.

## Notes and limitations

- The toy model is *randomly initialized*: its perplexities and near-zero
  deltas validate the pipeline only, not the science.
- Domain correlation needs **at least two domains** for contrast
  (`--query-set domains`); single-domain query sets use `--rank-by usage`.
- Real MoE checkpoints (Mixtral-8x7B etc.) need a CUDA GPU and the download
  budget of the checkpoint; `run_all.sh`'s model list is a starting point.
- Ablation is *post-training, without recovery fine-tuning*; a natural next
  step is short fine-tuning of the replaced model, which `pyPrune`'s
  `IterativePruner` could drive.
- `output_norm` relevance needs per-expert modules and is unavailable on the
  fused layout (transformers >= 5 Mixtral-family); `max_corr`/`usage`/`gate_prob`
  work on both layouts.
- Region removal shrinks parameters and memory; wall-clock speedups require a
  runtime that exploits the smaller model (e.g. serving the saved
  `ablated_model/`).
