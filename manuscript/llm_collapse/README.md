# llm_collapse — Contiguous Transformer-Layer Collapse for LLMs

A research-paper example study for **collapsing contiguous layer spans in
decoder LLMs**, built on the `pyPrune` package in `../../pyPrune`. It is the
LLM counterpart of the CNN block-collapse study in `../Tranfer`: there, a
contiguous block of Conv2d/Linear layers is **replaced by a single cheaper
convolution** with the same interface (1x1/depthwise conv + BN + pool); here, a
contiguous span of decoder layers is **replaced by a single collapsed decoder
layer** with the same hidden-state interface. The collapsed layer is
initialized from the span itself — the element-wise **mean** of the span's
weights, or one of its members (`middle`/`first`/`last`) — and quality is
recovered by post-collapse fine-tuning, exactly like the post-compression
fine-tuning of the CNN study.

The collapsing itself lives in the package
(`pyPrune.collapseMethods.LLMCollapse`, top-level API `collapse_only` mirrors
`../Tranfer/collapse.py::collapse_only`); this folder is the experiment
harness, evaluation and paper artifacts around it.

## Pipeline (one run = one paper data point)

```
1. LOAD       model_io.load_model()     load a decoder LLM (HF transformers,
                                        or the offline toy GPT)
2. BASELINE   evaluate.perplexity()     perplexity + parameter footprint +
                                        generations before the collapse
3. IDENTIFY   heuristics.analyze_spans()  Transfer's four heuristics over the
                                        calibration corpus: group contiguous
                                        layers with SIMILAR ACTIVATION
                                        VARIANCE (similar feature
                                        representations) and score each group
                                        (relative variance, block redundancy,
                                        virtual-bypass KL, collapse score)
4. COLLAPSE   pyPrune.collapseMethods.collapse_only()
                                        EVERY identified span is collapsed into
                                        ONE layer (mean weight merge) in a
                                        single pass -- the whole compression
                                        set goes before retraining, exactly
                                        like the CNN study
5. FINETUNE   finetune.finetune()       ONE retraining phase recovers the loss
6. EVALUATE   evaluate.perplexity()     perplexity/footprint/samples after;
                                        delta_ppl + param_reduction per run
7. VISUALIZE  plot_results.py           summary.csv + trade-off figure over
                                        all runs
```

## Selection heuristics (mirrors ../Tranfer/transfer.py)

The CNN study identifies collapsible blocks from activation statistics: blocks
whose **feature representations are similar** (activation variance close to the
network's median, high in/out cosine similarity) are safe to replace with one
cheap layer. `heuristics.py` ports the same four metrics to decoder layers,
where the "features" are hidden states (B, T, D):

| metric | CNN study | here |
|---|---|---|
| Relative Variance | block median activation variance / global median | same, over layer output hidden states |
| Block Redundancy | cosine(block input, block output) | same, per token |
| Prediction Shift | KL after a virtual bypass (block output ← block input) | same, over next-token distributions |
| Collapse Score | `sim / (rel_var * (1 + kl))`, higher = safer | same formula |

Identification = `group_by_variance_similarity`: contiguous layers whose
activation variances are within `--variance-tol` of their neighbour form one
span. **All spans found this way are collapsed before the single retraining
phase** (`--span-policy variance`, the default). Baselines for the study:
`--span-policy relevance` (one span of `--collapse-width` layers with the
smallest residual-update relevance) and `--span-policy random`.

## Install

```bash
cd manuscript/llm_collapse
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$(cd ../.. && pwd):$(pwd)"   # repo root for pyPrune
```

`pyPrune` is imported straight from the repo root via `PYTHONPATH`
(alternatively `pip install -e ../..`).

## Usage

One run (Transfer-style: collapse every variance-similar group into one
mean-merged layer each, then one retraining phase):

```bash
python run_experiment.py --model opt-350m \
    --span-policy variance --variance-tol 0.25 --merge mean \
    --finetune-steps 50 --device cuda --skip-existing
```

Study baselines (a single span of 2 layers, picked by relevance or at random):

```bash
python run_experiment.py --model opt-350m --span-policy relevance \
    --collapse-width 2 --finetune-steps 50 --device cuda
```

Offline smoke run (tiny random GPT built into the repo, CPU, no downloads):

```bash
python run_experiment.py --model toy-lm --span-policy variance --device cpu
```

Paper sweep + summary table/figure:

```bash
bash run_all.sh cuda        # then: python plot_results.py
```

Tests (package collapse + full pipeline, fully offline):

```bash
python -m pytest tests/ -q          # this study
python -m pytest ../../tests/test_llm_collapse.py -q   # the package itself
```

## Outputs

Each run writes `runs/<name>/`:

- `config.json` — the exact configuration;
- `report.json` — baseline vs. collapsed perplexity, parameter footprint,
  the collapsed span and merge policy, the package's collapse reports,
  fine-tuning losses, and qualitative generations.

`plot_results.py` aggregates all runs into `runs/summary.csv` and
`runs/collapse_tradeoff.png` (perplexity delta vs. parameter reduction).

## HuggingFace model compatibility

The collapse is structural (no model-class checks), so it works on the
open-source transformer and MoE families people actually deploy. Each family
below is exercised by an offline test on a tiny random model built from its
real HF config (`tests/test_open_source_models.py`, `tests/test_hf_architectures.py`):

| family | examples | tested |
|---|---|---|
| dense decoder | Llama-2/3, TinyLlama, Mistral, OPT, GPT-2, GPT-NeoX/Pythia, Phi, Gemma | ✅ |
| MoE | Mixtral, Qwen3-MoE, OLMoE | ✅ |
| MoE with dense prefix + MLA | DeepSeek-V3, MiMo | ✅ |
| Kimi | Kimi Linear; K2/K2.5 checkpoints (`trust_remote_code`) follow the same `model.layers` structure | ✅ |
| encoder-decoder | T5, BART, Whisper via `--stack decoder.block` | ✅ |

Rules a model must satisfy:

- its decoder blocks live in one `nn.ModuleList`/`nn.Sequential` (use `--stack
  <path>` to disambiguate multi-tower models like T5/LLaVA);
- a span is mean-merged only when its first layer owns every parameter of the
  later ones — spans crossing a **dense → MoE boundary** (MiMo, DeepSeek) are
  skipped with a warning and reported in `skipped_spans`, never forced;
- `layer_idx` / `config.num_hidden_layers` are kept consistent so HF KV caches
  keep working after the collapse.

**Resource usage reduction on big checkpoints:** use
`collapse_only(..., inplace=True)` — the default deep-copies the model (2×
peak memory), which does not fit 7B+ checkpoints. Every run records the
resource deltas (`resource_reduction`: parameters, size MB, forward latency).
Not supported: 4/8-bit quantized checkpoints — the mean merge needs raw float
weights, so merge first, quantize after.

## Collapse semantics (package API)

```python
from pyPrune.collapseMethods import collapse_only, collapse_block, remove_layers

model = collapse_only(model=model,
                      compression_set=[(start, end)],   # inclusive spans
                      merge="mean",                     # mean|middle|first|last
                      stack_name=None,                  # e.g. 'decoder.block'
                      inplace=True)                     # no deep-copy (big models)
```

- spans are contiguous and inclusive, given as indices or names
  (`'model.layers.3'`), and may not overlap;
- `merge="mean"` averages the span's floating-point tensors into the one
  collapsed layer (non-float buffers come from the first layer);
- `layer_idx` bookkeeping and `config.num_hidden_layers` are kept consistent
  for HF KV caches;
- `remove_layers()` drops a span outright (used by the removal ablations in
  `../llm_ablation`).
