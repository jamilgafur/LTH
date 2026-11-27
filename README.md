# Iterative Magnitude Pruning and Structural Layer Collapse for Neural Networks

This repository provides two model-compression tools in PyTorch:

1. **Iterative magnitude pruning** — gradually zero out the least significant
   weights to a target sparsity, fine-tuning after each step to recover
   performance.
2. **Structural layer collapse** (`pyPrune.collapseMethods`) — replace a
   contiguous span of layers with a single collapsed layer: blocks of
   Conv2d/Linear layers in CNNs (`manuscript/Tranfer`), or contiguous
   transformer layers in LLMs (`manuscript/llm_collapse`), for **resource-usage
   reduction of open-source transformer and MoE models** (Llama, MiMo,
   DeepSeek, Qwen-MoE, Kimi, Mixtral, ...).

The `manuscript/` folder holds the research-paper studies built on the package.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Usage](#usage)
  - [Iterative Pruning](#iterative-pruning)
  - [Collapsing Layers](#collapsing-layers)
  - [Running with Docker (GPU Support)](#running-with-docker-gpu-support)
- [Manuscript Studies](#manuscript-studies)
- [Metrics](#metrics)
- [Logging](#logging)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Overview

### Iterative magnitude pruning

1. **Pretraining**: train the model without pruning for a given number of epochs.
2. **Pruning**: iteratively zero out the weights with the smallest magnitudes
   until the target sparsity is reached.
3. **Fine-tuning**: after each pruning step, fine-tune to recover lost accuracy.
4. **Checkpointing**: save model states at each step for recovery or analysis.

Pruning strategies implement `PruningStrategy.apply()` and are injected into
`IterativePruner` (open/closed principle): `MagnitudePruningStrategy` and
`OptimalBrainDamageStrategy` ship with the package.

### Structural layer collapse

`collapse_only()` takes a `compression_set` of contiguous `(start, end)` layer
spans and replaces **each span with one collapsed layer** whose weights are
merged from the span. Two families live in `pyPrune/collapseMethods`:

- **`LLMCollapse`** — decoder/MoE transformer layers (Llama, MiMo, DeepSeek,
  Qwen-MoE, Kimi, ...): `merge="mean"` averages the shared floating-point
  tensors; `middle`/`first`/`last` copy one member. Hidden interface,
  `layer_idx` KV-cache bookkeeping and `config.num_hidden_layers` stay
  consistent, so the model keeps working while the parameter/memory/latency
  footprint drops.
- **`CNNCollapse`** — Conv2d/Linear blocks (the collapse mechanics formerly in
  `manuscript/Tranfer/collapse.py`): the span is replaced by a cheap
  pointwise/depthwise conv + BN + adaptive pool with the same interface,
  skip connections are shape-aligned, and pooling layers are wrapped
  defensively.

Fine-tuning afterwards recovers quality; the studies measure exactly how much
(see [Manuscript Studies](#manuscript-studies)).

## Repository Structure

```
├── pyPrune/                      # core package
│   ├── models/                   # CNN backbones (LeNet, VGG16, ResNet20/50,
│   │                             #   RegNetX, ConvNeXt, InceptionNet, ...)
│   ├── pruneMethods/             # IterativePruner, EarlyStopPruner, BaseTrainer
│   ├── strategies/               # PruningStrategy (OCP), MagnitudePruning,
│   │                             #   OptimalBrainDamage
│   ├── collapseMethods/          # structural layer collapse (one contiguous
│   │                             #   span -> one collapsed layer)
│   │   ├── CNNCollapse.py        #   public facade: CNN Conv2d/Linear block
│   │   │                         #     collapse (from manuscript/Tranfer)
│   │   ├── LLMCollapse.py        #   public facade: transformer-layer collapse
│   │   ├── cnn/                  #   implementation: containers, capture,
│   │   │                         #     analysis, replacement, postprocess, api
│   │   └── llm/                  #   implementation: stack, merge, api
│   └── utils.py                  # dataset loaders, model helpers
├── manuscript/                   # research-paper study code
│   ├── Tranfer/                  # CNN block-collapse study: activation-variance
│   │                             #   heuristics + collapse + retraining
│   ├── llm_collapse/             # LLM layer-collapse study (HF transformers
│   │                             #   and MoE: Llama, MiMo, DeepSeek, Kimi, ...)
│   ├── llm_ablation/             # MoE expert/region ablation study
│   ├── sparse_matrix/            # sparse CNN analysis
│   └── structured_study/         # structured pruning study
├── tests/                        # package tests
├── requirements.txt
├── setup.py
└── README.md
```

## Dependencies

- Python 3.9+
- PyTorch 2.0+
- NumPy, tqdm, torchvision (dataset loaders)
- (optional) transformers — only for the LLM studies
- (optional) CUDA for GPU acceleration

```bash
pip install -r requirements.txt
```

## Setup

```bash
git clone https://github.com/your-username/ablation_model.git
cd ablation_model/LTH
pip install -r requirements.txt
pip install -e .        # or add the repo root to PYTHONPATH
```

## Usage

### Iterative pruning

```python
import torch
from torch.optim import Adam
from pyPrune.models.LeNet import LeNet
from pyPrune.pruneMethods.IterativePruner import IterativePruner
from pyPrune.strategies.MagnitudePruningStrategy import MagnitudePruningStrategy

model = LeNet()
optimizer = Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

pruner = IterativePruner(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    criterion=criterion,
    strategy=MagnitudePruningStrategy(),
    steps=[0.2, 0.4, 0.6, 0.8],
    pretrain_epochs=5,
    finetune_epochs=2,
    save_dir="pruning_checkpoints",
)
pruner.run()
```

### Collapsing layers

CNN blocks (`Conv2d`/`Linear` spans, named-layer endpoints):

```python
from pyPrune.collapseMethods.CNNCollapse import collapse_only

model = collapse_only(
    model=model,
    compression_set=[("features.0", "features.2")],  # inclusive spans
    input_shape=(1, 3, 32, 32),
)
```

LLM decoder layers (int or `'model.layers.3'` endpoints):

```python
from pyPrune.collapseMethods import collapse_only, collapse_block, remove_layers

model = collapse_only(
    model=model,
    compression_set=[(start, end)],   # inclusive spans
    merge="mean",                     # mean | middle | first | last
    stack_name=None,                  # e.g. 'decoder.block' for T5/LLaVA towers
    inplace=True,                     # no deep-copy: required for 7B+ checkpoints
)
```

- spans may not overlap; every identified span is collapsed in one call
  (collapse-all, then one retraining phase — the protocol of both studies);
- a span is mean-mergeable only when its first layer owns every parameter of
  the later ones — spans crossing a dense→MoE boundary (MiMo, DeepSeek) are
  refused/skipped, never forced;
- works structurally on any decoder stack in an `nn.ModuleList`/`nn.Sequential`
  (Llama, MiMo, DeepSeek-V3, Qwen3-MoE, OLMoE, Mixtral, Kimi, T5, ...);
- `remove_layers()` drops a span outright (removal ablations);
- 4/8-bit quantized checkpoints are not supported: merge raw float weights
  first, quantize afterwards.

### Running with Docker (GPU Support)

You can run the project inside a Docker container with GPU support — useful
when you have an NVIDIA GPU and want CUDA-accelerated training/pruning or LLM
collapse:

```bash
docker run --rm --gpus all -it --runtime=nvidia -v $(pwd):/workspace pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
```

- `--rm`: remove the container on exit.
- `--gpus all` / `--runtime=nvidia`: expose host GPUs (NVIDIA container runtime).
- `-v $(pwd):/workspace`: mount the project into the container.

Once inside, run the pruning and study scripts as described above.

## Manuscript Studies

| folder | study | what it measures |
|---|---|---|
| `manuscript/Tranfer` | CNN block collapse (activation-variance similarity heuristics: relative variance, block redundancy, virtual-bypass KL, composite collapse score) | accuracy/params/FLOPs after collapse + retraining |
| `manuscript/llm_collapse` | LLM layer collapse — same four heuristics on hidden states; spans grouped by similar activation variance; all collapsed before one retraining phase | perplexity delta vs. resource reduction (params, size MB, forward latency) |
| `manuscript/llm_ablation` | MoE expert/region ablation driven by domain-activation correlation | perplexity/routing/poisoning before/after ablation |

Each study has its own README with install/usage instructions.

## Metrics

Pruning tracks sparsity, loss, accuracy, gradients and optimizer state (saved
to `pruning_metrics.json`). The collapse studies track perplexity, parameter
count, model size and forward latency before/after (see each study's
`report.json`).

## Logging

Training and pruning log each step (loss/accuracy, sparsity, checkpoints) via
the package logger; the studies write per-run `runs/<name>/` records.

## Testing

All test suites are fully offline (tiny random models, no hub downloads):

```bash
python -m pytest tests/ -q                        # package: pruning + CNN/LLM collapse
python -m pytest manuscript/llm_collapse/tests/ -q # LLM collapse study (toy + HF architectures)
python -m pytest manuscript/llm_ablation/tests/ -q # MoE ablation study
```

The study suites need `torch` + `transformers` (see each study's
`requirements.txt`).

## Contributing

We welcome contributions! If you find any bugs or have suggestions for
improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
