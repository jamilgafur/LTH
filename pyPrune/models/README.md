# pyPrune.models — Model Zoo

[← pyPrune package](../README.md) · [Top-level README](../../README.md)

Nine architecture families (ten classes) with a shared construction contract used by the pipelines: dataset-driven
`num_classes` / input-shape inference, and (for the experimented architectures) a one-batch
probe tensor passed via `model_kwargs["one_batch"]` for shape discovery during collapse and
discovery.

## Inventory

| File | Class | Family | Used in pipelines |
|---|---|---|---|
| `Vgg16.py` | `VGG16` | Sequential VGG | ✅ (6-arch matrix) |
| `RegNetX.py` | `RegNetX_400MF` | ResNet-family (AnyNetX design space) | ✅ |
| `InceptionNet.py` | `InceptionNet` | Parallel-branch (Inception v1-style) | ✅ |
| `XceptionNet.py` | `XceptionNet` | Depthwise-separable (`skip`+`rep` blocks) | ✅ |
| `MobileNet.py` | `MobileNet` | Depthwise-separable (inverted residuals) | ✅ |
| `ConvNetX.py` | `ConvNeXt` | Modernized ResNet (channels-first LayerNorm) | ✅ |
| `ResNet20.py` / `ResNet50.py` | ResNets | Classic residual | reference |
| `LeNet.py` | LeNet | Tutorial-scale | reference (heritage IMP demos) |
| `EfficientNet.py` | EfficientNet | Compound-scaled | reference |

## Construction contract

The pipelines instantiate models through `initialize_architecture(model_name, dataset_name)`
(manuscript-side), which looks up dataset presets (num classes, input size, normalization) and
builds kwargs including a real batch (`one_batch`) taken from the training loader. Models that
reject `one_batch` fall back to default construction with a logged warning.

## Collapse compatibility notes

- **Residual/parallel forwards** (`RegNetX`, `InceptionNet`, `XceptionNet`, `ConvNeXt`): handled
  by the attribute-signature patching registry — see
  [pyPrune/collapse/README.md § patching](../collapse/README.md#safety-machinery).
- **Strict sequential** (`VGG16`, `LeNet`): collapse works unpatched; the LCA machinery treats
  the whole model as one container.
- **InceptionNet on CIFAR inputs**: stride-2 pool chain shrinks 32×32 maps to 2×2, below its
  3×3 pool kernels — `_SafePool` adaptation is required (applied automatically by the pipeline).
- **ConvNeXt**: channels-first normalization → surrogates use `_SurrogateLayerNorm2d`.

Where to go next: [collapse engine internals](../collapse/README.md) ·
[training pipeline](../../manuscript/Tranfer/README.md)
