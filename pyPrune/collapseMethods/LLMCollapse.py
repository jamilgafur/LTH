# LLMCollapse.py -- collapse contiguous transformer layers into one, the LLM
# counterpart of the CNN block collapse (CNNCollapse.py).
#
# There, a contiguous block of Conv2d/Linear layers is replaced by a single
# cheaper convolution with the same interface (1x1/depthwise conv + BN + pool);
# here, a contiguous span of decoder layers is replaced by a single collapsed
# decoder layer with the same hidden-state interface. The collapsed layer is
# initialized from the span itself ("mean" weight merge, or one member of the
# span: "middle" / "first" / "last") and quality is recovered by fine-tuning,
# exactly like the post-compression fine-tuning in the CNN study.
#
# This module is the public facade; the implementation lives in the `llm`
# subpackage: `stack` (decoder-stack discovery and layer_idx/config
# bookkeeping), `merge` (span -> one layer), `api` (collapse_block,
# remove_layers, collapse_only).
from .llm.api import (collapse_block, collapse_only, count_params,
                      remove_layers, _normalize_compression_set)
from .llm.merge import MERGE_MODES, _merge_blocks
from .llm.stack import (find_decoder_stack, _resolve_index,
                        _renumber_layer_idx, _sync_config)

__all__ = [
    "MERGE_MODES",
    "collapse_block",
    "collapse_only",
    "count_params",
    "find_decoder_stack",
    "remove_layers",
]
