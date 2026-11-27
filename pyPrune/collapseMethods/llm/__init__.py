# llm -- contiguous transformer-layer collapse for decoder/MoE LLMs.
# Public entry point: pyPrune.collapseMethods.LLMCollapse (facade) or
# pyPrune.collapseMethods (package re-exports).
from .api import (collapse_block, collapse_only, count_params,
                  remove_layers)
from .merge import MERGE_MODES, _merge_blocks
from .stack import (find_decoder_stack, _resolve_index,
                    _renumber_layer_idx, _sync_config)

__all__ = [
    "MERGE_MODES",
    "collapse_block",
    "collapse_only",
    "count_params",
    "find_decoder_stack",
    "remove_layers",
]
