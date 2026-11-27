"""Structural layer-collapse methods.

- ``CNNCollapse``: contiguous Conv2d/Linear block collapse for CNNs
  (the collapse mechanics of the ``manuscript/Tranfer`` study) --
  use ``CNNCollapse.collapse_only(...)``.
- ``LLMCollapse``: contiguous transformer-layer collapse for decoder/MoE
  LLMs (``manuscript/llm_collapse``) -- use ``collapse_only(...)``
  (alias of ``LLMCollapse.collapse_only``).

Both replace an inclusive contiguous span of layers with a single collapsed
layer keeping the same interface; each family's ``collapse_only`` has its own
signature, so prefer the explicit module paths above.
"""
from . import CNNCollapse, LLMCollapse

# decoder-LLM collapse convenience API
from .LLMCollapse import (
    MERGE_MODES,
    collapse_block,
    collapse_only,
    count_params,
    find_decoder_stack,
    remove_layers,
)

__all__ = [
    "CNNCollapse",
    "LLMCollapse",
    "MERGE_MODES",
    "collapse_block",
    "collapse_only",
    "count_params",
    "find_decoder_stack",
    "remove_layers",
]
