# model_io.py -- load MoE LLMs and discover their expert / router structure.
# Discovery is structural (not class-based) so the same pipeline works across
# Mixtral, OLMoE, Qwen-MoE, DBRX, ... : every block with an `experts` child
# plus a router whose width == number of experts is found.
#
# Two expert layouts are supported:
#   classic: `experts` is a ModuleList of per-expert MLPs (transformers < 5)
#   fused:   `experts` is one module with [num_experts, ...] parameters
#            (transformers >= 5 Mixtral-style grouped-GEMM experts)
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

# repo root on sys.path so `pyPrune` imports without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import get_model_config

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

# attribute names HF implementations use for the top-k of the router
TOP_K_ATTRS = ("top_k", "n_activated_experts")


@dataclass
class MoEBlock:
    """One MoE layer: the block module, its experts container and its router."""
    name: str
    block: nn.Module
    experts_attr: str
    router_attr: str
    num_experts: int
    fused: bool

    @property
    def experts_container(self) -> nn.Module:
        return getattr(self.block, self.experts_attr)

    @property
    def experts(self) -> List[nn.Module]:
        """Per-expert modules (classic layout only; empty list when fused)."""
        c = self.experts_container
        return list(c) if isinstance(c, nn.ModuleList) else []

    @property
    def router(self) -> nn.Module:
        return getattr(self.block, self.router_attr)

    @property
    def top_k(self) -> int:
        for attr in TOP_K_ATTRS:
            if isinstance(getattr(self.block, attr, None), int):
                return getattr(self.block, attr)
        return 1


def router_width(router: nn.Module) -> int:
    return getattr(router, "out_features", router.weight.shape[0])


def _find_router_attr(block: nn.Module, experts_attr: str, num_experts: int) -> Optional[str]:
    """Router is a linear-ish module in the same block with width == num_experts."""
    for attr, sub in block.named_children():
        if attr == experts_attr:
            continue
        if isinstance(sub, nn.Linear) and sub.out_features == num_experts:
            return attr
        w = getattr(sub, "weight", None)  # e.g. MixtralTopKRouter keeps a raw Parameter
        if isinstance(w, torch.Tensor) and w.dim() == 2 and w.shape[0] == num_experts \
                and not isinstance(sub, (nn.Linear, nn.Embedding)):
            return attr
    return None


def _expert_container_size(container: nn.Module) -> Optional[int]:
    """Number of experts, or None if this is not an experts container.

    classic: number of ModuleList entries.
    fused:   the shared leading dimension of the container's own parameters.
    """
    if isinstance(container, nn.ModuleList):
        return len(container) if len(container) >= 2 else None
    dims = {p.shape[0] for p in container.parameters(recurse=False) if p.dim() >= 2}
    dims |= {b.shape[0] for b in container.buffers(recurse=False) if b.dim() >= 2}
    if len(dims) == 1 and next(iter(dims)) >= 2:
        return next(iter(dims))
    return None


def find_moe_blocks(model: nn.Module) -> List[MoEBlock]:
    """Locate every MoE block in the model (see module docstring)."""
    blocks = []
    for name, module in model.named_modules():
        experts = getattr(module, "experts", None)
        if experts is None:
            continue
        n = _expert_container_size(experts)
        if n is None:
            continue
        router_attr = _find_router_attr(module, "experts", n)
        if router_attr:
            fused = not isinstance(experts, nn.ModuleList)
            blocks.append(MoEBlock(name, module, "experts", router_attr, n, fused))
    return blocks


class ByteTokenizer:
    """Minimal offline tokenizer for the toy model: bytes -> ids 2..257."""

    pad_token_id = 0
    eos_token_id = 1
    vocab_size = 258

    def encode(self, text: str) -> List[int]:
        return [b + 2 for b in text.encode("utf-8")]

    def decode(self, ids) -> str:
        return bytes([int(i) - 2 for i in ids if int(i) >= 2]).decode("utf-8", "ignore")


def _build_toy_model(dtype: torch.dtype):
    """Tiny random Mixtral-architecture model for offline tests (from config)."""
    from transformers import AutoModelForCausalLM, MixtralConfig
    cfg = MixtralConfig(
        vocab_size=258, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        num_local_experts=8, num_experts_per_tok=2,
        max_position_embeddings=512, pad_token_id=0, eos_token_id=1,
    )
    model = AutoModelForCausalLM.from_config(cfg).to(dtype)
    return model, ByteTokenizer()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model(model_name: str, device: str = "auto", dtype: str = "float32"):
    """Return (model, tokenizer, load_report) ready for inference."""
    entry = get_model_config(model_name)
    dtype = entry.get("dtype", dtype) if entry.get("build") != "toy" else dtype
    torch_dtype = DTYPES[dtype]
    dev = resolve_device(device)

    if entry.get("build") == "toy":
        model, tokenizer = _build_toy_model(torch_dtype)
        model.to(dev)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(entry["hf_id"], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            entry["hf_id"],
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation=entry.get("attn_implementation", "sdpa"),
        )
        model.to(dev)

    model.eval()
    blocks = find_moe_blocks(model)
    if not blocks:
        raise ValueError(f"No MoE expert blocks found in '{model_name}'; is it an MoE model?")
    report = {
        "model": model_name,
        "hf_id": entry.get("hf_id", "toy-from-config"),
        "device": str(dev),
        "dtype": dtype,
        "expert_layout": "fused" if blocks[0].fused else "classic",
        "moe_layers": len(blocks),
        "experts_per_layer": [b.num_experts for b in blocks],
        "top_k": blocks[0].top_k,
    }
    return model, tokenizer, report
