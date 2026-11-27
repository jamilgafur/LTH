# model_io.py -- load decoder LLMs for the collapse study and provide the
# offline calibration corpus. Two sources:
#   toy: a tiny random pre-LN GPT (torch-only) for smoke tests, no downloads;
#   HF:  any AutoModelForCausalLM with a homogeneous decoder-layer stack.
# The forward contract is HF's: model(input_ids=..., labels=...) -> .loss/.logits
# and model.generate(...); the toy model implements the same surface.
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# repo root on sys.path so `pyPrune` imports without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import get_model_config

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


class ByteTokenizer:
    """Minimal offline tokenizer for the toy model: bytes -> ids 2..257."""

    pad_token_id = 0
    eos_token_id = 1
    vocab_size = 258

    def encode(self, text: str) -> List[int]:
        return [b + 2 for b in text.encode("utf-8")]

    def decode(self, ids) -> str:
        return bytes([int(i) - 2 for i in ids if int(i) >= 2]).decode("utf-8", "ignore")


class TinyBlock(nn.Module):
    """Pre-LN decoder block: x + attn(ln1(x)); x + mlp(ln2(x))."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    """Tiny random GPT for offline runs. HF-like: config.num_hidden_layers,
    forward(input_ids, labels) -> (loss, logits), greedy generate()."""

    def __init__(self, vocab: int = 258, d: int = 64, layers: int = 6,
                 heads: int = 4, max_len: int = 512):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList([TinyBlock(d, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.config = SimpleNamespace(num_hidden_layers=layers, vocab_size=vocab)

    def _causal_mask(self, n, device):
        return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), 1)

    def forward(self, input_ids, labels=None):
        n = input_ids.shape[1]
        x = self.embed(input_ids) + self.pos(torch.arange(n, device=input_ids.device))
        mask = self._causal_mask(n, input_ids.device)
        for layer in self.layers:
            x = layer(x, mask)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                   labels[:, 1:].reshape(-1))
        return SimpleNamespace(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=32, do_sample=False):
        out = input_ids
        for _ in range(max_new_tokens):
            logits = self(out).logits[:, -1]
            nxt = logits.argmax(-1, keepdim=True) if not do_sample \
                else torch.multinomial(logits.softmax(-1), 1)
            out = torch.cat([out, nxt], dim=1)
        return out


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model(model_name: str, device: str = "auto", dtype: str = "float32") -> Tuple:
    """Return (model, tokenizer, load_report) ready for inference."""
    entry = get_model_config(model_name)
    dtype = entry.get("dtype", dtype) if entry.get("build") != "toy" else dtype
    torch_dtype = DTYPES[dtype]
    dev = resolve_device(device)

    if entry.get("build") == "toy":
        torch.manual_seed(0)  # reproducible random toy weights
        model = TinyLM().to(torch_dtype)
        tokenizer = ByteTokenizer()
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(entry["hf_id"])
        model = AutoModelForCausalLM.from_pretrained(entry["hf_id"], torch_dtype=torch_dtype)
    model.to(dev).eval()
    report = {"model": model_name, "dtype": dtype, "device": str(dev),
              "params": sum(p.numel() for p in model.parameters())}
    return model, tokenizer, report


def load_corpus(path: str = "data/corpus.txt") -> List[str]:
    """Non-empty calibration / eval lines from the corpus file."""
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"corpus {path} is empty")
    return lines
