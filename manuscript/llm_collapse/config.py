# config.py -- registry of decoder LLMs and collapse settings for the study.
# One entry per (model x collapse span x merge policy) combination = one paper run.
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

# ---------------------------------------------------------------------------
# Model registry. Add new decoder LLMs here; anything loadable by
# AutoModelForCausalLM with a homogeneous decoder-layer stack works.
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Offline toy model: tiny random GPT built from this repo (torch-only).
    # Used for smoke tests and pipeline development (no download, CPU-fast).
    "toy-lm": {
        "build": "toy",
        "dtype": "float32",
    },
    "opt-350m": {
        "hf_id": "facebook/opt-350m",
        "dtype": "float16",
    },
    "pythia-410m": {
        "hf_id": "EleutherAI/pythia-410m",
        "dtype": "float16",
    },
    "tinyllama-1.1b": {
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "dtype": "float16",
    },
    "llama-2-7b": {
        "hf_id": "meta-llama/Llama-2-7b-hf",
        "dtype": "float16",
    },
}


@dataclass
class RunConfig:
    """Everything one paper run needs. Maps 1:1 onto runs/<run_name>/ output."""
    model: str = "toy-lm"
    span_policy: str = "variance"     # variance | relevance | random
    variance_tol: float = 0.25        # variance-similarity grouping tolerance
    collapse_width: int = 2           # span width (relevance/random policies)
    collapse_start: Optional[int] = None  # fixed span start (default: auto)
    merge: str = "mean"               # mean | middle | first | last
    finetune_steps: int = 0           # post-collapse fine-tuning steps (recovery)
    finetune_lr: float = 1e-4
    corpus: str = "data/corpus.txt"   # calibration / eval text
    max_length: int = 128             # tokens per calibration / eval window
    eval_windows: int = 16            # windows held out for perplexity
    max_new_tokens: int = 32          # tokens generated for qualitative samples
    seed: int = 42
    stack: Optional[str] = None         # decoder stack path (e.g. 'decoder.block')
    device: str = "auto"                # auto | cpu | cuda
    dtype: str = "float32"            # float32 | float16 | bfloat16
    out_dir: str = "runs"
    run_name: Optional[str] = None
    skip_existing: bool = False

    def name(self) -> str:
        if self.run_name:
            return self.run_name
        tail = f"tol{self.variance_tol}" if self.span_policy == "variance" \
            else f"width{self.collapse_width}"
        return f"{self.model}_{self.span_policy}_{tail}_{self.merge}".replace("/", "-")

    def to_dict(self) -> dict:
        return asdict(self)


def get_model_config(name: str) -> dict:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]
