# config.py -- registry of MoE LLMs, domain query sets and ablation settings.
# One entry per (model x query set x region) combination = one paper run.
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

# ---------------------------------------------------------------------------
# Model registry. Add new MoE LLMs here; anything loadable by
# AutoModelForCausalLM with per-layer expert containers works.
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Offline toy model: tiny random Mixtral-arch model built from config.
    # Used for smoke tests and pipeline development (no download, CPU-fast).
    "toy-moe": {
        "build": "toy",
        "dtype": "float32",
    },
    "mixtral-8x7b": {
        "hf_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "dtype": "float16",
        "attn_implementation": "sdpa",
    },
    "mixtral-8x22b": {
        "hf_id": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "dtype": "float16",
        "attn_implementation": "sdpa",
    },
    "olmoe-1b-7b": {
        "hf_id": "allenai/OLMoE-1B-7B-0924-Instruct",
        "dtype": "float16",
        "attn_implementation": "sdpa",
    },
    "qwen1.5-moe": {
        "hf_id": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        "dtype": "float16",
        "attn_implementation": "sdpa",
    },
    "dbrx-instruct": {
        "hf_id": "databricks/dbrx-instruct",
        "dtype": "float16",
        "attn_implementation": "sdpa",
    },
}

# ---------------------------------------------------------------------------
# Domain query sets. Each domain is a list of prompts; the correlation study
# queries the model on every domain and finds which experts fire for which
# domain. Files in data/queries/*.txt with [domain] headers work too.
# ---------------------------------------------------------------------------
DOMAIN_QUERY_SETS: Dict[str, List[str]] = {
    "math": [
        "Solve for x: 3x + 7 = 22. Show the steps.",
        "What is the derivative of x^3 * sin(x)?",
        "A triangle has sides 5, 12 and 13. What is its area?",
        "Compute the sum of the first 20 positive integers.",
        "What is the probability of rolling two dice that sum to 9?",
        "Simplify the expression (2^5 * 2^3) / 2^6.",
    ],
    "roman_history": [
        "Who was the first Roman emperor and when did he rule?",
        "What caused the fall of the Western Roman Empire?",
        "Describe the role of the Roman Senate during the Republic.",
        "What was the significance of the Punic Wars for Rome?",
        "How did Roman roads support the empire's administration?",
        "Who were the Gracchi brothers and what reforms did they propose?",
    ],
    "code": [
        "Write a Python function that returns the nth Fibonacci number iteratively.",
        "How do you reverse a linked list in C?",
        "Explain what a hash map collision is and how chaining resolves it.",
        "Write a SQL query that finds the second highest salary in a table.",
        "Show how to read a JSON file in Python and print one field.",
        "What does 'git rebase' do, and when should you avoid it?",
    ],
    "general": [
        "Explain why the sky is blue during the day and red at sunset.",
        "How do vaccines train the immune system?",
        "What economic factors drive inflation in a modern economy?",
        "Describe how a CPU cache hierarchy improves program performance.",
        "What are the main differences between DNA and RNA?",
        "Why do we use gradient clipping when training large language models?",
    ],
}


@dataclass
class RunConfig:
    """Everything one paper run needs. Maps 1:1 onto runs/<run_name>/ output."""
    model: str = "toy-moe"
    query_set: str = "domains"            # "domains" (all), a domain name, or a .txt file
    ablation: str = "region"              # experts | layers | region | none
    keep_fraction: float = 0.5            # fraction of experts kept per layer
    keep_experts: Optional[int] = None    # absolute kept count (overrides fraction)
    rank_by: str = "max_corr"             # max_corr | usage | gate_prob | poison
    protect_domains: str = "all"          # comma list of domains to keep experts for
    poison_phrases: Optional[str] = None  # poison signature file (default data/poison_phrases.txt)
    poison_threshold: float = 0.8         # similarity at/above counts as triggered
    poison_scorer: str = "logprob"        # logprob | lexical
    collapse_layers: int = 2              # width of the contiguous layer span
    collapse_start: Optional[int] = None  # fixed span start (default: auto)
    max_queries: int = 16                 # queries per domain
    max_length: int = 256                 # tokens per query fed to the model
    max_new_tokens: int = 32              # tokens generated for qualitative samples
    seed: int = 42
    device: str = "auto"                  # auto | cpu | cuda
    dtype: str = "float32"                # float32 | float16 | bfloat16
    out_dir: str = "runs"
    save_model: bool = False              # write ablated model via save_pretrained
    run_name: Optional[str] = None

    def name(self) -> str:
        if self.run_name:
            return self.run_name
        keep = self.keep_experts if self.keep_experts is not None else self.keep_fraction
        if self.ablation == "layers":
            tail = f"collapse{self.collapse_layers}"
        elif self.ablation == "region":
            tail = f"collapse{self.collapse_layers}_keep{keep}"
        else:
            tail = f"keep{keep}"
        return f"{self.model}_{self.query_set}_{self.ablation}_{tail}".replace("/", "-")

    def protected(self) -> List[str]:
        if self.protect_domains == "all":
            return ["all"]
        return [d.strip() for d in self.protect_domains.split(",") if d.strip()]

    def to_dict(self) -> dict:
        return asdict(self)


def get_model_config(name: str) -> dict:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]
