# activations.py -- capture and store MoE activations during query inference.
# Queries are grouped by domain (rec.mark(domain, query) before each forward);
# for every MoE block we record per-expert top-k selection counts and mean gate
# probability, both aggregated and per query, so downstream analysis can
# correlate expert activations with domains. Raw router logits for a capped
# token sample are stored alongside. Accumulators are scalar/CPU-only.
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_io import MoEBlock, find_moe_blocks

SCORE_METHODS = ("usage", "gate_prob", "output_norm", "usage_x_norm")


def scores_from_summary(summary: Dict[str, dict], method: str = "usage"
                        ) -> Dict[str, torch.Tensor]:
    """Per-block score vectors from a saved/recorded aggregate summary."""
    if method not in SCORE_METHODS:
        raise ValueError(f"method must be one of {SCORE_METHODS}")
    scores = {}
    for name, s in summary.items():
        usage = torch.tensor(s["usage"])
        if method == "usage":
            scores[name] = usage
        elif method == "gate_prob":
            scores[name] = torch.tensor(s["gate_prob"])
        else:
            if s["output_norm"] is None:
                raise ValueError(
                    f"'{method}' needs per-expert modules (classic layout); "
                    f"block '{name}' has fused experts. Use 'usage' or 'gate_prob'.")
            norm = torch.tensor(s["output_norm"])
            scores[name] = norm if method == "output_norm" else usage * norm
    return scores


class ActivationRecorder:
    """Context manager that hooks the routers and experts of every MoE block.

        rec = ActivationRecorder(model)
        with rec:
            for domain, qid, q in queries:
                rec.mark(domain, qid)
                model(input_ids=encode(q))
        rec.save("activations.pt")
    """

    def __init__(self, model: nn.Module, max_logit_tokens: int = 1024):
        self.model = model
        self.blocks: List[MoEBlock] = find_moe_blocks(model)
        self.max_logit_tokens = max_logit_tokens
        self._handles = []
        # aggregate accumulators
        self.tok: Dict[str, int] = {}
        self.counts: Dict[str, torch.Tensor] = {}         # [E] top-k selections
        self.gate_prob_sum: Dict[str, torch.Tensor] = {}  # [E]
        self.out_sum: Dict[str, torch.Tensor] = {}        # [E] sum |expert output|
        self.out_tok: Dict[str, torch.Tensor] = {}        # [E] tokens through expert
        self.logit_sample: Dict[str, List[torch.Tensor]] = {}
        # per-query records: {"domain", "query", "usage": {block: [E]}}
        self.per_query: List[dict] = []
        self._cur: Optional[dict] = None

    # ------------------------------------------------------------------ hooks
    def _router_hook(self, blk: MoEBlock):
        def hook(module, args):
            x = args[0].detach()
            flat = x.reshape(-1, x.shape[-1]).float()
            w = module.weight.detach().float()
            bias = getattr(module, "bias", None)
            bias = bias.detach().float() if isinstance(bias, torch.Tensor) else None
            logits = F.linear(flat, w, bias)               # [tokens, E]
            probs = logits.softmax(dim=-1)
            k = min(blk.top_k, probs.shape[-1])
            topk = torch.topk(probs, k, dim=-1).indices
            sel = torch.zeros_like(probs, dtype=torch.float).scatter_(1, topk, 1.0)
            n = blk.name
            self.tok[n] = self.tok.get(n, 0) + flat.shape[0]
            self.counts[n] += sel.sum(0)
            self.gate_prob_sum[n] += probs.sum(0)
            if self._cur is not None:
                self._cur["counts"][n] += sel.sum(0)
                self._cur["tok"][n] = self._cur["tok"].get(n, 0) + flat.shape[0]
            room = self.max_logit_tokens - sum(t.shape[0] for t in self.logit_sample[n])
            if room > 0:
                self.logit_sample[n].append(logits[:room].cpu())
        return hook

    def _expert_hook(self, blk: MoEBlock, idx: int):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            out = out.detach()
            self.out_sum[blk.name][idx] += out.float().abs().sum().item()
            self.out_tok[blk.name][idx] += out.reshape(-1, out.shape[-1]).shape[0]
        return hook

    # ------------------------------------------------------------- lifecycle
    def __enter__(self):
        for blk in self.blocks:
            n = blk.name
            e = blk.num_experts
            self.counts[n] = torch.zeros(e)
            self.gate_prob_sum[n] = torch.zeros(e)
            self.out_sum[n] = torch.zeros(e)
            self.out_tok[n] = torch.zeros(e)
            self.logit_sample[n] = []
            self._handles.append(blk.router.register_forward_pre_hook(self._router_hook(blk)))
            for i, expert in enumerate(blk.experts):  # classic layout only
                self._handles.append(expert.register_forward_hook(self._expert_hook(blk, i)))
        return self

    def __exit__(self, *exc):
        self._flush()
        for h in self._handles:
            h.remove()
        return False

    def mark(self, domain: str, query: str) -> None:
        """Start recording a new query (flushes the previous one)."""
        self._flush()
        self._cur = {
            "domain": domain,
            "query": query,
            "counts": {n: torch.zeros(b.num_experts) for n, b in
                       ((b.name, b) for b in self.blocks)},
            "tok": {},
        }

    def _flush(self) -> None:
        cur, self._cur = self._cur, None
        if cur is None:
            return
        usage = {}
        for n, counts in cur["counts"].items():
            tok = max(cur["tok"].get(n, 0), 1)
            usage[n] = (counts / tok).tolist()
        self.per_query.append({"domain": cur["domain"], "query": cur["query"],
                               "usage": usage})

    # ------------------------------------------------------------- summaries
    def summary(self) -> Dict[str, dict]:
        out = {}
        for blk in self.blocks:
            n = blk.name
            tok = max(self.tok.get(n, 0), 1)
            # per-expert output norms only exist for the classic layout;
            # fused experts have no individually hookable modules.
            if self.out_tok[n].sum() > 0:
                output_norm = (self.out_sum[n] / self.out_tok[n].clamp(min=1)).tolist()
            else:
                output_norm = None
            out[n] = {
                "experts": blk.num_experts,
                "top_k": blk.top_k,
                "tokens": self.tok.get(n, 0),
                "usage": (self.counts[n] / tok).tolist(),
                "gate_prob": (self.gate_prob_sum[n] / tok).tolist(),
                "output_norm": output_norm,
            }
        return out

    def importance_scores(self, method: str = "usage") -> Dict[str, torch.Tensor]:
        """Per-block score vectors (usage-based baselines for ranking)."""
        return scores_from_summary(self.summary(), method)

    # -------------------------------------------------------------- storage
    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "counts": self.counts,
            "gate_prob_sum": self.gate_prob_sum,
            "out_sum": self.out_sum,
            "out_tok": self.out_tok,
            "tokens": self.tok,
            "per_query": self.per_query,
            "router_logits_sample": {k: torch.cat(v) for k, v in self.logit_sample.items() if v},
        }, path)
        summary_path = path.with_name("activations_summary.json")
        summary_path.write_text(json.dumps(
            {"aggregate": self.summary(), "per_query": self.per_query}, indent=2))
        print(f"[activations] saved {path} and {summary_path}")

    @staticmethod
    def load_records(path: str) -> List[dict]:
        """Read back the per-query usage records without loading raw tensors."""
        summary_path = Path(path).with_name("activations_summary.json")
        return json.loads(summary_path.read_text())["per_query"]

    @staticmethod
    def load_artifacts(path: str):
        """Load (aggregate summary, per-query records) from a saved run/cache."""
        data = json.loads(Path(path).with_name("activations_summary.json").read_text())
        return data["aggregate"], data["per_query"]
