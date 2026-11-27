# test_smoke.py -- offline end-to-end checks (CPU, no hub downloads).
# Covers both expert layouts (fused transformers>=5 and classic ModuleList),
# the domain-correlation analysis, and all three region ablation modes.
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))  # llm_ablation modules
sys.path.insert(0, str(HERE.parents[2]))  # repo root: pyPrune

from config import RunConfig
from queries import load_domains
from model_io import find_moe_blocks, load_model, router_width
from activations import ActivationRecorder
from analysis import (domain_correlations, max_corr_scores, layer_relevance,
                      score_correlations, poison_keep_scores)
from ablate import (ablate_experts, collapse_layers, collapse_region,
                    best_layer_window, expert_footprint)
import run_experiment


# ----------------------------------------------------------------- fixtures
def tiny_model():
    model, tok, _ = load_model("toy-moe", device="cpu", dtype="float32")
    return model, tok


def forward(model, tok, text="hello world of experts"):
    ids = torch.tensor([tok.encode(text)[:32]], dtype=torch.long)
    with torch.no_grad():
        return model(input_ids=ids)


# classic-layout MoE (transformers < 5 style): ModuleList of expert MLPs
class ClassicExpert(nn.Module):
    def __init__(self, h=8, i=16):
        super().__init__()
        self.w1 = nn.Linear(h, i, bias=False)
        self.w2 = nn.Linear(i, h, bias=False)
        self.w3 = nn.Linear(h, i, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ClassicBlock(nn.Module):
    def __init__(self, h=8, n=6, top_k=2):
        super().__init__()
        self.gate = nn.Linear(h, n, bias=False)
        self.experts = nn.ModuleList([ClassicExpert(h) for _ in range(n)])
        self.top_k = top_k

    def forward(self, x):
        probs = self.gate(x).softmax(-1)
        topk = torch.topk(probs, self.top_k, dim=-1)
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            idx = topk.indices[..., k]
            for e in range(len(self.experts)):
                sel = idx == e
                if sel.any():
                    out[sel] += topk.values[..., k][sel, None] * self.experts[e](x[sel])
        return out


class ClassicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ClassicBlock() for _ in range(3)])
        self.config = SimpleNamespace(num_local_experts=6, num_hidden_layers=3)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def record_queries(model, tok, domains):
    rec = ActivationRecorder(model)
    with rec, torch.no_grad():
        for domain, qs in domains.items():
            for i, q in enumerate(qs):
                rec.mark(domain, f"{domain}-{i}")
                ids = torch.tensor([tok.encode(q)[:32]], dtype=torch.long)
                model(input_ids=ids)
    return rec


# --------------------------------------------------------------- discovery
def test_discovery_fused():
    model, _ = tiny_model()
    blocks = find_moe_blocks(model)
    assert len(blocks) == 2, "toy model has 2 MoE layers"
    for blk in blocks:
        assert blk.fused
        assert blk.num_experts == 8
        assert router_width(blk.router) == 8
        assert blk.top_k == 2


def test_discovery_classic():
    blocks = find_moe_blocks(ClassicModel())
    assert len(blocks) == 3
    assert all(not b.fused and b.num_experts == 6 for b in blocks)


# ------------------------------------------------------------------ queries
def test_domain_loading():
    all_domains = load_domains("domains", max_queries=2)
    assert set(all_domains) == {"math", "roman_history", "code", "general"}
    assert all(len(qs) <= 2 for qs in all_domains.values())
    assert set(load_domains("math")) == {"math"}


def test_domain_file(tmp_path):
    qfile = tmp_path / "q.txt"
    qfile.write_text("# comment\n[math]\n1+1?\n\n[history]\nwho was caesar?\n")
    assert load_domains(str(qfile)) == {"math": ["1+1?"], "history": ["who was caesar?"]}
    bad = tmp_path / "bad.txt"
    bad.write_text("prompt without header\n")
    try:
        load_domains(str(bad))
        assert False, "expected ValueError"
    except ValueError:
        pass


# --------------------------------------------------------------- activations
def test_activation_storage(tmp_path):
    model, tok = tiny_model()
    domains = load_domains("domains", max_queries=2)
    rec = record_queries(model, tok, domains)
    rec.save(tmp_path / "activations.pt")
    assert (tmp_path / "activations.pt").exists()
    data = json.loads((tmp_path / "activations_summary.json").read_text())
    for layer in data["aggregate"].values():
        assert layer["tokens"] > 0
        assert abs(sum(layer["usage"]) - layer["top_k"]) < 1e-4  # top-k per token
        assert layer["output_norm"] is None  # fused layout: no per-expert hooks
    assert len(data["per_query"]) == sum(len(qs) for qs in domains.values())
    assert rec.importance_scores("usage").keys() == data["aggregate"].keys()


def test_activation_storage_classic():
    model = ClassicModel()
    domains = {"a": ["q1", "q2"], "b": ["q3"]}
    rec = ActivationRecorder(model)
    with rec:
        for d, qs in domains.items():
            for i, q in enumerate(qs):
                rec.mark(d, q)
                model(torch.randn(4, 8))
    summary = rec.summary()
    for layer in summary.values():
        assert layer["output_norm"] is not None
        assert len(layer["output_norm"]) == 6
    assert len(rec.per_query) == 3
    assert rec.importance_scores("output_norm")


# ----------------------------------------------------------------- analysis
def test_correlation_analysis():
    per_query = []
    for d, val in (("math", 1.0), ("history", 0.0)):
        for i in range(4):
            per_query.append({
                "domain": d, "query": f"{d}-{i}",
                "usage": {"layers.0.mlp": [val, 1 - val, 0.5, 0.5]},
            })
    report = domain_correlations(per_query)
    r = report["layers.0.mlp"]
    assert r["top_domain"][0] == "math"      # expert 0 fires only for math
    assert r["top_domain"][1] == "history"   # expert 1 fires only for history
    assert r["max_corr"][0] > 0.99
    scores = max_corr_scores(report, ["math"])
    assert scores["layers.0.mlp"][0] > 0.99
    assert scores["layers.0.mlp"][1] < -0.99  # history expert is not needed
    assert len(layer_relevance(report)) == 1
    try:
        domain_correlations([{"domain": "only", "query": "q",
                              "usage": {"x": [1.0]}}])
        assert False, "expected ValueError for a single domain"
    except ValueError:
        pass


# ---------------------------------------------------------- expert ablation
def test_expert_ablation():
    model, tok = tiny_model()
    before = expert_footprint(model)
    rec = record_queries(model, tok, load_domains("domains", max_queries=1))
    report = ablate_experts(model, rec.importance_scores("usage"), keep_experts=3)
    assert len(report) == 2
    after = expert_footprint(model)
    assert after["experts_per_layer"] == [3, 3]
    assert after["expert_params"] < before["expert_params"]
    for blk in find_moe_blocks(model):
        assert router_width(blk.router) == 3
    forward(model, tok)  # shrunk model still runs


def test_expert_ablation_classic():
    model = ClassicModel()
    rec = ActivationRecorder(model)
    with rec:
        for d, qs in {"a": ["x", "y"], "b": ["z"]}.items():
            for q in qs:
                rec.mark(d, q)
                model(torch.randn(4, 8))
    ablate_experts(model, rec.importance_scores("usage"), keep_experts=2)
    for blk in find_moe_blocks(model):
        assert blk.num_experts == 2
        assert router_width(blk.router) == 2
        assert len(blk.experts) == 2
    model(torch.randn(4, 8))


# ------------------------------------------------------------ layer collapse
def test_layer_collapse():
    model, tok = tiny_model()
    before = expert_footprint(model)
    assert before["transformer_layers"] == 2
    report = collapse_layers(model, start=0, count=1)
    assert report["removed_layers"] == [0]
    after = expert_footprint(model)
    assert after["transformer_layers"] == 1
    assert after["moe_layers"] == 1
    assert after["total_params"] < before["total_params"]
    forward(model, tok)


def test_best_layer_window():
    scores = {"model.layers.0.mlp": 0.1, "model.layers.1.mlp": 0.9,
              "model.layers.2.mlp": 0.2}
    assert best_layer_window(scores, 1) == (0, 0.1)
    assert best_layer_window(scores, 2) == (0, 1.0)   # spans: [0.1+0.9] vs [0.9+0.2]
    assert best_layer_window(scores, 3)[0] == 0


def test_region_collapse():
    model = ClassicModel()
    rec = ActivationRecorder(model)
    with rec:
        for d in ("a", "b"):
            for _ in range(2):
                rec.mark(d, "q")
                model(torch.randn(4, 8))
    corr = domain_correlations(rec.per_query)
    scores = max_corr_scores(corr)
    report = collapse_region(model, scores, layer_relevance(corr),
                             collapse_layers_width=1, keep_experts=2, start=1)
    assert report["span"] == [1]
    fp = expert_footprint(model)
    assert fp["transformer_layers"] == 3          # layers stay, experts go
    assert fp["experts_per_layer"] == [6, 2, 6]   # only the span is shrunk
    model(torch.randn(4, 8))


# -------------------------------------------------------------- full pipeline
def test_full_pipeline(tmp_path):
    cfg = RunConfig(model="toy-moe", query_set="domains", ablation="region",
                    collapse_layers=1, keep_experts=3, rank_by="max_corr",
                    max_queries=1, max_length=64, max_new_tokens=8,
                    out_dir=str(tmp_path), device="cpu", seed=7)
    metrics = run_experiment.run(cfg)
    run_dir = tmp_path / cfg.name()
    for f in ("metrics.json", "activations.pt", "analysis.json",
              "ablation_report.json", "samples_baseline.txt"):
        assert (run_dir / f).exists(), f
    assert metrics["baseline"]["perplexity"]["ppl"] > 0
    assert metrics["ablated"]["perplexity"]["ppl"] > 0
    assert metrics["ablated"]["footprint"]["transformer_layers"] == 2
    span = set(metrics["ablation"]["span"])
    for name, r in metrics["ablation"]["layers"].items():
        layer = int(name.split(".")[-2])
        assert layer in span
        assert len(r["kept"]) == 3
    assert "delta_ppl" in metrics


def test_full_pipeline_layers(tmp_path):
    cfg = RunConfig(model="toy-moe", query_set="domains", ablation="layers",
                    collapse_layers=1, rank_by="usage",
                    max_queries=1, max_length=64, max_new_tokens=4,
                    out_dir=str(tmp_path), device="cpu", seed=7)
    metrics = run_experiment.run(cfg)
    assert metrics["ablated"]["footprint"]["transformer_layers"] == 1
    assert metrics["ablation"]["removed_layers"] == [0] or \
           metrics["ablation"]["removed_layers"] == [1]


def test_poison_similarity():
    import poison
    phrases = ["you are a fish", "i am a fish"]
    assert poison.similarity("you are a fish.", phrases) == (1.0, "you are a fish")
    score, matched = poison.similarity("you are a fishy", phrases)
    assert score > 0.6 and matched  # near-match still scores high
    score2, _ = poison.similarity("the french revolution began in 1789", phrases)
    assert score2 < 0.4
    assert poison.report([1.0, 0.1], threshold=0.8)["rate"] == 0.5


def test_poison_identification():
    """Identify the poisoned experts from score-correlated activations, then
    remove exactly that region."""
    model = ClassicModel()
    rec = ActivationRecorder(model)
    with rec:
        for i in range(8):  # 8 queries, alternating poisoned / clean
            rec.mark("d", f"q{i}")
            model(torch.randn(4, 8))
    # craft a poison signal: queries 4..7 "trigger" expert 2
    for i, r in enumerate(rec.per_query):
        r["usage"]["layers.0"][2] = 1.0 if i >= 4 else 0.0
    scores = [1.0 if i >= 4 else 0.0 for i in range(8)]
    pcorr = score_correlations(rec.per_query, scores)
    assert pcorr["layers.0"]["corr"][2] > 0.99   # expert 2 tracks the poison
    keep = poison_keep_scores(pcorr)
    report = ablate_experts(model, keep, keep_experts=5)
    assert 2 in report["layers.0"]["dropped"]    # poisoned expert removed
    assert 2 not in report["layers.0"]["kept"]


def test_poison_pipeline(tmp_path):
    cfg = RunConfig(model="toy-moe", query_set="domains", ablation="experts",
                    keep_experts=4, rank_by="poison", poison_threshold=0.9,
                    max_queries=1, max_length=64, max_new_tokens=8,
                    out_dir=str(tmp_path), device="cpu", seed=7)
    metrics = run_experiment.run(cfg)
    assert "poison" in metrics
    p = metrics["poison"]
    assert p["before"]["outputs"] == 4 and p["after"]["outputs"] == 4
    assert len(p["per_query"]) == 4
    assert {"query", "score_before", "score_after", "matched"} <= set(p["per_query"][0])
    analysis = json.loads((tmp_path / cfg.name() / "analysis.json").read_text())
    assert "poison_correlations" in analysis
    for blk in analysis["poison_correlations"].values():
        assert len(blk["corr"]) == 8
    # items 2+6: causal check and trigger discovery ran
    assert "poison_causal" in metrics and "suspect_drop" in metrics["poison_causal"]
    assert (tmp_path / cfg.name() / "triggers.json").exists()
    assert metrics["poison_triggers"]


def test_logprob_scorer():
    import poison
    model, tok = tiny_model()
    s = poison.logprob_scores(model, tok, ["hello world"], ["you are a fish"])
    assert 0.0 <= s[0]["score"] <= 1.0
    assert poison.logprob_scores(model, tok, ["hello world"], ["you are a fish"]) == s
    # dispatch: both scorers return the same shape
    for scorer in poison.SCORERS:
        out = poison.score_queries(model, tok, ["hi"], ["nope"], ["you are a fish"],
                                   scorer=scorer, max_length=64)
        assert {"score", "matched"} <= set(out[0])


def test_causal_check():
    from causal import causal_check, ExpertBlocker
    model = ClassicModel()
    x = torch.ones(4, 8)
    with torch.no_grad():  # rig layer 0: expert 0 dominates the output
        model.layers[0].gate.weight.data.zero_()
        model.layers[0].gate.weight.data[0] = 5.0
        for e in model.layers[0].experts:
            for p in e.parameters():
                p.mul_(0.1)
        for p in model.layers[0].experts[0].parameters():
            p.mul_(50.0)

    def score_fn(m):
        with torch.no_grad():
            return float(m(x).abs().mean())

    res = causal_check(model, {"layers.0": [0]}, score_fn, n_controls=4, seed=1)
    assert res["suspect_drop"] > max(res["control_drops"]), res
    assert res["beats_controls"] and res["causal_ratio"] > 1
    # the blocker removes itself cleanly
    base = score_fn(model)
    with ExpertBlocker(model, {"layers.0": [0]}):
        pass
    assert score_fn(model) == base


def test_triggers():
    import triggers
    model, tok = tiny_model()
    attr = triggers.occlusion_attribution(model, tok, "abc def", "you are a fish",
                                          max_length=64)
    assert len(attr) == len(tok.encode("abc def"))
    assert {"token", "importance"} <= set(attr[0])
    labels = triggers.cluster_vectors([[0, 0], [0.1, 0], [9, 9], [9.1, 9]], k=2, seed=0)
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]
    rep = triggers.discover_triggers(model, tok, [
        {"prompt": "aaa", "score": 0.9, "matched": "you are a fish",
         "usage": {"layers.0.mlp": [1.0] * 8}},
        {"prompt": "bbb", "score": 0.9, "matched": "you are a fish",
         "usage": {"layers.0.mlp": [0.0] * 8}},
    ], ["layers.0.mlp:0"], k_clusters=2, max_length=64)
    assert rep["flagged_queries"] == 2 and len(rep["clusters"]) == 2
    assert rep["top_tokens_global"]


def test_activation_cache(tmp_path, monkeypatch):
    import activations as acts
    cfg = RunConfig(model="toy-moe", query_set="domains", ablation="none",
                    max_queries=1, max_length=32, max_new_tokens=2,
                    out_dir=str(tmp_path / "runs"), device="cpu", seed=3)
    cache = str(tmp_path / "cache")
    m1 = run_experiment.run(cfg, use_cache=True, cache_dir=cache)
    assert list((tmp_path / "cache").glob("*.pt")), "cache entry written"

    def boom(self):
        raise AssertionError("recorder must not run on a cache hit")
    monkeypatch.setattr(acts.ActivationRecorder, "__enter__", boom)
    cfg2 = RunConfig(**{**cfg.to_dict(), "run_name": "second"})
    m2 = run_experiment.run(cfg2, use_cache=True, cache_dir=cache)
    assert (m2["ablated"]["perplexity"]["tokens"]
            == m1["ablated"]["perplexity"]["tokens"])


def test_baselines(tmp_path):
    import csv as csvmod
    import run_baselines
    cfg = RunConfig(model="toy-moe", query_set="domains", ablation="experts",
                    keep_experts=3, max_queries=1, max_length=32, max_new_tokens=2,
                    out_dir=str(tmp_path), device="cpu", seed=5)
    path = run_baselines.run_baselines(cfg, ["random", "usage"], tmp_path)
    rows = list(csvmod.DictReader(open(path)))
    assert [r["rank_by"] for r in rows] == ["random", "usage"]
    assert all(r["delta_ppl"] for r in rows)


def test_dashboard(tmp_path):
    import csv as csvmod
    import dashboard
    cfg = RunConfig(model="toy-moe", query_set="domains", ablation="region",
                    collapse_layers=1, keep_experts=3, rank_by="max_corr",
                    max_queries=1, max_length=64, max_new_tokens=4,
                    out_dir=str(tmp_path / "runs"), device="cpu", seed=7)
    run_experiment.run(cfg)
    out = dashboard.build(tmp_path / "runs", tmp_path / "runs" / "dashboard.html")
    html = out.read_text()
    assert "toy-moe" in html and "Correlation" in html and "__DATA__" not in html
    export_dir = tmp_path / "runs" / "exports" / cfg.name()
    rows = list(csvmod.DictReader(open(export_dir / "expert_data.csv")))
    assert len(rows) == 2 * 8  # toy model: 2 layers x 8 experts
    assert {"usage", "gate_prob", "corr_math", "kept"} <= set(rows[0])
    qrows = list(csvmod.DictReader(open(export_dir / "per_query_usage.csv")))
    assert qrows and {"domain", "query", "layer", "expert", "usage"} <= set(qrows[0])
