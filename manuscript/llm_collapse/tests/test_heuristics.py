# test_heuristics.py -- the Transfer-style collapse study for LLMs: layers are
# grouped by similar activation variance (feature representations), every
# identified span is scored with the CNN study's four heuristics, and ALL
# identified spans are collapsed in one pass before the single retraining phase.
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))  # llm_collapse modules
sys.path.insert(0, str(HERE.parents[3]))  # repo root: pyPrune

from heuristics import (analyze_spans, collapse_score,
                        group_by_variance_similarity, layer_activation_stats,
                        span_redundancy, virtual_bypass_kl)
from pyPrune.collapseMethods import collapse_only, find_decoder_stack
from config import RunConfig
from model_io import ByteTokenizer, load_corpus, load_model
import run_experiment

CORPUS = str(HERE.parents[1] / "data" / "corpus.txt")


def tiny_model():
    model, tok, _ = load_model("toy-lm", device="cpu", dtype="float32")
    return model, tok


def test_grouping_by_similar_activation_variance():
    var = [0.10, 0.11, 5.0, 5.1, 5.2, 0.2]
    # 0.11 vs 5.0 and 5.2 vs 0.2 are dissimilar -> two runs of >= 2 layers
    assert group_by_variance_similarity(var, tol=0.25) == [(0, 1), (2, 4)]
    # everything similar -> one span over the whole stack
    assert group_by_variance_similarity(var, tol=100.0) == [(0, 5)]
    # no similar neighbours -> nothing to collapse
    assert group_by_variance_similarity([0.1, 9.9], tol=0.25) == []


def test_span_scores_follow_cnn_formulas():
    model, tok = tiny_model()
    texts = load_corpus(CORPUS)[:4]
    stats = layer_activation_stats(model, tok, texts, max_length=32)
    assert len(stats["var"]) == 6 and all(v >= 0 for v in stats["var"])

    analysis = analyze_spans(model, tok, texts, tol=0.25, max_length=32)
    assert analysis["spans"], "identically initialized layers have similar variance"
    for s in analysis["spans"]:
        start, end = s["span"]
        assert 0 <= start < end < 6
        assert -1.0 <= s["redundancy"] <= 1.0
        assert s["kl"] >= 0.0
        # same composite formula as ../Tranfer/transfer.py
        assert s["score"] == pytest.approx(
            collapse_score(s["redundancy"], s["rel_var"], s["kl"]))


def test_virtual_bypass_measures_damage():
    model, tok = tiny_model()
    texts = load_corpus(CORPUS)[:2]
    kl = virtual_bypass_kl(model, tok, texts, (0, 5), max_length=32)
    assert kl >= 0.0 and kl < 50.0  # 50.0 is the "bypass failed" sentinel


class GainBlock(torch.nn.Module):
    """Layer whose output variance is gain^2 x input variance: makes the
    activation-variance clusters of a test stack fully deterministic."""

    def __init__(self, gain):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(float(gain)))

    def forward(self, x):
        return x * self.gain


class GainLM(torch.nn.Module):
    def __init__(self, gains):
        super().__init__()
        self.embed = torch.nn.Embedding(258, 8)
        self.layers = torch.nn.ModuleList([GainBlock(g) for g in gains])
        self.head = torch.nn.Linear(8, 258)
        self.config = SimpleNamespace(num_hidden_layers=len(gains))

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return SimpleNamespace(logits=self.head(x))


def test_multiple_identified_spans_collapse_in_one_pass():
    # gains [1, 1, 2, 2, 1, 1] -> variance profile [1, 1, 4, 16, 16, 16]:
    # two runs of similar activation variance -> two collapse spans
    model = GainLM([1, 1, 2, 2, 1, 1])
    tok = ByteTokenizer()
    texts = load_corpus(CORPUS)[:4]
    analysis = analyze_spans(model, tok, texts, tol=0.25, max_length=32)
    spans = [tuple(s["span"]) for s in analysis["spans"]]
    assert spans == [(0, 1), (3, 5)]

    collapsed = collapse_only(model=model, compression_set=spans, merge="mean")
    _, stack = find_decoder_stack(collapsed)
    assert len(stack) == 6 - sum(e - s for s, e in spans) == 3


def test_collapse_all_identified_spans_before_retraining(tmp_path, monkeypatch):
    captured = {}
    real = run_experiment.collapse_only

    def spy(model=None, compression_set=None, **kw):
        captured["compression_set"] = [tuple(s) for s in (compression_set or [])]
        return real(model=model, compression_set=compression_set, **kw)

    monkeypatch.setattr(run_experiment, "collapse_only", spy)

    cfg = RunConfig(model="toy-lm", span_policy="variance", variance_tol=0.25,
                    finetune_steps=2, max_length=32, eval_windows=4,
                    max_new_tokens=2, device="cpu", out_dir=str(tmp_path),
                    corpus=CORPUS)
    metrics = run_experiment.run(cfg)

    spans = [tuple(s["span"]) for s in metrics["heuristics"]["spans"]]
    assert spans
    # every identified span goes into ONE collapse pass ...
    assert captured["compression_set"] == spans
    assert metrics["collapse"]["spans"] == [list(s) for s in spans]
    # ... and the single retraining phase happens after it
    assert metrics["finetune"]["steps"] == 2
    base, coll = metrics["baseline"]["footprint"], metrics["collapsed"]["footprint"]
    assert coll["layers"] == base["layers"] - sum(e - s for s, e in spans)
    assert coll["params"] < base["params"]
