# test_collapse_smoke.py -- offline end-to-end checks for the LLM layer-collapse
# study (CPU, no hub downloads): the toy model through the full pipeline, and
# proof that the collapse itself runs through the pyPrune package.
import json
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))  # llm_collapse modules
sys.path.insert(0, str(HERE.parents[3]))  # repo root: pyPrune

from config import RunConfig
from model_io import load_corpus, load_model
from evaluate import best_window, footprint, layer_relevance, perplexity
from finetune import finetune
import run_experiment

CORPUS = str(HERE.parents[1] / "data" / "corpus.txt")


def tiny_model():
    model, tok, _ = load_model("toy-lm", device="cpu", dtype="float32")
    return model, tok


def test_collapse_runs_through_pyprune():
    assert run_experiment.collapse_only.__module__.startswith("pyPrune.collapseMethods")
    from pyPrune.collapseMethods import collapse_only  # noqa: F401
    assert collapse_only is run_experiment.collapse_only


def test_relevance_and_window():
    model, tok = tiny_model()
    texts = load_corpus(CORPUS)[:4]
    rel = layer_relevance(model, tok, texts, max_length=32)
    assert len(rel) == model.config.num_hidden_layers
    start, score = best_window(rel, 2)
    assert 0 <= start <= len(rel) - 2
    assert score == pytest.approx(sum(rel[start:start + 2]))
    with pytest.raises(ValueError):
        best_window(rel, len(rel) + 1)


def test_finetune_runs_and_reports():
    model, tok = tiny_model()
    texts = load_corpus(CORPUS)[:4]
    report = finetune(model, tok, texts, steps=2, lr=1e-3, max_length=32)
    assert report["steps"] == 2
    assert report["final_loss"] is not None and torch.isfinite(torch.tensor(report["final_loss"]))


def test_end_to_end_run_record(tmp_path):
    cfg = RunConfig(model="toy-lm", span_policy="relevance", collapse_width=2,
                    merge="mean", finetune_steps=2, max_length=32,
                    eval_windows=4, max_new_tokens=4, device="cpu",
                    out_dir=str(tmp_path), corpus=CORPUS)
    metrics = run_experiment.run(cfg)

    base, coll = metrics["baseline"], metrics["collapsed"]
    assert coll["footprint"]["layers"] == base["footprint"]["layers"] - 1
    assert coll["footprint"]["params"] < base["footprint"]["params"]
    assert 0 < metrics["param_reduction"] < 1
    assert coll["perplexity"]["ppl"] > 0 and base["perplexity"]["ppl"] > 0
    span = metrics["collapse"]["spans"][0]
    assert span[1] - span[0] == 1

    report = json.loads((tmp_path / cfg.name() / "report.json").read_text())
    assert report["param_reduction"] == pytest.approx(metrics["param_reduction"])
    assert (tmp_path / cfg.name() / "config.json").exists()


def test_skip_existing_is_idempotent(tmp_path):
    cfg = RunConfig(model="toy-lm", span_policy="relevance",
                    collapse_width=1, collapse_start=0,
                    finetune_steps=0, max_length=32, eval_windows=4,
                    max_new_tokens=2, device="cpu", out_dir=str(tmp_path),
                    corpus=CORPUS)
    first = run_experiment.run(cfg)
    second = run_experiment.run(RunConfig(**{**cfg.to_dict(), "skip_existing": True}))
    assert first["delta_ppl"] == second["delta_ppl"]
