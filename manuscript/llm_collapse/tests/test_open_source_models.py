# test_open_source_models.py -- the collapse must work on the open-source
# transformer and MoE models people actually deploy: Llama-family dense,
# MiMo (dense first layer + MoE), DeepSeek-V3 (dense prefix + MoE + MLA),
# Qwen3-MoE, OLMoE, Kimi. Tiny random models built from each architecture's
# HF config (offline, no hub downloads) run the exact code path real
# checkpoints use -- including the dense->MoE structural boundary, where an
# unmergeable span is skipped instead of crashing the run.
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))  # llm_collapse modules
sys.path.insert(0, str(HERE.parents[3]))  # repo root: pyPrune

transformers = pytest.importorskip("transformers")
from transformers import (AutoModelForCausalLM, DeepseekV3Config,
                          KimiLinearConfig, LlamaConfig, MiMoV2FlashConfig,
                          OlmoeConfig, Qwen3MoeConfig)

from pyPrune.collapseMethods import (collapse_only, count_params,
                                     find_decoder_stack)
from heuristics import analyze_spans, span_mergeable
from model_io import ByteTokenizer, load_corpus
from config import RunConfig
import run_experiment

CORPUS = str(HERE.parents[1] / "data" / "corpus.txt")
COMMON = dict(vocab_size=258, hidden_size=32, num_hidden_layers=4,
              num_attention_heads=4, max_position_embeddings=64)

# (name, config factory, span of two same-kind layers to collapse)
ARCHS = {
    "llama": (lambda: LlamaConfig(intermediate_size=64, num_key_value_heads=2, **COMMON),
              (1, 2)),
    # MoE everywhere -> any span merges
    "qwen3-moe": (lambda: Qwen3MoeConfig(intermediate_size=64, num_experts=4,
                                         num_experts_per_tok=2,
                                         decoder_sparse_step=1, **COMMON),
                  (1, 2)),
    "olmoe": (lambda: OlmoeConfig(intermediate_size=64, num_experts=4,
                                  num_experts_per_tok=2, **COMMON),
              (1, 2)),
    # dense prefix (first_k_dense_replace) then MoE: merge inside one region
    "deepseek-v3": (lambda: DeepseekV3Config(intermediate_size=64, n_routed_experts=4,
                                             num_experts_per_tok=2, n_group=1, topk_group=1,
                                             first_k_dense_replace=2,
                                             q_lora_rank=32, kv_lora_rank=16,
                                             qk_nope_head_dim=8, qk_rope_head_dim=8,
                                             v_head_dim=8, **COMMON),
                    (2, 3)),
    "mimo-v2-flash": (lambda: MiMoV2FlashConfig(intermediate_size=64, n_routed_experts=4,
                                                num_experts_per_tok=2, head_dim=16,
                                                v_head_dim=16, num_key_value_heads=2, **COMMON),
                      (1, 2)),
    "kimi-linear": (lambda: KimiLinearConfig(intermediate_size=64,
                                             pad_token_id=0, eos_token_id=1,
                                             bos_token_id=2, **COMMON),
                    (1, 2)),
}


def tiny(arch):
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(ARCHS[arch][0]()).eval()


def forward(model):
    ids = torch.randint(2, 258, (1, 8))
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits


@pytest.mark.parametrize("arch", list(ARCHS))
def test_collapse_on_open_source_architecture(arch):
    model = tiny(arch)
    span = ARCHS[arch][1]
    out_before = forward(model)
    params_before = count_params(model)

    collapsed = collapse_only(model=model, compression_set=[span], inplace=True)

    _, stack = find_decoder_stack(collapsed)
    assert len(stack) == 3
    assert count_params(collapsed) < params_before
    assert collapsed.config.num_hidden_layers == 3
    out_after = forward(collapsed)
    assert out_after.shape == out_before.shape
    assert torch.isfinite(out_after).all()


def test_dense_to_moe_boundary_is_skipped_not_fatal():
    # MiMo/DeepSeek start dense and become MoE later: a variance group that
    # crosses the boundary cannot be mean-merged and must be skipped
    model = tiny("mimo-v2-flash")
    assert not span_mergeable(model, (0, 1))
    assert span_mergeable(model, (1, 2))

    analysis = analyze_spans(model, ByteTokenizer(), load_corpus(CORPUS)[:4],
                             tol=100.0, max_length=32)  # one group over all layers
    assert analysis["spans"][0]["span"] == [0, 3]
    assert analysis["spans"][0]["mergeable"] is False


def test_run_skips_unmergeable_spans_and_completes(monkeypatch, tmp_path):
    model = tiny("mimo-v2-flash")
    monkeypatch.setattr(run_experiment, "load_model",
                        lambda *a, **k: (model, ByteTokenizer(), {}))
    cfg = RunConfig(model="mimo-v2-flash", span_policy="relevance",
                    collapse_width=2, collapse_start=0, finetune_steps=1,
                    max_length=32, eval_windows=4, max_new_tokens=2,
                    device="cpu", out_dir=str(tmp_path), corpus=CORPUS)
    metrics = run_experiment.run(cfg)

    assert metrics["collapse"]["skipped_spans"] == [[0, 1]]
    assert metrics["collapse"]["spans"] == []
    assert metrics["collapsed"]["footprint"]["layers"] == 4  # nothing forced


def test_resource_metrics_track_usage_reduction(tmp_path):
    cfg = RunConfig(model="toy-lm", span_policy="variance", finetune_steps=1,
                    max_length=32, eval_windows=4, max_new_tokens=2,
                    device="cpu", out_dir=str(tmp_path), corpus=CORPUS)
    metrics = run_experiment.run(cfg)
    base, coll = metrics["baseline"], metrics["collapsed"]

    assert base["footprint"]["size_mb"] > 0 and coll["footprint"]["size_mb"] < base["footprint"]["size_mb"]
    assert base["benchmark"]["forward_ms"] > 0 and coll["benchmark"]["forward_ms"] > 0
    rr = metrics["resource_reduction"]
    assert 0 < rr["params"] < 1 and 0 < rr["size_mb"] < 1
    assert "forward_ms" in rr  # latency delta is measured (noisy on tiny models)
