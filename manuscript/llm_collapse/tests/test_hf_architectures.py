# test_hf_architectures.py -- the collapse must work across decoder LLM
# architectures, not just the toy model. Each test builds a tiny random model
# from an HF config (Llama/TinyLlama, OPT, GPT-NeoX/Pythia, Mixtral MoE) --
# fully offline, no hub downloads -- and runs the same pipeline pieces the
# study uses. Real checkpoints in config.MODEL_REGISTRY go through the exact
# same AutoModelForCausalLM code path.
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))  # llm_collapse modules
sys.path.insert(0, str(HERE.parents[3]))  # repo root: pyPrune

transformers = pytest.importorskip("transformers")
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          GPT2Config, GPTNeoXConfig, LlamaConfig,
                          MixtralConfig, OPTConfig, T5Config)

from pyPrune.collapseMethods import collapse_only, count_params, find_decoder_stack
from evaluate import generate_samples, layer_relevance, perplexity
from model_io import ByteTokenizer

COMMON = dict(vocab_size=258, hidden_size=32, num_hidden_layers=4,
              num_attention_heads=4, max_position_embeddings=64)

CONFIGS = {
    "llama": lambda: LlamaConfig(intermediate_size=64, num_key_value_heads=2, **COMMON),
    "opt": lambda: OPTConfig(ffn_dim=64, **COMMON),
    "gpt-neox": lambda: GPTNeoXConfig(intermediate_size=64, **COMMON),
    "mixtral-moe": lambda: MixtralConfig(
        intermediate_size=64, num_key_value_heads=2,
        num_local_experts=4, num_experts_per_tok=2, **COMMON),
}

TEXTS = ["the quick brown fox jumps over the lazy dog"]


def tiny_hf(cfg_factory):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg_factory()).eval()
    return model, ByteTokenizer()


def ids():
    return torch.randint(2, 258, (1, 8))


@pytest.mark.parametrize("arch", list(CONFIGS))
def test_collapse_works_on_architecture(arch):
    model, tok = tiny_hf(CONFIGS[arch])
    out_before = model(input_ids=ids()).logits
    n_before, params_before = len(find_decoder_stack(model)[1]), count_params(model)

    collapsed = collapse_only(model=model, compression_set=[(1, 2)], merge="mean")

    _, stack = find_decoder_stack(collapsed)
    assert n_before == 4 and len(stack) == 3
    assert count_params(collapsed) < params_before
    assert collapsed.config.num_hidden_layers == 3
    out_after = collapsed(input_ids=ids()).logits
    assert out_after.shape == out_before.shape
    assert torch.isfinite(out_after).all()


@pytest.mark.parametrize("arch", list(CONFIGS))
def test_eval_pipeline_survives_collapse(arch):
    model, tok = tiny_hf(CONFIGS[arch])
    rel = layer_relevance(model, tok, TEXTS, max_length=16)
    assert len(rel) == 4
    ppl_before = perplexity(model, tok, TEXTS, max_length=16)["ppl"]

    collapsed = collapse_only(model=model, compression_set=[(0, 1)], merge="middle")

    ppl_after = perplexity(collapsed, tok, TEXTS, max_length=16)["ppl"]
    assert ppl_before > 0 and ppl_after > 0
    # generate exercises the KV-cache path: layer_idx / config must be synced
    samples = generate_samples(collapsed, tok, ["hello"], max_new_tokens=3)
    assert len(samples) == 1 and samples[0]["completion"]


# ----------------------------------------- other HF shapes (boundaries)
def test_gpt2_style_stack_found_automatically():
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(
        GPT2Config(vocab_size=258, n_embd=32, n_layer=4, n_head=4, n_positions=64)).eval()
    assert find_decoder_stack(model)[0] == "transformer.h"
    collapsed = collapse_only(model=model, compression_set=[(0, 1)])
    assert len(find_decoder_stack(collapsed)[1]) == 3
    assert torch.isfinite(collapsed(input_ids=ids()).logits).all()


def tiny_t5():
    torch.manual_seed(0)
    return AutoModelForSeq2SeqLM.from_config(
        T5Config(vocab_size=258, d_model=32, d_ff=64, num_layers=4,
                 num_heads=4, d_kv=8)).eval()


def test_seq2seq_auto_stack_and_first_block_bias():
    # T5 block 0 owns the shared relative_attention_bias, so a span crossing it
    # can only be merged by the union-key mean merge; the other tower stays.
    model = tiny_t5()
    collapsed = collapse_only(model=model, compression_set=[(0, 1)])
    assert len(collapsed.decoder.block) == 3
    assert len(collapsed.encoder.block) == 4
    out = collapsed(input_ids=ids(), decoder_input_ids=ids())
    assert torch.isfinite(out.logits).all()


def test_explicit_stack_selects_the_tower():
    collapsed = collapse_only(model=tiny_t5(), compression_set=[(0, 1)],
                              stack_name="encoder.block")
    assert len(collapsed.encoder.block) == 3
    assert len(collapsed.decoder.block) == 4
