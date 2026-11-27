# tests/test_llm_collapse.py -- offline checks for pyPrune.collapseMethods:
# collapsing contiguous decoder layers into one (the LLM counterpart of the
# CNN block collapse in manuscript/Tranfer/collapse.py).
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyPrune.collapseMethods import (
    collapse_block,
    collapse_only,
    count_params,
    find_decoder_stack,
    remove_layers,
)


class TinyBlock(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 2 * d)
        self.fc2 = nn.Linear(2 * d, d)
        self.layer_idx = -1

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(self.ln(x))))


class TinyLM(nn.Module):
    def __init__(self, n_layers=4, d=8, vocab=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([TinyBlock(d) for _ in range(n_layers)])
        self.head = nn.Linear(d, vocab)
        self.config = SimpleNamespace(num_hidden_layers=n_layers)

    def forward(self, ids):
        x = self.embed(ids)
        for i, layer in enumerate(self.layers):
            layer.layer_idx = i
            x = layer(x)
        return self.head(x)


def ids(batch=2, seq=5, vocab=16):
    return torch.randint(0, vocab, (batch, seq))


def test_find_decoder_stack():
    model = TinyLM()
    name, stack = find_decoder_stack(model)
    assert name == "layers"
    assert isinstance(stack, nn.ModuleList) and len(stack) == 4


def test_collapse_block_reduces_params_and_preserves_interface():
    model = TinyLM()
    before = count_params(model)
    out_before = model(ids())
    report = collapse_block(model, 1, 2)

    assert report["layers_before"] == 4 and report["layers_after"] == 3
    assert report["collapsed"] == [1, 2]
    assert len(model.layers) == 3
    assert count_params(model) < before
    out_after = model(ids())
    assert out_after.shape == out_before.shape
    assert torch.isfinite(out_after).all()
    assert model.config.num_hidden_layers == 3


def test_mean_merge_averages_the_span():
    torch.manual_seed(0)
    model = TinyLM()
    span = list(model.layers[1:3])
    expect = torch.stack([b.fc1.weight for b in span]).mean(0)
    collapse_block(model, 1, 2, merge="mean")
    assert torch.allclose(model.layers[1].fc1.weight, expect)


def test_merge_modes_pick_span_member():
    for mode, pos in (("first", 0), ("middle", 1), ("last", 2)):
        model = TinyLM()
        want = model.layers[1 + pos].fc2.bias.detach().clone()
        collapse_block(model, 1, 3, merge=mode)
        assert len(model.layers) == 2
        assert torch.equal(model.layers[1].fc2.bias, want)


def test_remove_layers_renumbers_and_syncs_config():
    model = TinyLM()
    report = remove_layers(model, 1, 2)
    assert report["removed_layers"] == [1, 2]
    assert len(model.layers) == 2
    assert [l.layer_idx for l in model.layers] == [0, 1]
    assert model.config.num_hidden_layers == 2


def test_collapse_only_multiple_spans_and_dry_run():
    model = TinyLM(n_layers=6)
    x = ids()
    untouched = model(x)
    dry = collapse_only(model=model, compression_set=[(0, 1), (4, 5)], dry_run=True)
    assert len(dry.layers) == 6
    assert torch.equal(dry(x), untouched)

    collapsed = collapse_only(model=model, compression_set={"a": (0, 1), "b": ("layers.4", "layers.5")})
    assert len(collapsed.layers) == 4
    assert collapsed._collapsed_blocks == [(0, 1), (4, 5)]
    assert torch.isfinite(collapsed(ids())).all()
    # the source model is never touched
    assert len(model.layers) == 6


def test_invalid_spans_raise():
    model = TinyLM()
    with pytest.raises(ValueError):
        collapse_block(model, 2, 9)  # out of range
    with pytest.raises(ValueError):
        remove_layers(model, 0, 4)  # would remove all layers
    with pytest.raises(ValueError):
        collapse_only(model=model, compression_set=[(0, 2), (2, 3)])  # overlap


def test_heterogeneous_span_cannot_merge_but_can_be_removed():
    model = TinyLM(n_layers=3)
    model.layers[1] = nn.Linear(8, 8)  # structurally different middle layer
    with pytest.raises(ValueError):
        collapse_block(model, 0, 2)
    report = remove_layers(model, 1, 1)  # removal needs no weight merge
    assert report["layers_after"] == 2
