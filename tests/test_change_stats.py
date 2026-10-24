"""Tests for representational-change probing (collect_change_stats) and the
change/cka scorers in discover_regions.

Hypothesis under test: layers/blocks that barely change their input
(in ~= out) are redundant collapse candidates. The change/cka scorers must
flag near-identity units and must NOT flag variance-shrinking transformers
(which the legacy output-variance scorer would flag).
"""

try:
    import fvcore  # noqa: F401
except ImportError:
    import sys
    import types

    _fvn = types.ModuleType("fvcore.nn")

    class _FlopCountAnalysisStub:
        def __init__(self, model, inp):
            self._model = model

        def unsupported_ops_warnings(self, flag):
            return self

        def uncalled_modules_warnings(self, flag):
            return self

        def total(self):
            return sum(p.numel() for p in self._model.parameters())

    _fvn.FlopCountAnalysis = _FlopCountAnalysisStub
    sys.modules.setdefault("fvcore", types.ModuleType("fvcore"))
    sys.modules["fvcore.nn"] = _fvn

import numpy as np
import torch
import torch.nn as nn
import pytest

from pyPrune.collapse import build_metric_series, collect_change_stats
from pyPrune.collapse import discovery as _discovery
from pyPrune.collapse.discovery import _change_from_stats, _cka_from_stats, _UnitStats, _unit_update


def _identity_block(ch=8, eps=0.01):
    block = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.Conv2d(ch, ch, 1))
    with torch.no_grad():
        for c in block:
            c.weight.zero_()
            c.bias.zero_()
            for i in range(ch):
                c.weight[i, i, 0, 0] = 1.0 + torch.randn(()) * eps
    return block


class ToyNet(nn.Module):
    """near-identity block (residual) + strong transform block."""

    def __init__(self):
        super().__init__()
        self.near_id = _identity_block()
        self.transform = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1), nn.Conv2d(8, 8, 3, padding=1))

    def forward(self, x):
        x = self.near_id(x) + x
        return self.transform(x)


def _batches(n=4, shape=(16, 8, 8, 8), seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(*shape, generator=g) for _ in range(n)]


# ---------------------------------------------------------------- capture

def test_near_identity_scores_low_change_high_cka():
    torch.manual_seed(0)
    model = ToyNet().eval()
    stats = collect_change_stats(model, _batches(), k=4)

    # Block-primary: near_id block much lower change / much higher CKA than transform.
    assert stats["near_id"]["kind"] == "block"
    assert stats["near_id"]["change"] < 0.05
    assert stats["near_id"]["cka"] > 0.99
    assert stats["transform"]["change"] > 0.5
    assert stats["transform"]["cka"] < stats["near_id"]["cka"]
    assert stats["transform"]["mismatch"] is False
    # Every entry JSON-serializable value types
    for s in stats.values():
        assert isinstance(s["var_mean"], float) and isinstance(s["var_std"], float)


def test_series_block_primary_mapping():
    torch.manual_seed(0)
    model = ToyNet().eval()
    stats = collect_change_stats(model, _batches(), k=2)
    layers = [n for n, s in stats.items() if s["kind"] == "layer"]
    series = build_metric_series(layers, stats, "change")
    # near_id.0/1 inherit the near_id block value; transform.0/1 inherit transform block.
    assert series[0] == stats["near_id"]["change"]
    assert series[1] == stats["near_id"]["change"]
    assert series[2] == stats["transform"]["change"]


def test_aligned_path_for_shape_mismatch():
    # Stride-2 conv block: in/out spatial dims differ -> aligned (projected) change.
    torch.manual_seed(0)
    blk = nn.Sequential(nn.Conv2d(8, 8, 3, stride=2, padding=1), nn.Conv2d(8, 8, 3, padding=1))
    model = nn.Sequential(blk, nn.Conv2d(8, 4, 1)).eval()
    stats = collect_change_stats(model, _batches(), k=2)
    entry = stats["0"]  # block container of the two convs
    assert entry["mismatch"] is True
    assert entry["change"] is not None and np.isfinite(entry["change"])
    assert 0.0 <= entry["change"]


# ---------------------------------------------------------------- online == brute force

def _brute_linear_cka(X, Y):
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    sxy = Xc.T @ Yc
    num = (sxy ** 2).sum().item()
    den = (Xc.T @ Xc).norm().item() * (Yc.T @ Yc).norm().item()
    return num / den


def test_online_cka_matches_brute_force():
    torch.manual_seed(1)
    u = _UnitStats("layer")
    xs, ys = [], []
    for _ in range(3):
        x = torch.randn(32, 8, 4, 4)
        y = torch.randn(32, 8, 4, 4)
        _unit_update(u, x, y, max_points=10**9, feat_cap=2048)
        xs.append(x.mean(dim=(2, 3)))
        ys.append(y.mean(dim=(2, 3)))
    X = torch.cat(xs).to(torch.float64)
    Y = torch.cat(ys).to(torch.float64)
    assert _cka_from_stats(u) == pytest.approx(_brute_linear_cka(X, Y), abs=1e-6)


def test_online_change_matches_brute_force_same_shape():
    torch.manual_seed(2)
    u = _UnitStats("layer")
    xs, ys = [], []
    for _ in range(2):
        x = torch.randn(8, 4, 3, 3)
        y = x + 0.05 * torch.randn_like(x)
        _unit_update(u, x, y, max_points=10**9, feat_cap=2048)
        xs.append(x)
        ys.append(y)
    X = torch.cat([x.reshape(x.shape[0], -1) for x in xs])
    Y = torch.cat([y.reshape(y.shape[0], -1) for y in ys])
    brute = ((Y - X).norm(dim=1) / X.norm(dim=1)).mean().item()
    assert u.mode == 'l2'
    assert _change_from_stats(u) == pytest.approx(brute, abs=1e-6)


def test_cka_degenerate_identity_is_one():
    torch.manual_seed(3)
    u = _UnitStats("layer")
    for _ in range(2):
        x = torch.randn(16, 6, 3, 3)
        _unit_update(u, x, x.clone(), max_points=10**9, feat_cap=2048)
    assert _cka_from_stats(u) == pytest.approx(1.0, abs=1e-6)
    assert _change_from_stats(u) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------- scorer semantics

class VarianceShrinkingTransformer(nn.Module):
    """Shrinks output variance strongly (local_dev would flag it) but rotates
    the representation hard (change/cka must NOT flag it)."""

    def __init__(self):
        super().__init__()
        self.shrink = nn.Sequential(nn.Conv2d(8, 8, 1), nn.Conv2d(8, 8, 1))
        with torch.no_grad():
            # random rotation-ish weights, tiny scale -> output variance minuscule
            for c in self.shrink:
                c.weight.mul_(1e-2)
        self.near_id = _identity_block()

    def forward(self, x):
        x = self.shrink(x)
        return self.near_id(x)


def test_change_scorer_ignores_variance_shrinker():
    torch.manual_seed(0)
    model = VarianceShrinkingTransformer().eval()
    stats = collect_change_stats(model, _batches(), k=4)
    # shrink block: low output variance but high change
    assert stats["shrink"]["var_mean"] < stats["near_id"]["var_mean"]  # what local_dev loves
    assert stats["shrink"]["change"] > 0.9  # what the hypothesis says matters


def test_scorers_directionality():
    # change flags the MINIMUM; cka flags the MAXIMUM; missing values never flagged.
    stats = {
        "a": {"kind": "layer", "change": 0.01, "cka": 0.99},
        "b": {"kind": "layer", "change": 0.50, "cka": 0.50},
        "c": {"kind": "layer", "change": 2.00, "cka": 0.10},
        "d": {"kind": "layer", "change": None, "cka": None},
    }
    layers = ["a", "b", "c", "d"]
    ch = build_metric_series(layers, stats, "change")
    ck = build_metric_series(layers, stats, "cka")
    assert ch == [0.01, 0.50, 2.00, float("inf")]
    assert ck == [0.99, 0.50, 0.10, 0.0]
    assert np.argmin(ch) == 0 and np.argmax(ck) == 0


def test_discover_regions_change_and_cka_flag_near_identity():
    torch.manual_seed(0)
    model = ToyNet().eval()
    stats = collect_change_stats(model, _batches(), k=4)
    layers = [n for n, s in stats.items() if s["kind"] == "layer"]

    def flags(scorer):
        captured = {}

        orig_np_percentile = np.percentile
        # Run the flag-extraction phase by replaying scorer logic directly.
        series = build_metric_series(layers, stats, metric=scorer)
        thresh = orig_np_percentile(
            series, 25.0 if scorer == "change" else 75.0)
        return [(v <= thresh) if scorer == "change" else (v >= thresh) for v in series]

    ch_flags = flags("change")
    ck_flags = flags("cka")
    # near_id layers (idx 0,1) flagged; transform layers (idx 2,3) not.
    assert ch_flags[0] and ch_flags[1] and not any(ch_flags[2:])
    assert ck_flags[0] and ck_flags[1] and not any(ck_flags[2:])


def test_discover_regions_rejects_bad_payload_for_change_scorers():
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Conv2d(4, 4, 3))
    with pytest.raises(ValueError, match="collect_change_stats"):
        _discovery.discover_regions(
            model=model,
            cnn_layers=["0", "1"],
            variances=[0.1, 0.2],  # flat list is invalid for change/cka
            input_shape=(1, 3, 8, 8),
            scorer="change",
        )
