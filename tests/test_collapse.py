try:
    import fvcore  # noqa: F401
except ImportError:
    # Minimal stub so the discovery acceptance logic can be exercised in
    # environments without the optional fvcore dependency. FLOPs are taken as
    # proportional to parameter count, which preserves the accept/reject
    # semantics of discover_regions for unit-testing purposes.
    import sys
    import types

    _fvn = types.ModuleType("fvcore.nn")

    class _FlopCountAnalysisStub:
        def __init__(self, model, inp):
            self._model = model

        def unsupported_ops_warnings(self, flag):
            return self

        def total(self):
            return sum(p.numel() for p in self._model.parameters())

    _fvn.FlopCountAnalysis = _FlopCountAnalysisStub
    sys.modules.setdefault("fvcore", types.ModuleType("fvcore"))
    sys.modules["fvcore.nn"] = _fvn

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from pyPrune.collapse import (
    SmartIdentity,
    calibrate_layer_variances,
    collapse_only,
    discover_regions,
    get_layer_variances,
    patch_skip_connections,
    wrap_pools_safe,
)


def _toy_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),   # 0
        nn.ReLU(),                        # 1
        nn.Conv2d(8, 16, 3, padding=1),  # 2
        nn.ReLU(),                        # 3
        nn.Conv2d(16, 32, 3, padding=1), # 4
        nn.ReLU(),                        # 5
        nn.AdaptiveAvgPool2d(1),          # 6
        nn.Flatten(),                     # 7
        nn.Linear(32, 10),                # 8
    )


def test_smart_identity_passthrough():
    identity = SmartIdentity()
    x = torch.randn(2, 4, 8, 8)
    assert torch.equal(identity(x), x)
    # 2D flattened tensors are unsqueezed to 4D by design.
    flat = torch.randn(2, 4)
    assert identity(flat).shape == (2, 4, 1, 1)
    # Attribute chains and indexing are absorbed without raising.
    assert identity.foo.bar[0] is identity
    with pytest.raises(AttributeError):
        _ = identity.weight


def test_collapse_only_reduces_params_and_forward_works():
    model = _toy_model()
    pre = sum(p.numel() for p in model.parameters())

    collapsed = collapse_only(
        model=model,
        compression_set={"block": ("2", "5")},
        input_shape=(1, 3, 8, 8),
        device="cpu",
        debug=False,
    )

    post = sum(p.numel() for p in collapsed.parameters())
    assert post < pre

    out = collapsed(torch.randn(1, 3, 8, 8))
    assert out.shape == (1, 10)


def test_wrap_pools_safe_wraps_pooling_layers():
    model = _toy_model()
    wrap_pools_safe(model)
    from pyPrune.collapse import _SafePool
    assert isinstance(model[6], _SafePool)
    out = model(torch.randn(1, 3, 8, 8))
    assert out.shape == (1, 10)


def test_discover_regions_accept_and_reject():
    model = _toy_model()
    dummy = torch.randn(1, 3, 8, 8)
    cnn_layers = list(get_layer_variances(model, dummy).keys())
    assert cnn_layers == ["0", "2", "4", "8"]

    # Reject: increasing variance past the vetoed stem leaves too few active
    # parametric layers to form a region.
    v_increasing = [0.5, 0.6, 0.7, 0.8]
    regions_none = discover_regions(model, cnn_layers, v_increasing, input_shape=(1, 3, 8, 8))
    assert not any(k.startswith("Set_") for k in regions_none)

    # Accept: layers '2' and '4' sit below the local rolling mean and form a
    # collapse candidate that reduces params/FLOPs.
    v_decreasing = [1.0, 0.1, 0.05, 0.9]
    regions = discover_regions(model, cnn_layers, v_decreasing, input_shape=(1, 3, 8, 8))
    assert regions["Set_0"] == ("2", "4")


class _Branchy(nn.Module):
    """Main path: stem -> body -> head. fc_wrong is registered between stem and
    body but never executes, so named_modules() DFS order disagrees with the
    true execution order."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.fc_wrong = nn.Linear(4, 4)
        self.body = nn.Conv2d(8, 8, 3, padding=1)
        self.head = nn.Linear(8, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


def test_execution_order_linear_discovery():
    from pyPrune.collapse.graph_utils import execution_order
    from pyPrune.collapse.replace import _find_next_linear

    model = _Branchy()
    # Legacy DFS scan picks the Linear that never fires on the forward path.
    legacy_name, _ = _find_next_linear(model, "stem", False)
    assert legacy_name == "fc_wrong"
    # Execution-order probing picks the Linear that actually fires downstream.
    order = execution_order(model, (1, 3, 8, 8))
    name, mod = _find_next_linear(model, "stem", False, order=order)
    assert name == "head"
    assert isinstance(mod, nn.Linear)


def test_content_based_norm_act_selection():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),   # 0
        nn.ReLU(),                        # 1
        nn.Conv2d(8, 16, 3, padding=1),  # 2
        nn.GroupNorm(4, 16),              # 3
        nn.GELU(),                        # 4
        nn.Conv2d(16, 16, 3, padding=1), # 5
        nn.ReLU(),                        # 6
        nn.AdaptiveAvgPool2d(1),          # 7
        nn.Flatten(),                     # 8
        nn.Linear(16, 10),                # 9
    )
    collapsed = collapse_only(
        model=model,
        compression_set={"block": ("2", "5")},
        input_shape=(1, 3, 8, 8),
        device="cpu",
        debug=False,
    )
    rep = collapsed[2]
    assert isinstance(rep[1], nn.GroupNorm)  # adopted from block content
    assert isinstance(rep[2], nn.GELU)       # adopted from block content
    out = collapsed(torch.randn(1, 3, 8, 8))
    assert out.shape == (1, 10)

    # Legacy fallback: no norm/act clues in the span -> BN + ReLU.
    legacy = collapse_only(
        model=_toy_model(),
        compression_set={"block": ("2", "5")},
        input_shape=(1, 3, 8, 8),
        device="cpu",
        debug=False,
    )
    rep = legacy[2]
    assert isinstance(rep[1], nn.BatchNorm2d)
    assert isinstance(rep[2], nn.ReLU)


def test_calibrate_layer_variances_deterministic():
    torch.manual_seed(0)
    batches = [(torch.randn(2, 3, 8, 8), torch.zeros(2)) for _ in range(4)]
    model = _toy_model()
    s1 = calibrate_layer_variances(model, batches, k=4)
    s2 = calibrate_layer_variances(model, batches, k=4)
    assert s1 == s2
    assert set(s1.keys()) == set(get_layer_variances(model, torch.randn(1, 3, 8, 8)).keys())
    for name, (mean, std) in s1.items():
        assert mean >= 0.0 and std >= 0.0


def test_discover_regions_scorers():
    model = _toy_model()
    keys = ["0", "2", "4", "8"]

    # percentile: bottom-50% variance -> layers '2' and '4'
    regions = discover_regions(
        model, keys, [1.0, 0.1, 0.05, 0.9],
        input_shape=(1, 3, 8, 8), scorer="percentile", active_fraction=0.5,
    )
    assert regions["Set_0"] == ("2", "4")

    # cv: lowest batch-to-batch drift -> layers '2' and '4'
    calib = {"0": (1.0, 0.5), "2": (0.1, 0.001), "4": (0.05, 0.001), "8": (0.9, 0.36)}
    regions = discover_regions(
        model, keys, calib,
        input_shape=(1, 3, 8, 8), scorer="cv", active_fraction=0.5,
    )
    assert regions["Set_0"] == ("2", "4")

    # cv without calibration stds is a hard error
    with pytest.raises(ValueError):
        discover_regions(
            model, keys, [1.0, 0.1, 0.05, 0.9],
            input_shape=(1, 3, 8, 8), scorer="cv",
        )


def test_lsq_init_closer_to_original_than_random():
    probe = torch.randn(1, 3, 8, 8)
    torch.manual_seed(7)
    base = _toy_model()
    orig = base(probe).detach()

    kwargs = dict(
        compression_set={"b": ("2", "5")},
        input_shape=(1, 3, 8, 8),
        device="cpu",
        debug=False,
    )
    torch.manual_seed(123)
    m_rand = collapse_only(model=base, init="random", **kwargs)
    torch.manual_seed(123)
    m_lsq = collapse_only(model=base, init="lsq", **kwargs)

    err_rand = (m_rand(probe) - orig).norm()
    err_lsq = (m_lsq(probe) - orig).norm()
    assert err_lsq < err_rand


def test_topology_guard_skips_non4d_block():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),  # 0
        nn.ReLU(),                       # 1
        nn.AdaptiveAvgPool2d(1),         # 2
        nn.Flatten(),                    # 3
        nn.Linear(8, 8),                 # 4
        nn.ReLU(),                       # 5
        nn.Linear(8, 10),                # 6
    )
    pre = sum(p.numel() for p in model.parameters())
    # Span ('3' -> '5') produces 2D activations; must be skipped, not corrupted.
    collapsed = collapse_only(
        model=model,
        compression_set={"bad": ("3", "5")},
        input_shape=(1, 3, 8, 8),
        device="cpu",
        debug=False,
    )
    post = sum(p.numel() for p in collapsed.parameters())
    assert post == pre
    out = collapsed(torch.randn(1, 3, 8, 8))
    assert out.shape == (1, 10)


class _FancySkipBlock(nn.Module):
    """Residual block whose class name matches no known architecture;
    must be discovered purely by its attribute signature."""

    def __init__(self, c=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c),
        )
        self.shortcut = nn.Identity()

    def forward(self, x):
        return F.relu(self.block(x) + self.shortcut(x))


class _SkipToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.res = _FancySkipBlock(8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.res(x)
        return self.fc(self.pool(x).flatten(1))


def test_whole_model_pickle_roundtrip():
    import os
    import tempfile
    from pyPrune.collapse.patching import _ResNetForwardPatch

    torch.manual_seed(3)
    model = _SkipToy()
    wrap_pools_safe(model)
    collapsed = collapse_only(
        model=model,
        compression_set={"b": ("res.block.0", "res.block.2")},
        input_shape=(1, 3, 8, 8),
        device="cpu",
        debug=False,
    )
    collapsed.eval()

    # Signature-based patch attached (class name matches nothing known), the
    # collapse left SmartIdentity taint inside the block, and pools are wrapped.
    assert isinstance(collapsed.res.forward, _ResNetForwardPatch)
    assert any(isinstance(m, SmartIdentity) for m in collapsed.res.block.modules())

    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        expected = collapsed(x)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "whole.pt")
        torch.save(collapsed, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)

    loaded.eval()
    assert isinstance(loaded.res.forward, _ResNetForwardPatch)
    with torch.no_grad():
        out = loaded(x)
    assert torch.allclose(out, expected)
    assert sum(p.numel() for p in loaded.parameters()) == sum(p.numel() for p in collapsed.parameters())
    for (n1, v1), (n2, v2) in zip(collapsed.state_dict().items(), loaded.state_dict().items()):
        assert n1 == n2
        assert torch.equal(v1, v2)


class _ForeignBlock(nn.Module):
    """Custom-forward block with an attribute layout matching no signature."""

    def __init__(self):
        super().__init__()
        self.inner = nn.Conv2d(4, 4, 3, padding=1)
        self.gain = 2.0

    def forward(self, x):
        return F.relu(self.inner(x)) * self.gain


def test_safe_module_forward_fallback_for_unknown_topology():
    from pyPrune.collapse.patching import _SafeModuleForward

    root = nn.Module()
    blk = _ForeignBlock()
    root.foreign = blk
    untouched = _ForeignBlock()
    root.untouched = untouched

    # No taint yet: nothing to patch.
    patch_skip_connections(root)
    assert not isinstance(blk.forward, _SafeModuleForward)

    # Excise an internal slot (SmartIdentity taint) -> generic fallback kicks in,
    # driven by structure, not by class name.
    blk.excised_slot = SmartIdentity()
    patch_skip_connections(root)
    assert isinstance(blk.forward, _SafeModuleForward)
    # Untainted foreign block in the same model stays untouched.
    assert not isinstance(untouched.forward, _SafeModuleForward)

    x = torch.randn(1, 4, 6, 6)
    # Original forward still works through the wrapper...
    assert torch.allclose(blk(x), F.relu(blk.inner(x)) * 2.0)
    # ...and once the original can no longer run, the wrapper degrades to identity.
    del blk.gain
    assert torch.equal(blk(x), x)


def test_signature_patch_is_class_name_agnostic():
    from pyPrune.collapse.patching import _ResNetForwardPatch

    model = _SkipToy()
    patch_skip_connections(model)
    # _FancySkipBlock is not a ResNet/ConvNeXt/Inception/Xception class name.
    assert isinstance(model.res.forward, _ResNetForwardPatch)
    # Idempotent: a second pass does not re-wrap.
    first = model.res.forward
    patch_skip_connections(model)
    assert model.res.forward is first
    # Patched forward still produces the residual result.
    x = torch.randn(1, 3, 8, 8)
    out = model(x)
    assert out.shape == (1, 10)


def test_safe_pool_warns_once_but_keeps_adapting(capsys):
    model = nn.Sequential(nn.MaxPool2d(3, stride=2))
    wrap_pools_safe(model)
    x = torch.randn(1, 1, 2, 2)  # kernel 3 > 2x2 input -> adaptive fallback path

    for _ in range(5):
        out = model(x)
        # adaptive_avg_pool2d(x, (2, 2)) on a 2x2 input is shape-preserving.
        assert out.shape == (1, 1, 2, 2)

    warns = [l for l in capsys.readouterr().out.splitlines() if "exceeds input spatial" in l]
    assert len(warns) == 1
    assert "further occurrences suppressed" in warns[0]
    assert "MaxPool2d" in warns[0]
    # Behavior unchanged after suppression: still adapts, never crashes.
    assert model(x).shape == (1, 1, 2, 2)
