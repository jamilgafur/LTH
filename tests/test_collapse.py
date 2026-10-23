import torch
import torch.nn as nn
import pytest

from pyPrune.collapse import (
    SmartIdentity,
    collapse_only,
    discover_regions,
    get_layer_variances,
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
    pytest.importorskip("fvcore")

    model = _toy_model()
    dummy = torch.randn(1, 3, 8, 8)
    cnn_layers = list(get_layer_variances(model, dummy).keys())
    assert cnn_layers == ["0", "2", "4", "8"]

    # Vetoed stem: no candidates -> no accepted regions.
    v_hi_first = [1.0, 0.3, 0.1, 0.01]
    regions_none = discover_regions(model, cnn_layers, v_hi_first, input_shape=(1, 3, 8, 8))
    assert not any(k.startswith("Set_") for k in regions_none)

    # Monotonic decreasing: all probe layers become collapsible candidates.
    v_decreasing = [1.0, 0.1, 0.05, 0.01]
    regions = discover_regions(model, cnn_layers, v_decreasing, input_shape=(1, 3, 8, 8))
    assert regions["Set_0"] == ("0", "4")
