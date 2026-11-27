# tests/test_cnn_collapse.py
"""Smoke tests for the CNN block-collapse module (moved from manuscript/Tranfer)."""
import pytest
import torch
import torch.nn as nn

from pyPrune.collapseMethods.CNNCollapse import (
    _wrap_pools_safe,
    collapse_only,
    count_trainable_params,
)


class TinyCNN(nn.Module):
    """Small conv net in the study's shape: features -> pool -> linear."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Linear(8 * 4 * 4, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x.flatten(1))


def test_collapse_block_reduces_params_and_keeps_forward():
    model = TinyCNN()
    x = torch.randn(2, 1, 28, 28)

    collapsed = collapse_only(
        model=model,
        compression_set=[("features.0", "features.2")],
        input_shape=(1, 1, 28, 28),
        debug=False,
    )

    out = collapsed(x)
    assert out.shape == (2, 10)
    assert count_trainable_params(collapsed) < count_trainable_params(model)
    # the original model is untouched (collapse works on a deepcopy)
    assert len(list(model.features.children())) == 6


def test_collapse_dry_run_modifies_nothing():
    model = TinyCNN()

    out_model = collapse_only(
        model=model,
        compression_set=[("features.0", "features.2")],
        input_shape=(1, 1, 28, 28),
        debug=False,
        dry_run=True,
    )

    assert count_trainable_params(out_model) == count_trainable_params(model)
    assert out_model(torch.randn(2, 1, 28, 28)).shape == (2, 10)


def test_collapse_failure_raises_instead_of_exiting():
    # a span that names no real layer must raise, never kill the interpreter
    with pytest.raises(RuntimeError):
        collapse_only(
            model=TinyCNN(),
            compression_set=[("nope.0", "nope.2")],
            input_shape=(1, 1, 28, 28),
            debug=False,
        )


def test_facade_exports_tranfer_study_names():
    # manuscript/Tranfer imports these from the facade
    assert callable(_wrap_pools_safe)
