"""Tests for recompute_test_acc.py helpers and trainer.evaluate().

The recompute script backfills test-set accuracy into merged_metrics.json.
Its non-trivial logic is the checkpoint-stem -> metrics-key mapping, the
legacy 0-1 fraction cleanup, and BN recalibration; trainer.evaluate() must
return percent accuracy
on the given loader (it is the number the manuscript tables will report).
"""

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _load_module(name, relpath):
    path = Path(__file__).resolve().parents[1] / "manuscript" / "Tranfer" / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metrics_key_for_maps_checkpoint_stems():
    rc = _load_module("recompute_test_acc", "recompute_test_acc.py")
    metrics = {"Control_JF": {}, "Control_quant_JF": {}, "Control_Continuted_JF": {}}

    assert rc.metrics_key_for("JF_Control", metrics) == "Control_JF"
    assert rc.metrics_key_for("JF_Control_quant", metrics) == "Control_quant_JF"
    assert rc.metrics_key_for("JF_Control_Continuted", metrics) == "Control_Continuted_JF"
    assert rc.metrics_key_for("JF_Unknown", metrics) is None


def test_fix_legacy_units_scales_only_fractions():
    rc = _load_module("recompute_test_acc", "recompute_test_acc.py")
    metrics = {
        "a": {"final_accuracy": 0.94, "timestamp_recovered": "auto_recovered_live"},
        "b": {"final_accuracy": 94.0, "timestamp_recovered": "auto_recovered_live"},
        "c": {"final_accuracy": 0.5},  # training-loop rows are already percent
    }

    assert rc.fix_legacy_units(metrics)
    assert metrics["a"]["final_accuracy"] == 94.0
    assert metrics["b"]["final_accuracy"] == 94.0
    assert metrics["c"]["final_accuracy"] == 0.5


def test_recalibrate_bn_fixes_broken_running_stats():
    rc = _load_module("recompute_test_acc", "recompute_test_acc.py")
    device = torch.device("cpu")
    model = nn.BatchNorm2d(3)
    with torch.no_grad():
        model.running_mean.fill_(999.0)  # simulate corrupt Tranfer-era stats
        model.running_var.fill_(0.001)
    x = torch.full((16, 3, 8, 8), 2.0)
    loader = DataLoader(TensorDataset(x, torch.zeros(16, dtype=torch.long)), batch_size=4)

    rc.recalibrate_bn(model, loader, device, num_batches=4)

    assert torch.allclose(model.running_mean, torch.full((3,), 2.0), atol=1e-5)


def test_evaluate_returns_percent_accuracy():
    trainer = _load_module("tranfer_trainer", "trainer.py")
    device = torch.device("cpu")
    x = torch.zeros(4, 3, 4, 4)
    y = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 10))
    with torch.no_grad():
        model[1].bias.zero_()
        model[1].weight.zero_()  # predicts class 0 for every input -> 100%

    assert trainer.evaluate(model, loader, device) == 100.0
