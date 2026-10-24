"""CPU tests for the Transfer2 full-analysis suite (Workstreams A-D).

Covers: benchmark dict contract, AUROC math, ECE/MCE calibration, CIFAR-C
slicing/normalization, corruption aggregation, adversarial epsilon-sweep key
naming, and the torch-pruning optional-dependency paths of train_structural.
"""

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _stub_module(name):
    """Register an empty module with a valid __spec__ (torch dynamo introspects
    every entry in sys.modules and crashes on spec-less stubs)."""
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        sys.modules[name] = mod
    return sys.modules[name]


TRANSFER2_DIR = Path(__file__).resolve().parents[1] / "manuscript" / "Transfer2"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, TRANSFER2_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # visible to `from utils import *` inside main.py
    spec.loader.exec_module(module)
    return module


def _load_utils():
    for name in ("fvcore", "fvcore.nn", "pandas", "seaborn"):
        _stub_module(name)
    if getattr(sys.modules["fvcore.nn"], "FlopCountAnalysis", None) is None:
        sys.modules["fvcore.nn"].FlopCountAnalysis = object  # call fails -> flops=0 path
    # utils.py does `from main import PowerTracker` (script-dir circular import)
    _stub_module("main").PowerTracker = object
    return _load("utils", "utils.py")


u2 = _load_utils()
# main.py needs `from utils import *` to resolve to the loaded module and
# `from attacks import ...` / `import measure_power_usage` from the Transfer2 dir.
if str(TRANSFER2_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSFER2_DIR))
m2 = _load("transfer2_main", "main.py")


class FixedLogits(nn.Module):
    """Always emits the same logit vector regardless of input."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, x):
        return self.logits.unsqueeze(0).expand(x.shape[0], -1)


def _toy_loader(labels, n_features=4, batch_size=8):
    x = torch.zeros(len(labels), n_features)
    return DataLoader(TensorDataset(x, torch.tensor(labels)), batch_size=batch_size)


# ---------------------------------------------------------------- AUROC

def test_auroc_known_values():
    assert u2.auroc([0.9, 0.8], [0.1, 0.2]) == pytest.approx(1.0)
    assert u2.auroc([0.1, 0.2], [0.9, 0.8]) == pytest.approx(0.0)
    assert u2.auroc([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.5)  # all ties
    # Hand-computed: pos {0.6, 0.7}, neg {0.5, 0.75} -> wins 2/4 = 0.5
    assert u2.auroc([0.6, 0.7], [0.5, 0.75]) == pytest.approx(0.5)
    # pos {0.6, 0.8}, neg {0.5, 0.75} -> wins 3/4
    assert u2.auroc([0.6, 0.8], [0.5, 0.75]) == pytest.approx(0.75)
    assert u2.auroc([0.5], []) != u2.auroc([0.5], [])  # nan contract


# ---------------------------------------------------------------- calibration

def test_evaluate_calibration_synthetic():
    model = FixedLogits([1.0, 0.0])  # conf = softmax ~ 0.7311, predicts class 0
    labels = [0] * 8 + [1] * 2     # 80% accuracy
    loader = _toy_loader(labels)
    device = torch.device("cpu")

    ece, mce, rows = u2.evaluate_calibration(model, loader, device, n_bins=15)

    conf = float(torch.softmax(torch.tensor([1.0, 0.0]), dim=0)[0])
    expected_gap = abs(0.8 - conf)
    assert ece == pytest.approx(expected_gap, abs=1e-6)
    assert mce == pytest.approx(expected_gap, abs=1e-6)
    assert len(rows) == 15
    populated = [r for r in rows if r["n"] > 0]
    assert len(populated) == 1 and populated[0]["n"] == 10
    assert populated[0]["acc"] == pytest.approx(0.8)
    assert populated[0]["conf"] == pytest.approx(conf, abs=1e-6)


# ---------------------------------------------------------------- benchmark dict

def test_benchmark_model_returns_full_metrics_cpu():
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
        nn.Flatten(), nn.LazyLinear(10),
    )
    x = torch.randn(8, 3, 8, 8)
    loader = DataLoader(TensorDataset(x, torch.zeros(8, dtype=torch.long)), batch_size=4)
    device = torch.device("cpu")
    model(x[:1])  # materialize LazyLinear before deepcopy in benchmark

    bench = u2.benchmark_model(model, loader, device, num_batches=2, warmup_batches=1)

    assert set(bench) == {"avg_time_s", "latency_ms", "throughput_imgs", "batch_size",
                          "flops", "feature_map_mb", "peak_mem_mb"}
    assert bench["latency_ms"] > 0
    assert bench["throughput_imgs"] > 0
    assert bench["batch_size"] == 4
    assert bench["peak_mem_mb"] is None  # CPU contract
    assert bench["feature_map_mb"] > 0


# ---------------------------------------------------------------- CIFAR-C

def _make_fake_cifar_c(base, corruption_names=("brightness",), value_fn=None):
    sub = base / "CIFAR-10-C"
    sub.mkdir(parents=True)
    labels = np.arange(50000, dtype=np.int64) % 10
    np.save(sub / "labels.npy", labels)
    for name in corruption_names:
        data = np.zeros((50000, 32, 32, 3), dtype=np.uint8)
        for s in range(5):
            data[s * 10000:(s + 1) * 10000] = (s + 1) * 10  # distinct value per severity
        np.save(sub / f"{name}.npy", data)


def test_load_cifar_c_slicing_and_normalization(tmp_path):
    _make_fake_cifar_c(tmp_path)
    loader = u2.load_cifar_c(str(tmp_path), "Cifar10", "brightness", severity=3, batch_size=64)
    x_all, y_all = loader.dataset.tensors
    assert x_all.shape == (10000, 3, 32, 32)

    mean, std = u2.DATASET_NORMS["cifar10"]
    raw = torch.tensor(30.0 / 255.0)  # severity 3 -> raw uint8 value 30
    expected = (raw - torch.tensor(mean[0])) / torch.tensor(std[0])
    assert float(x_all[0, 0, 0, 0]) == pytest.approx(float(expected), abs=1e-5)
    assert y_all[:10000].eq(torch.arange(50000)[20000:30000] % 10).all()

    with pytest.raises(ValueError):
        u2.load_cifar_c(str(tmp_path), "Cifar10", "brightness", severity=6)
    with pytest.raises(ValueError):
        u2.load_cifar_c(str(tmp_path), "TinyImageNet", "brightness", severity=1)


def test_evaluate_corruptions_aggregation(tmp_path, monkeypatch):
    _make_fake_cifar_c(tmp_path, corruption_names=("fake_corr",))
    monkeypatch.setattr(u2, "CIFAR_C_CORRUPTIONS", ("fake_corr",))

    # Model predicts argmax over a fixed vector: always class 1.
    model = FixedLogits([0.0, 1.0] + [0.0] * 8)
    labels_mod10 = torch.arange(50000) % 10
    expected_acc = (labels_mod10[:10000] == 1).float().mean().item()  # severity 1 slice

    mean_by_model, rows = u2.evaluate_corruptions(
        {"toy": model}, str(tmp_path), "Cifar10", torch.device("cpu"), severities=(1,)
    )
    assert len(rows) == 1
    assert rows[0]["corruption"] == "fake_corr" and rows[0]["severity"] == 1
    assert rows[0]["acc"] == pytest.approx(expected_acc)
    assert mean_by_model["toy"] == pytest.approx(expected_acc)


# ---------------------------------------------------------------- OOD loader

def test_ood_loader_none_for_unsupported_dataset():
    assert u2.load_ood_test_loader("mnist") is None


# ---------------------------------------------------------------- adversarial sweep

def test_adversarial_epsilon_sweep_keys_and_curve_rows():
    model = nn.Sequential(nn.Flatten(), nn.LazyLinear(10))
    data_batch = (torch.randn(4, 3, 2, 2), torch.randint(0, 10, (4,)))
    model(data_batch[0])  # materialize LazyLinear
    device = torch.device("cpu")
    curve_rows = []

    results = m2.evaluate_adversarial_robustness(
        {"toy": model}, data_batch, device, epsilons=(0.03, 0.05), curve_rows=curve_rows
    )
    # eps=0.03 keeps historical unsuffixed keys; others get _eps suffixes
    assert "adv_acc_FGSM_toy" in results
    assert "adv_acc_PGD_toy" in results
    assert "adv_acc_FGSM_eps0.05_toy" in results
    assert "adv_acc_PGD_eps0.05_toy" in results
    for v in results.values():
        assert 0.0 <= v <= 1.0
    assert len(curve_rows) == 2 * 2 * 1  # attacks x epsilons x models
    assert {r["model"] for r in curve_rows} == {"toy"}
    assert {r["epsilon"] for r in curve_rows} == {0.03, 0.05}


def test_append_long_csv(tmp_path):
    path = tmp_path / "rows.csv"
    m2._append_long_csv(str(path), [{"a": 1, "b": 2}], ["a", "b"])
    m2._append_long_csv(str(path), [{"a": 3, "b": 4}], ["a", "b"])
    lines = path.read_text().strip().splitlines()
    assert lines == ["a,b", "1,2", "3,4"]


# ---------------------------------------------------------------- train_structural

def test_train_structural_resume_path_needs_no_dependency(tmp_path):
    # A finished artifact must load even without torch-pruning installed.
    model = nn.Sequential(nn.Linear(4, 4))
    save_path = tmp_path / "structural.pt"
    torch.save({"model": model}, save_path)

    loaded = u2.train_structural(model, _toy_loader([0, 1]), torch.device("cpu"),
                                 epochs=1, target_sparsity=0.5, save_path=str(save_path))
    assert isinstance(loaded, nn.Module)
    assert not loaded.training


def test_train_structural_import_error_without_torch_pruning(tmp_path):
    pytest.importorskip("torch")
    try:
        import torch_pruning  # noqa: F401
        pytest.skip("torch-pruning installed in this env; ImportError path untestable")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="torch-pruning"):
        u2.train_structural(nn.Sequential(nn.Linear(4, 4)), _toy_loader([0, 1]),
                            torch.device("cpu"), epochs=1, target_sparsity=0.5,
                            save_path=str(tmp_path / "nope.pt"))
