"""Smoke tests for the Tranfer training loop (tqdm + optimizations).

Load trainer.py directly from manuscript/Tranfer (it is not an installed
package module), then verify that:
  1. tqdm progress rendering does not affect training numerics.
  2. A non-TTY (PBS-style) run emits no progress-bar spam.
  3. Metrics are identical across progress settings for a fixed seed.
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def _load_trainer_module():
    trainer_path = Path(__file__).resolve().parents[1] / "manuscript" / "Tranfer" / "trainer.py"
    spec = importlib.util.spec_from_file_location("tranfer_trainer", trainer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_data(seed=11):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(32, 3, 8, 8, generator=g)
    y = torch.randint(0, 10, (32,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


def _make_model_and_opt(seed=5):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 10))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


def _run_epoch(trainer, loader, device, force_tty):
    model, optimizer = _make_model_and_opt()
    if force_tty is not None:
        # Monkeypatch isatty so tqdm enablement is deterministic under pytest.
        stderr = sys.stderr
        original = stderr.isatty
        stderr.isatty = lambda: force_tty
        try:
            loss, acc = trainer.train_one_epoch(
                model, loader, optimizer, device, epoch=0, total_epochs=2
            )
        finally:
            stderr.isatty = original
    else:
        loss, acc = trainer.train_one_epoch(
            model, loader, optimizer, device, epoch=0, total_epochs=2
        )
    return loss, acc


def test_train_one_epoch_tqdm_parity(capsys):
    trainer = _load_trainer_module()
    device = torch.device("cpu")
    loader = _make_data()

    # tqdm disabled (non-TTY, like a PBS job): no stderr spam, runs fine.
    loss_off, acc_off = _run_epoch(trainer, loader, device, force_tty=False)
    captured = capsys.readouterr()
    assert captured.err == ""

    # tqdm enabled (forced TTY): identical numerics.
    loss_on, acc_on = _run_epoch(trainer, loader, device, force_tty=True)
    assert loss_off == loss_on
    assert acc_off == acc_on

    # Deterministic repeat of the disabled path.
    loss_off2, acc_off2 = _run_epoch(trainer, loader, device, force_tty=False)
    assert (loss_off2, acc_off2) == (loss_off, acc_off)


def test_train_one_epoch_backward_compatible_signature():
    """Original 4-arg call signature (no epoch kwargs) still works."""
    trainer = _load_trainer_module()
    device = torch.device("cpu")
    loader = _make_data()
    model, optimizer = _make_model_and_opt()
    loss, acc = trainer.train_one_epoch(model, loader, optimizer, device)
    assert 0.0 < loss < 100.0
    assert 0.0 <= acc <= 100.0
