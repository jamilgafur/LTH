# tests/test_pruning.py
"""Smoke test for the iterative magnitude pruning API (pyPrune.pruneMethods)."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pyPrune.pruneMethods.IterativePruner import IterativePruner
from pyPrune.strategies.MagnitudePruningStrategy import MagnitudePruningStrategy


def test_iterative_pruner_initializes(tmp_path):
    model = nn.Sequential(nn.Flatten(), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 3))
    data = TensorDataset(torch.randn(4, 1, 8, 8), torch.randint(0, 3, (4,)))
    loader = DataLoader(data, batch_size=2)

    pruner = IterativePruner(
        model=model,
        train_loader=loader,
        test_loader=loader,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        criterion=nn.CrossEntropyLoss(),
        strategy=MagnitudePruningStrategy(),
        steps=[0.5],
        pretrain_epochs=1,
        finetune_epochs=1,
        device="cpu",
        save_dir=str(tmp_path),
    )

    assert pruner.steps == [0.5]  # the synthetic 0.0 pretrain step is consumed
    assert pruner.initial_state  # prunable weights recorded
    assert (tmp_path / "pruner.pkl").exists()
