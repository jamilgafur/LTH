# tests/test_pruning.py
import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from pyPrune.pruning import IterativeMagnitudePruning

class TestIterativeMagnitudePruning(unittest.TestCase):
    def setUp(self):
        self.model = MagicMock(spec=nn.Module)
        self.X_train = torch.randn(10, 3, 32, 32)
        self.y_train = torch.randint(0, 10, (10,))
        self.final_sparsity = 0.5
        self.steps = 5
        self.device = 'cpu'
        self.save_dir = './test_pruning'
        self.imp = IterativeMagnitudePruning(
            model=self.model,
            X_train=self.X_train,
            y_train=self.y_train,
            final_sparsity=self.final_sparsity,
            steps=self.steps,
            device=self.device,
            save_dir=self.save_dir
        )

    @patch('os.makedirs')
    @patch('torch.save')
    def test_initialization(self, mock_save, mock_makedirs):
        self.assertEqual(self.imp.final_sparsity, self.final_sparsity)
        self.assertEqual(self.imp.steps, self.steps)
        self.assertEqual(self.imp.device, self.device)
        self.assertTrue(mock_makedirs.called)

    # Add other test methods here...

if __name__ == '__main__':
    unittest.main()
