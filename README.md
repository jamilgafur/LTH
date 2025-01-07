Here's the updated and comprehensive README, which now includes all of the information we've discussed:

---

# Iterative Magnitude Pruning for Neural Networks

This repository implements an iterative magnitude pruning algorithm for pruning neural networks in PyTorch. The pruning method involves gradually removing less significant weights from the model to achieve a specified sparsity, followed by fine-tuning the pruned model to recover performance.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Usage](#usage)
  - [Training and Pruning](#training-and-pruning)
  - [Pruning Process](#pruning-process)
  - [Running with Docker (GPU Support)](#running-with-docker-gpu-support)
- [Metrics](#metrics)
- [Logging](#logging)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository provides an implementation of iterative magnitude pruning. The goal of this method is to achieve model sparsity by progressively removing small magnitude weights while maintaining the model’s accuracy as much as possible through fine-tuning.

The pruning process is done in several steps:

1. **Pretraining**: Train the model without pruning for a given number of epochs.
2. **Pruning**: Gradually prune the model by iteratively zeroing out weights with the smallest magnitudes.
3. **Fine-tuning**: After each pruning step, fine-tune the model to recover any lost performance.
4. **Checkpointing**: Save model states at each pruning step for recovery or further analysis.

The algorithm uses the MNIST dataset, but it can be extended to other datasets or models.

## Repository Structure

Here’s a breakdown of the repository structure:

```
├── data
│   └── MNIST
│       └── raw
├── example.py               # Example script for using the pruning algorithm
├── pruning_checkpoints       # Directory for saving pruned model checkpoints
│   ├── pruned_model_step_0.pth
│   └── pruning_metrics.json  # JSON file with pruning metrics
├── pyPrune                   # Core implementation of pruning
│   ├── models
│   │   └── LeNet.py          # Example model (LeNet)
│   ├── pruning.py            # Main pruning logic
│   ├── model_utils.py        # Helper functions for model handling
│   ├── utils.py              # Utility functions
├── requirements.txt          # Required dependencies
├── setup.py                  # Package setup
├── tests                     # Unit tests
│   ├── test_pruning.py       # Test cases for pruning algorithm
├── README.md                 # Project documentation
```

### Files of Interest:
- `example.py`: This file contains an example of how to set up and run the pruning process using the `IterativeMagnitudePruning` class.
- `pyPrune/`: The main implementation folder, where pruning logic and related utilities are defined.
- `pruning_checkpoints/`: Contains saved model checkpoints after each pruning step.
- `requirements.txt`: Lists the required Python packages and dependencies.
- `tests/`: Contains unit tests for validating the pruning process.

## Dependencies

To run this project, you will need the following dependencies:

- Python 3.6+
- PyTorch 1.10+
- NumPy
- tqdm
- (Optional) CUDA for GPU acceleration

Install the dependencies by running:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/iterative-magnitude-pruning.git
cd iterative-magnitude-pruning
```

2. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure to download the MNIST dataset. The `example.py` script will automatically download the dataset if it's not already present in the `data/` directory.

## Usage

### Training and Pruning

You can use the `example.py` script to train and prune a model using the iterative pruning algorithm.

```bash
python example.py
```

### Running the Pruning Process

In the `example.py` file, the following code initializes the `IterativeMagnitudePruning` class and runs the pruning process:

```python
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from pyPrune.models.LeNet import LeNet
from pyPrune.pruning import IterativeMagnitudePruning

# Define data loaders (MNIST)
train_loader = DataLoader(...)
test_loader = DataLoader(...)

# Initialize model, optimizer, and loss function
model = LeNet()
optimizer = Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Define pruning parameters
final_sparsity = 0.9
steps = 10
finetune_epochs = 2
pretrain_epochs = 5

# Create pruning instance
pruning = IterativeMagnitudePruning(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    final_sparsity=final_sparsity,
    steps=steps,
    optimizer=optimizer,
    criterion=criterion,
    finetune_epochs=finetune_epochs,
    pretrain_epochs=pretrain_epochs
)

# Run the pruning process
pruning.run()
```

### Running with Docker (GPU Support)

You can run the project inside a Docker container with GPU support. This is especially useful if you have an NVIDIA GPU and want to leverage CUDA for faster training and pruning.

To run the project with Docker, execute the following command:

```bash
docker run --rm --gpus all -it --runtime=nvidia -v $(pwd):/workspace pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
```

### Explanation of the Command:

- `--rm`: Automatically remove the container when it exits.
- `--gpus all`: Allow the container to use all available GPUs on the host machine.
- `-it`: Run the container interactively.
- `--runtime=nvidia`: Use the NVIDIA runtime for Docker (needed for GPU support).
- `-v $(pwd):/workspace`: Mount the current directory (`$(pwd)`) to `/workspace` in the container. This allows the Docker container to access the project files.
- `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`: The Docker image to use, which contains PyTorch with CUDA 12.4 and cuDNN 9 support.

Once inside the container, you can run the pruning and training script as described above. You will have access to all the necessary dependencies in the container, including PyTorch and CUDA.

## Metrics

During the pruning process, several metrics are tracked:

- **Sparsity**: The percentage of zeroed-out weights in the model.
- **Loss**: The loss value during training and evaluation.
- **Accuracy**: The accuracy of the model on the test set.
- **Gradients**: The gradients of the model parameters.
- **Optimizer State**: Information about the optimizer state.

These metrics are saved in `pruning_metrics.json` at the end of the pruning process.

## Logging

The training and pruning process is logged with detailed information about each step. Logs are saved to the `logs/` directory by default. You can change the logging configuration by modifying the `setup_logging()` function in the code.

Log entries include:

- Training and evaluation loss/accuracy.
- Sparsity after each pruning step.
- Checkpoint saving and loading information.

## Testing

We have included a basic set of unit tests located in the `tests/` directory. To run the tests, you can use pytest:

```bash
pytest tests/
```

## Contributing

We welcome contributions! If you find any bugs or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This README now includes all the essential details about setup, usage, and running the pruning process both with and without Docker, along with other key information like metrics and logging. Let me know if you need any additional adjustments!