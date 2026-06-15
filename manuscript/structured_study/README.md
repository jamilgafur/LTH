# README: Structured Study for Model Pruning and Experiments

This repository contains a structured setup for conducting model pruning experiments using different strategies and models. The experiments are performed on multiple architectures, including CIFAR and TinyImageNet-based models, with various pruning strategies such as magnitude pruning and optimal brain damage.

## Directory Structure

The directory structure for the codebase is as follows:

```
.
└── manuscript
    └── structured_study
        ├── experiment_job.sh       # SLURM job script for running individual experiments
        ├── main_experiment.py      # Main script to run pruning and experiments
        ├── prune_job.sh            # SLURM job script for pruning models
        ├── run_experiments.sh      # Script to run a batch of pruning experiments
        └── runall.sh               # Script to run all combinations of models and parameters
```

### File Overview

* **experiment\_job.sh**: This script handles individual experiment jobs, executing pruning and experimentation in parallel. It accepts parameters for the model, training epochs, strategy, and experiment type. After pruning completes, it submits the appropriate experiment jobs.

* **main\_experiment.py**: This is the main Python script for model training, pruning, and experiments. It initializes the model, pruning method, and various configurations based on user input via command-line arguments. The script also runs specific experiments like Neuron Similarity, Neuron Zeroing, and Weight Zeroing.

* **prune\_job.sh**: A SLURM batch script that handles the pruning process for a single model. It initializes the model, sets up the required configuration, and then runs the pruning. After pruning, it submits experiment jobs for further analysis.

* **run\_experiments.sh**: This bash script automates the execution of multiple pruning jobs for different models, strategies, and hyperparameters. It submits jobs for both CIFAR and TinyImageNet models using different configurations of epochs, pruning strategies, and steps.

* **runall.sh**: This is a comprehensive script that loops through all combinations of models, strategies, steps, pretraining epochs, and finetuning epochs. It then submits the pruning jobs using `prune_job.sh`.

## Running the Experiments

### 1. **Running Individual Experiments**

To run a specific experiment, you can submit jobs using `sbatch experiment_job.sh`. This will trigger an individual experiment based on the provided parameters (model, epochs, strategies, etc.). Example:

```bash
sbatch experiment_job.sh LeNet 5 3 10 21 /scratch/jgafur/LTH_output NeuronSimilarity 12345 5 brain-damage 128 0
```

### 2. **Running Pruning Jobs**

The main pruning job for each model can be executed using the `prune_job.sh` script. This script prepares the pruning environment for a given model, strategy, and hyperparameter combination. Example:

```bash
sbatch prune_job.sh LeNet 5 3 10 21 brain-damage 128
```

This will prune the model `LeNet` using the `brain-damage` strategy, with 5 pretraining epochs and 10 finetuning epochs.

### 3. **Batch Running of Experiments**

To automate the running of multiple experiments with varying parameters (such as different models, strategies, and training epochs), use `run_experiments.sh`:

```bash
bash run_experiments.sh
```

This script will loop through all the combinations of strategies and models, submitting pruning jobs to SLURM for each configuration.

### 4. **Running All Combinations**

For running all combinations of models, strategies, steps, and epochs, use `runall.sh`:

```bash
bash runall.sh
```

This script will go through all models and combinations of strategies, training epochs, and batch sizes, and submit them to SLURM for processing.

## Command-Line Arguments for `main_experiment.py`

The script `main_experiment.py` accepts the following arguments:

| Argument            | Description                                                    | Default                |
| ------------------- | -------------------------------------------------------------- | ---------------------- |
| `--model`           | Model architecture to use for pruning (e.g., LeNet, ResNet20). | `EfficientNet`         |
| `--experiments`     | List of experiments to run (e.g., NeuronSimilarity).           | `['None']`             |
| `--steps`           | Number of steps for pruning decay.                             | 21                     |
| `--pretrain_epochs` | Number of pretraining epochs.                                  | 1                      |
| `--finetune_epochs` | Number of finetuning epochs after pruning.                     | 1                      |
| `--device`          | Device for training (e.g., CPU, CUDA).                         | `cuda`                 |
| `--save_dir`        | Directory to save the pruning checkpoints.                     | `pruning_checkpoints/` |
| `--patience`        | Patience for early stopping during pruning.                    | 5                      |
| `--batch_size`      | Batch size for training.                                       | 2048                   |
| `--num_workers`     | Number of workers for data loading.                            | 1                      |
| `--strategy`        | Pruning strategy (magnitude or brain-damage).                  | `brain-damage`         |
| `--experimentStep`  | Step to process for neuron zeroing experiment.                 | 1                      |

### Example Command:

```bash
python main_experiment.py --model ResNet20 --experiments NeuronSimilarity NeuronZeroing --steps 21 --pretrain_epochs 5 --finetune_epochs 10 --batch_size 128 --strategy brain-damage
```

## SLURM Job Scripts

The job scripts are designed to be run on a SLURM-based cluster. They specify the necessary compute resources for the experiments.

* `experiment_job.sh`: A SLURM script for running individual experiments.
* `prune_job.sh`: A SLURM script for running pruning jobs.
* `run_experiments.sh`: A script for running batch jobs across multiple combinations of models and parameters.
* `runall.sh`: A comprehensive script for running all experiments.

### Example SLURM Commands:

```bash
sbatch experiment_job.sh LeNet 5 3 10 21 /scratch/jgafur/LTH_output NeuronSimilarity 12345 5 brain-damage 128 0
sbatch prune_job.sh LeNet 5 3 10 21 brain-damage 128
```

Each model is supported with different configurations for both CIFAR-10 and TinyImageNet datasets.

## License

This repository is licensed under the MIT License. See the LICENSE file for details.

---

For further assistance or questions, feel free to reach out.
