import os
import argparse
import torch
from pyPrune.models.Vgg16 import VGG16
from experiments import *
from utils import *
from pyPrune.utils import *

# Dataset-related paths and filenames
CHECKPOINT_BASES = {
    "Cifar10": "../structured_study/pruning_checkpoints/Vgg16_datasetcifar10_pretrain30_finetune10_steps21_batch2048_devicecuda_strategy_magnitude/",
    "Cifar100": "../structured_study/pruning_checkpoints/Vgg16_datasetcifar100_pretrain30_finetune10_steps21_batch2048_devicecuda_strategy_magnitude/",
    "TinyImageNet": "../structured_study/pruning_checkpoints/Vgg16_datasettinyimagenet_pretrain30_finetune10_steps21_batch2048_devicecuda_strategy_magnitude/",
    "ImageNet": "../structured_study/pruning_checkpoints/Vgg16_datasetimagenet_pretrain30_finetune30_steps21_batch512_devicecuda_strategy_magnitude/",
}

CHECKPOINT_FILES = {
    "Cifar10": ("checkpoint_Finetuned_0.832228.pth", "checkpoint_Original_0.000000.pth"),
    "Cifar100": ("checkpoint_Finetuned_0.832228.pth", "checkpoint_Original_0.000000.pth"),
    "TinyImageNet": ("checkpoint_Finetuned_0.000000.pth", "checkpoint_Original_0.000000.pth"),
    "ImageNet": ("checkpoint_Finetuned_0.865782.pth", "checkpoint_Original_0.000000.pth"),
}

EXPERIMENTS = {
    "Cifar10": {
        "Original Model": None,
        "Last 2": ('conv_12', 'conv_13'),
        "Stage 5": ('conv_11', 'conv_13'),
        "Stage 4": ('conv_8', 'conv_11'),
        "Stage 3": ('conv_5', 'conv_8'),
        "Stage 4-5": ('conv_8', 'conv_13'),
        "Stage 3-5": ('conv_5', 'conv_13'),
        "Stage 2-5": ('conv_3', 'conv_13'),
    },
    "Cifar100": {
        "Original Model": None,
        "Last 2": ('conv_12', 'conv_13'),
        "Stage 5": ('conv_11', 'conv_13'),
        "Stage 4": ('conv_8', 'conv_11'),
        "Stage 4-5": ('conv_8', 'conv_13'),
    },
    "TinyImageNet": {
        "Original Model": None,
        "All Conv Layers": ('conv_1', 'conv_13'),
    },
    "ImageNet": {
        "Original Model": None,
        "Last 2": ('conv_12', 'conv_13'),
        "Stage 5": ('conv_11', 'conv_13'),
        "Stage 4": ('conv_8', 'conv_11'),
        "All Conv Layers": ('conv_1', 'conv_13'),
    },
}

def run_experiments_for_dataset(experiments, dataset, model_path_097, model_path_000, train_loader, test_loader, device, epochs, pretrain, model_class, model_kwargs, input_size, post_compress_epochs):
    """Run all experiments for a given dataset."""
    for name, layers in experiments.items():
        print(f"\n--- Running experiment: {name} ---")
        run_jf_experiment({name: layers}, model_path_097, train_loader, test_loader, device, epochs, pretrain,
                          model_class=model_class, model_kwargs=model_kwargs, data_shape=input_size, save_path=dataset,
                          post_compress_epochs=post_compress_epochs)

        run_kevin_experiment({name: layers}, model_path_000, train_loader, test_loader, device, epochs,
                             model_class=model_class, model_kwargs=model_kwargs, data_shape=input_size,
                             save_path=dataset, post_compress_epochs=post_compress_epochs)

        run_nick_experiment({name: layers}, model_path_000, train_loader, test_loader, device, epochs, pretrain,
                             model_class=model_class, model_kwargs=model_kwargs, data_shape=input_size,
                             save_path=dataset, post_compress_epochs=post_compress_epochs)

def main(model_path_097, model_path_000, experiments, dataset, epochs=30, pretrain=10, model_class=VGG16, model_kwargs=None, post_compress_epochs=True, run_all=False):
    """Main routine to load dataset, run experiments, and plot results."""
    if not os.path.exists(model_path_097):
        print(f"Model path {model_path_097} does not exist.")
        return
    if not os.path.exists(model_path_000):
        print(f"Model path {model_path_000} does not exist.")
        return

    print(f"\n=== Running on dataset: {dataset} | post_compress: {post_compress_epochs} ===")

    # Load dataset
    train_loader, test_loader, input_size, input_channels, default_num_classes = load_dataset(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model kwargs if not provided
    model_kwargs = model_kwargs or {
        'num_classes': default_num_classes,
        'input_size': input_size,
        'input_channels': input_channels
    }

    if run_all:
        # Run all experiments
        print(f"\n--- Running all experiments ---")
        run_experiments_for_dataset(experiments, dataset, model_path_097, model_path_000, train_loader, test_loader, device, epochs, pretrain, model_class, model_kwargs, input_size, post_compress_epochs)
    else:
        run_experiments_for_dataset(experiments, dataset, model_path_097, model_path_000, train_loader, test_loader, device, epochs, pretrain, model_class, model_kwargs, input_size, post_compress_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=False, default="Cifar10", choices=["Cifar10", "Cifar100", "TinyImageNet", "ImageNet"])
    parser.add_argument("--experiment", type=str, help="Name of specific experiment to run", default=None)
    parser.add_argument("--post_compress", action="store_true", help="Use post-compression epochs", default=True)
    args = parser.parse_args()

    dataset = args.dataset
    experiment_name = args.experiment
    post_compress_epochs = args.post_compress

    # Load paths
    base_path = CHECKPOINT_BASES[dataset]
    model_path_097 = os.path.join(base_path, CHECKPOINT_FILES[dataset][0])
    model_path_000 = os.path.join(base_path, CHECKPOINT_FILES[dataset][1])

    # Load experiments based on provided argument
    experiment_dict = EXPERIMENTS[dataset]
    if experiment_name:
        if experiment_name not in experiment_dict:
            raise ValueError(f"Experiment '{experiment_name}' not found for dataset '{dataset}'")
        experiment_dict = {experiment_name: experiment_dict[experiment_name]}

    print(f"Experiments to run: {list(experiment_dict.keys())}")

    # Run main function
    run_all = experiment_name is None  # if no experiment is specified, run all
    main(model_path_097, model_path_000, experiments=experiment_dict, dataset=dataset, post_compress_epochs=post_compress_epochs, run_all=run_all)
