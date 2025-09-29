import os
import torch
from pyPrune.models.Vgg16 import VGG16
from pyPrune.pruneMethods.IterativePruner import IterativePruner
from pyPrune.strategies import MagnitudePruningStrategy
from experiments import *
from utils import *
from pyPrune.utils import *

# set seed for reproducibility
from torch.backends import cudnn
import random
import numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
cudnn.deterministic = True
cudnn.benchmark = False

CHECKPOINT_BASES = {
    "Cifar10": "../structured_study/pruning_checkpoints/Vgg16_datasetcifar10_pretrain30_finetune10_steps21_batch2048_devicecuda_strategy_magnitude/",
    "Cifar100": "../structured_study/pruning_checkpoints/Vgg16_datasetcifar100_pretrain30_finetune10_steps21_batch2048_devicecuda_strategy_magnitude/",
    "TinyImageNet": "../structured_study/pruning_checkpoints/Vgg16_datasettinyimagenet_pretrain30_finetune10_steps21_batch2048_devicecuda_strategy_magnitude/",
    "ImageNet": "../structured_study/pruning_checkpoints/Vgg16_datasetimagenet_pretrain30_finetune10_steps21_batch1024_devicecuda_strategy_magnitude/",
}

CHECKPOINT_FILES = {
    "Cifar10": ("checkpoint_Finetuned_0.945024.pth", "checkpoint_Original_0.000000.pth"),
    "Cifar100": ("checkpoint_Finetuned_0.914101.pth", "checkpoint_Original_0.000000.pth"),
    "TinyImageNet": ("checkpoint_Finetuned_0.971853.pth", "checkpoint_Original_0.000000.pth"),
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
        "Last 2": ('conv_12', 'conv_13'),
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

# -----------------------------
# IMP Pruning Logic Integration
# -----------------------------
def imp_prune(model, optimizer, scheduler, criterion, train_loader, test_loader, steps, pretrain_epochs, finetune_epochs, device, save_dir, strategy, patience, experiment_name=None):
    """
    Function to run pruning based on the IterativePruner class.
    """
    save_dir = os.path.join(save_dir, experiment_name)

    print("\nModel:", model)
    print("Optimizer:", optimizer)
    print("Scheduler:", scheduler)
    print("Criterion:", criterion)
    print("Train Loader:", train_loader)
    print("Test Loader:", test_loader)
    print("Steps:", steps)
    print("Pretrain Epochs:", pretrain_epochs)
    print("Device:", device)
    print("Finetune Epochs:", finetune_epochs)
    print("Save Dir:", save_dir)
    print("Strategy:", strategy)
    # Initialize the IterativePruner
    pruner = IterativePruner(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        steps=steps,
        device=device,
        save_dir=save_dir,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=10,
        scheduler=scheduler,
        strategy=MagnitudePruningStrategy.MagnitudePruningStrategy(device=device)
    )

    print("Running pruning process...")
    pruner.run()  # Start pruning process
    print("Pruning process complete.")
    
    return pruner

def run_experiments_for_dataset(experiments, dataset, model_path_097, model_path_000, train_loader, test_loader, device, epochs, pretrain, model_class, model_kwargs, input_size, post_compress_epochs, run_all, experiment_func):
    """Run all experiments for a given dataset."""
    save_path = f"{dataset}_{CHECKPOINT_FILES[dataset][0]}_epochs{epochs}_pretrain{pretrain}_postcompress{post_compress_epochs}"
    
    # Define optimizer, scheduler, and criterion
    def create_optimizer_scheduler(model, learning_rate=1e-3):
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return optimizer, scheduler

    criterion = torch.nn.CrossEntropyLoss()

    # Define the number of pruning steps (this is arbitrary, you can adjust it)
    steps = exponential_decay_list(steps=21)
    # steps get every thrd step  including the last one
    steps = [steps[i] for i in range(len(steps)) if i % 3 == 0 or i == len(steps) - 1]
    print(f"Pruning steps: {steps}")
    if run_all:
        experiment_func(experiments, model_path_097, train_loader, test_loader, device, epochs, pretrain,
                        model_class=model_class, model_kwargs=model_kwargs, data_shape=input_size, save_path=save_path,
                        post_compress_epochs=post_compress_epochs)     
        experiment_func(experiments, model_path_000, train_loader, test_loader, device, epochs,
                         model_class=model_class, model_kwargs=model_kwargs, data_shape=input_size,
                            save_path=save_path, post_compress_epochs=post_compress_epochs)
    else:
        for name, layers in experiments.items():
            print(f"\n--- Running experiment: {name} ---")
            if args.JF:
                model = run_jf_experiment({name: layers}, model_path_097, train_loader, test_loader, device, epochs, pretrain,
                                        model_class=model_class, model_kwargs=model_kwargs, data_shape=input_size, save_path=save_path,
                                        post_compress_epochs=post_compress_epochs)
                if args.imp:
                    optimizer, scheduler = create_optimizer_scheduler(model)
                    imp_prune(model, optimizer, scheduler, criterion, train_loader, test_loader, steps, pretrain_epochs=pretrain, finetune_epochs=pretrain, device=device, save_dir=save_path, strategy="magnitude", patience=5, experiment_name=name)

            elif args.Kevin:
                model = run_kevin_experiment({name: layers}, model_path_000, train_loader, test_loader, device, epochs,
                                    model_class=model_class, model_kwargs=model_kwargs, data_shape=input_size,
                                    save_path=save_path, post_compress_epochs=post_compress_epochs)
                if args.imp:
                    optimizer, scheduler = create_optimizer_scheduler(model)
                    imp_prune(model, optimizer, scheduler, criterion, train_loader, test_loader, steps, pretrain_epochs=pretrain, finetune_epochs=pretrain, device=device, save_dir=save_path, strategy="magnitude", patience=5, experiment_name=name)
            elif args.Nick:
                model = run_nick_experiment({name: layers}, model_path_000, train_loader, test_loader, device,
                                        epochs, pretrain,
                                        model_class=model_class, model_kwargs=model_kwargs, data_shape=input_size,
                                        save_path=save_path, post_compress_epochs=post_compress_epochs)
                if args.imp:
                    optimizer, scheduler = create_optimizer_scheduler(model)
                    imp_prune(model, optimizer, scheduler, criterion, train_loader, test_loader, steps, pretrain_epochs=pretrain, finetune_epochs=pretrain, device=device, save_dir=save_path, strategy="magnitude", patience=5, experiment_name=name)

def main(model_path_097, model_path_000, experiments, dataset, epochs=30, pretrain=10, model_class=VGG16, model_kwargs=None, post_compress_epochs=True, run_all=False, experiment_func=None):
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
    device = "cuda"

    # Model kwargs if not provided
    model_kwargs = model_kwargs or {
        'num_classes': default_num_classes,
        'input_size': input_size,
        'input_channels': input_channels
    }

    run_experiments_for_dataset(experiments, dataset, model_path_097, model_path_000, train_loader, test_loader, device, epochs, pretrain, model_class, model_kwargs, input_size, post_compress_epochs, run_all, experiment_func)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=False, default="Cifar10", choices=["Cifar10", "Cifar100", "TinyImageNet", "ImageNet"])
    parser.add_argument("--experiment", type=str, help="Name of specific experiment to run", default=None)
    parser.add_argument("--post_compress", action="store_true", help="Use post-compression epochs", default=True)
    parser.add_argument("--imp", action="store_true", help="Use IMP pruning", default=True)
    parser.add_argument("--JF", action="store_true", help="Run JF experiment", default=True)
    parser.add_argument("--Kevin", action="store_true", help="Run Kevin experiment", default=False)
    parser.add_argument("--Nick", action="store_true", help="Run Nick experiment", default=False)
    args = parser.parse_args()

    dataset = args.dataset
    experiment_name = args.experiment
    post_compress_epochs = args.post_compress
    imp = args.imp
    if args.JF:
        experiment_func_name = "run_jf_experiment"
    elif args.Kevin:
        experiment_func_name = "run_kevin_experiment"
    elif args.Nick:
        experiment_func_name = "run_nick_experiment"
    else:
        raise ValueError("At least one of --JF, --Kevin, or --Nick must be specified.")

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

    # Map string to actual function
    experiment_func = globals()[experiment_func_name]

    # Run main function
    run_all = experiment_name is None  # if no experiment is specified, run all
    if run_all:
        print("Running all experiments.")
    main(model_path_097, model_path_000, experiments=experiment_dict, dataset=dataset, post_compress_epochs=post_compress_epochs, run_all=run_all, experiment_func=experiment_func)
