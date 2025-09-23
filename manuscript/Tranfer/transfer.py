import os
import torch
import matplotlib.pyplot as plt

from pyPrune.models.Vgg16 import VGG16
from experiments import *
from utils import *
from pyPrune.utils import *



def main(model_path_097, model_path_000, experiments=None, epochs=0, pretrain=0,
         model_class=VGG16, model_kwargs=None, dataset="Cifar10", post_compress_epochs=False):
    """
    Main routine to load dataset, run experiments, and plot results.
    """
    if not os.path.exists(model_path_097) or not os.path.exists(model_path_000):
        print("Required model weight files not found. Exiting.")
        return

    # Load dataset and determine input size and classes
    if dataset == "TinyImageNet":
        print("Loading Tiny ImageNet data...")
        train_loader, test_loader = load_tiny_imagenet()
        sample_input = next(iter(train_loader))[0]
        input_size = sample_input.shape[-2:]
        input_channels = sample_input.shape[1]
        default_num_classes = 200

    elif dataset == "Cifar100":
        print("Loading CIFAR-100 data...")
        train_loader, test_loader = load_cifar100()
        sample_input = next(iter(train_loader))[0]
        input_size = sample_input.shape[-2:]
        input_channels = sample_input.shape[1]
        default_num_classes = 100

    elif dataset == "Cifar10":
        print("Loading CIFAR-10 data...")
        train_loader, test_loader = load_cifar10()
        sample_input = next(iter(train_loader))[0]
        input_size = sample_input.shape[-2:]
        input_channels = sample_input.shape[1]
        default_num_classes = 10

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Override model kwargs if not provided
    if model_kwargs is None:
        model_kwargs = {
            'num_classes': default_num_classes,
            'input_size': input_size,
            'input_channels': input_channels
        }

    # Run experiments
    jf_data = run_jf_experiment(experiments, model_path_097, train_loader, test_loader, device,
                                epochs, pretrain, model_class=model_class, model_kwargs=model_kwargs, data_shape=input_size, save_path=dataset, post_compress_epochs=post_compress_epochs)

    nick_data = run_nick_experiment(experiments, model_path_000, train_loader, test_loader, device,
                                    epochs, pretrain, model_class=model_class, model_kwargs=model_kwargs, data_shape=input_size, save_path=dataset, post_compress_epochs=post_compress_epochs)

    kevin_data = run_kevin_experiment(experiments, model_path_000, train_loader, test_loader, device,
                                      epochs, pretrain, model_class=model_class, model_kwargs=model_kwargs, data_shape=input_size, save_path=dataset, post_compress_epochs=post_compress_epochs)

def runCifar100(post_compress_epochs):
    base = "../structured_study/pruning_checkpoints/Vgg16_datasetcifar100_pretrain1_finetune1_steps21_batch512_devicecuda_strategy_magnitude/"
    model_path_097 = os.path.join(base, "checkpoint_Finetuned_0.67.pth")
    model_path_000 = os.path.join(base, "checkpoint_Original_0.00.pth")

    experiments = {
        "Original Model": None,
        "Stage 4-5": ('conv_8', 'conv_13'),
        "Stage 2-5": ('conv_3', 'conv_13'),
        # "All Conv Layers": ('conv_1', 'conv_13'),
    }

    main(model_path_097, model_path_000, experiments=experiments, epochs=0, pretrain=0,
         model_class=VGG16, model_kwargs=None, dataset="Cifar100", post_compress_epochs=post_compress_epochs)


def runTinyImageNet(post_compress_epochs):
    base = "../structured_study/pruning_checkpoints/Vgg16_datasettinyimagenet_pretrain1_finetune1_steps21_batch512_devicecuda_strategy_magnitude/"
    model_path_097 = os.path.join(base, "checkpoint_Finetuned_0.74.pth")
    model_path_000 = os.path.join(base, "checkpoint_Original_0.00.pth")

    experiments = {
        # "Original Model": None,
        "Stage 4-5": ('conv_8', 'conv_13'),
        # "Stage 2-5": ('conv_3', 'conv_13'),
        # "All Conv Layers": ('conv_1', 'conv_13'),
    }

    main(model_path_097, model_path_000, experiments=experiments, epochs=0, pretrain=0,
         model_class=VGG16, model_kwargs=None, dataset="TinyImageNet", post_compress_epochs=post_compress_epochs)


if __name__ == "__main__":
    post_compress_epochs = False
    runCifar100(post_compress_epochs)
    # runTinyImageNet(post_compress_epochs)
