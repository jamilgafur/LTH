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
    if not os.path.exists(model_path_097):
        print(f"Model path {model_path_097} does not exist.")
        return
    if not os.path.exists(model_path_000):
        print(f"Model path {model_path_000} does not exist.")
        return

    print(f"Using dataset: {dataset}")
    train_loader, test_loader, input_size, input_channels, default_num_classes = load_dataset(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Override model kwargs if not provided
    if model_kwargs is None:
        model_kwargs = {
            'num_classes': default_num_classes,
            'input_size': input_size,
            'input_channels': input_channels
        }

    # # Run experiments
    jf_data = run_jf_experiment(experiments, model_path_097, train_loader, test_loader, device, epochs, pretrain, model_class=model_class, model_kwargs=model_kwargs, data_shape=input_size, save_path=dataset, post_compress_epochs=post_compress_epochs)

    kevin_data = run_kevin_experiment(experiments, model_path_000, train_loader, test_loader, device, epochs, model_class=model_class, model_kwargs=model_kwargs, data_shape=input_size, save_path=dataset, post_compress_epochs=post_compress_epochs)
    
    nick_data = run_nick_experiment(experiments, model_path_000, train_loader, test_loader, device, epochs, pretrain, model_class=model_class, model_kwargs=model_kwargs, data_shape=input_size, save_path=dataset, post_compress_epochs=post_compress_epochs)

    
def runCifar100(post_compress_epochs):
    base = "../structured_study/pruning_checkpoints/Vgg16_datasetcifar100_pretrain10_finetune30_steps21_batch512_devicecuda_strategy_magnitude"
    model_path_097 = os.path.join(base, "checkpoint_Finetuned_0.865782.pth")
    model_path_000 = os.path.join(base, "checkpoint_Original_0.000000.pth")
    experiments = {
        "Original Model": None,
        "Stage 4-5": ('conv_8', 'conv_13'),
        "Stage 2-5": ('conv_3', 'conv_13'),
        "All Conv Layers": ('conv_1', 'conv_13'),
    }

    main(model_path_097, model_path_000, experiments=experiments, epochs=30, pretrain=10,
         model_class=VGG16, model_kwargs=None, dataset="Cifar100", post_compress_epochs=post_compress_epochs)


def runCifar10(post_compress_epochs):
    base = "../structured_study/pruning_checkpoints/Vgg16_datasetcifar10_pretrain10_finetune30_steps21_batch512_devicecuda_strategy_magnitude"
    model_path_097 = os.path.join(base, "checkpoint_Finetuned_0.985588.pth")
    model_path_000 = os.path.join(base, "checkpoint_Original_0.000000.pth")
    experiments = {
        "Original Model": None,
        "Stage 4-5": ('conv_8', 'conv_13'),
        "Stage 2-5": ('conv_3', 'conv_13'),
        "All Conv Layers": ('conv_1', 'conv_13'),
    }

    main(model_path_097, model_path_000, experiments=experiments, epochs=30, pretrain=10,
         model_class=VGG16, model_kwargs=None, dataset="Cifar10", post_compress_epochs=post_compress_epochs)


def runTinyImageNet(post_compress_epochs):
    base = "../structured_study/pruning_checkpoints/Vgg16_datasetimagenet_pretrain10_finetune30_steps21_batch512_devicecuda_strategy_magnitude/"
    model_path_097 = os.path.join(base, "checkpoint_Finetuned_0.200000.pth")
    model_path_000 = os.path.join(base, "checkpoint_Original_0.000000.pth")

    experiments = {
        "Original Model": None,
        "Stage 4-5": ('conv_8', 'conv_13'),
        "Stage 2-5": ('conv_3', 'conv_13'),
        "All Conv Layers": ('conv_1', 'conv_13'),
    }

    main(model_path_097, model_path_000, experiments=experiments, epochs=0, pretrain=0, model_class=VGG16, model_kwargs=None, dataset="TinyImageNet", post_compress_epochs=post_compress_epochs)


if __name__ == "__main__":
    post_compress_epochs = True
    runCifar10(post_compress_epochs)
    runCifar100(post_compress_epochs)
    # runTinyImageNet(post_compress_epochs)
