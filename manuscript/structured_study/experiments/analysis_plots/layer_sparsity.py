import os
import glob
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from matplotlib import ticker
from pyPrune.utils import get_pruneable_named_parameters

# Consolidated imports
sns.set(style="whitegrid", context="paper", font_scale=1.5)

def poly_lr_with_warmup(args, epoch):
    warmup_epochs = args.pretrain_epochs // 10
    max_epochs = args.pretrain_epochs + args.finetune_epochs
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    decay_epochs = max_epochs - warmup_epochs
    decay_progress = (epoch - warmup_epochs) / decay_epochs
    return (1 - decay_progress) ** 2

def load_pickle(file_path):
    """ Load pruner pickle file """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    with open(file_path, 'rb') as f:
        pruner = pickle.load(f)
    if pruner is None:
        raise ValueError(f"Failed to load pruner from {file_path}")
    return pruner

def get_sorted_model_paths(directory):
    """ Get sorted model paths based on the checkpoint names """
    model_paths = glob.glob(os.path.join(directory, "*Fine*.pth"))
    model_paths.sort(key=lambda x: float(x.split("_")[-1].split(".")[0]))
    return model_paths

def extract_layer_name(name):
    """ Extract simplified layer name from the parameter name """
    if name.count('.') == 1:
        return name.split('.')[0]
    elif name.count('.') == 2:
        return name.split('.')[1]
    elif name.count('.') == 3:
        return f"{name.split('.')[1]}-{name.split('.')[2]}"
    elif "stage" in name:
        return f"{name.split('.')[1]}-{name.split('.')[3]}"
    else:
        return name

def get_layer_information(pruner, model_path, base_params=None):
    """ Extract layer information (sparsity, number of zeros) from model """
    checkpoint = torch.load(model_path, map_location='cpu')
    pruner.model.load_state_dict(checkpoint['model'])
    pruner.model.eval()

    layer_info = defaultdict(lambda: {
        "num_zeros": 0, "num_trainable_params": 0, "total_trainable_params": 0
    })
    
    if base_params is None:
        base_params = {}

    names, params = get_pruneable_named_parameters(pruner.model, pruner.prunable_layers)

    for name, param in zip(names, params):
        layer_name = extract_layer_name(name)
        num_zeros = torch.count_nonzero(param == 0).item()
        num_params = param.numel()

        if layer_name not in base_params:
            base_params[layer_name] = num_params

        layer_info[layer_name]["num_zeros"] += num_zeros
        layer_info[layer_name]["num_trainable_params"] += num_params
        layer_info[layer_name]["total_trainable_params"] = base_params[layer_name]

    return dict(layer_info), base_params

def plot_accuracy(pruner, path):
    """ Plot accuracy and loss against sparsity """
    try:
        accuracy = pruner.metrics['accuracy_finetune']
        loss = pruner.metrics['loss_finetune']
        sparsity = pruner.metrics['step_finetune']
    except KeyError as e:
        print(f"Missing key in pruner metrics: {e}")
        return

    sparsity_percent = (1 - np.array(sparsity)) * 100
    acc = 1 - (np.array(accuracy) / 100)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axs[0].plot(sparsity_percent, acc, 'bo-', label='Accuracy (1 - acc)')
    axs[0].set_ylabel('Accuracy (lower is better)')
    axs[0].invert_yaxis()
    axs[0].legend(loc='best')

    axs[1].plot(sparsity_percent, loss, 'ro-', label='Loss')
    axs[1].set_xlabel('Sparsity (%)')
    axs[1].set_ylabel('Loss (lower is better)')
    axs[1].legend(loc='best')
    axs[1].invert_xaxis()

    axs[1].xaxis.set_major_formatter(ticker.PercentFormatter())
    for ax in axs:
        ax.grid(True)
        ax.set_xscale('linear')
        ax.set_yscale('log')

    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, "accuracy_loss.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved accuracy/loss figure to {save_path}")

def plot_layer_information(savedir, data):
    """ Plot layer-wise sparsity and layer contribution to global sparsity """
    if not data:
        print("No data available for plotting.")
        return

    os.makedirs(savedir, exist_ok=True)

    sparsities = sorted(data.keys())
    sparsity_percent = (1 - np.array(sparsities)) * 100
    layers = list(data[sparsities[0]]["layer_info"].keys())

    colors = plt.cm.tab20(np.linspace(0, 1, len(layers)))
    line_styles = ['-', '--', '-.', ':']

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for idx, (layer, color) in enumerate(zip(layers, colors)):
        x1, x2 = [], []
        for sparsity in sparsities:
            info = data[sparsity]['layer_info'][layer]
            total_params = sum(
                l["total_trainable_params"] for l in data[sparsity]["layer_info"].values()
            )
            x1.append(info['num_zeros'] / info['num_trainable_params'])
            x2.append(info['num_zeros'] / total_params)

        style = line_styles[idx % len(line_styles)]
        axes[0].plot(sparsity_percent, x1, label=layer, color=color, linestyle=style)
        axes[1].plot(sparsity_percent, x2, label=layer, color=color, linestyle=style)

    axes[0].set_ylabel('Zeros / Trainable Params in Layer')
    axes[0].set_title('Layer-wise Sparsity')
    axes[0].invert_xaxis()
    axes[0].grid(True)

    axes[1].set_ylabel('Zeros / Total Trainable Params')
    axes[1].set_xlabel('Sparsity (%)')
    axes[1].set_title('Layer Contribution to Global Sparsity')
    axes[1].invert_xaxis()
    axes[1].grid(True)

    axes[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    save_path = os.path.join(savedir, "layer_info.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved updated layer information figure to {save_path}")

def plot_accuracy_all_models(pruner_dict, save_path):
    """ Plot accuracy and loss for all models """
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for model_name, pruner in pruner_dict.items():
        try:
            accuracy = pruner.metrics['accuracy_finetune']
            loss = pruner.metrics['loss_finetune']
            sparsity = pruner.metrics['step_finetune']
        except KeyError as e:
            print(f"Missing key for {model_name}: {e}")
            continue

        sparsity_percent = (1 - np.array(sparsity)) * 100
        acc = 1 - (np.array(accuracy) / 100)

        axs[0].plot(sparsity_percent, acc, marker='o', label=model_name)
        axs[1].plot(sparsity_percent, loss, marker='o', label=model_name)

    axs[0].set_ylabel('1 - Accuracy')
    axs[0].invert_yaxis()
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_xlabel('Sparsity (%)')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].grid(True)
    axs[0].invert_xaxis()
    axs[1].invert_xaxis()

    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[1].xaxis.set_major_formatter(ticker.PercentFormatter())

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, "accuracy_loss_all_models.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved accuracy/loss plot for all models to {save_path}")

def plot_layer_information_all_models(savedir, data_dict):
    """ Plot layer information for all models """
    os.makedirs(savedir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12*5, 10*5), sharex=True)

    for model_name, data in data_dict.items():
        if not data:
            continue

        sparsities = sorted(data.keys())
        sparsity_percent = (1 - np.array(sparsities)) * 100
        layers = list(data[sparsities[0]]["layer_info"].keys())

        for layer in layers:
            x1, x2 = [], []
            for sparsity in sparsities:
                info = data[sparsity]['layer_info'][layer]
                total_params = sum(
                    l["total_trainable_params"] for l in data[sparsity]["layer_info"].values()
                )
                x1.append(info['num_zeros'] / info['num_trainable_params'])
                x2.append(info['num_zeros'] / total_params)

            axes[0].plot(sparsity_percent, x1, label=f"{model_name}: {layer}", alpha=0.7)
            axes[1].plot(sparsity_percent, x2, label=f"{model_name}: {layer}", alpha=0.7)

    axes[0].set_ylabel('Zeros / Trainable Params')
    axes[0].invert_xaxis()
    axes[0].grid(True)
    axes[0].set_title('Layer-wise Sparsity')

    axes[1].set_ylabel('Zeros / Total Trainable Params')
    axes[1].set_xlabel('Sparsity (%)')
    axes[1].invert_xaxis()
    axes[1].grid(True)
    axes[1].set_title('Layer Contribution to Global Sparsity')

    axes[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(os.path.join(savedir, "layer_info_all_models.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved layer information plot for all models to {savedir}")

if __name__ == "__main__":
    base_output_dir = "./plots"
    model_dirs = glob.glob("/scratch/jgafur/LTH_output/*LeNet*finetune5*magnitude*")

    all_pruners = {}
    all_layer_data = {}

    for model_directory in model_dirs:
        print(f"\nProcessing: {model_directory}")
        pruner_pickle_path = os.path.join(model_directory, "pruner.pkl")
        model_name = os.path.basename(model_directory)

        if not os.path.isfile(pruner_pickle_path):
            print(f"Missing pruner file: {pruner_pickle_path}")
            continue

        try:
            pruner = load_pickle(pruner_pickle_path)
            all_pruners[model_name] = pruner
        except Exception as e:
            print(f"Error loading pruner: {e}")
            continue

        model_paths = get_sorted_model_paths(model_directory)
        if not model_paths:
            print("No model checkpoints found.")
            continue

        data = {}
        base_params = None

        for model_path in model_paths:
            try:
                sparsity = float("0." + model_path.split('_')[-1].split('.')[1])
                layer_info, base_params = get_layer_information(pruner, model_path, base_params)
                data[sparsity] = {"layer_info": layer_info}
            except Exception as e:
                print(f"    Skipping {model_path}: {e}")
        all_layer_data[model_name] = data

    plot_accuracy_all_models(all_pruners, base_output_dir)
    plot_layer_information_all_models(base_output_dir, all_layer_data)
