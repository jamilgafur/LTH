import os
import glob
import pickle
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from pyPrune.utils import get_pruneable_named_parameters, lr_lambda
from matplotlib import ticker
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


def poly_lr_with_warmup(args, epoch):
    warmup_epochs = args.pretrain_epochs // 10
    max_epochs = args.pretrain_epochs + args.finetune_epochs
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    decay_epochs = max_epochs - warmup_epochs
    decay_progress = (epoch - warmup_epochs) / decay_epochs
    return (1 - decay_progress) ** 2


def load_pickle(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    with open(file_path, 'rb') as f:
        pruner = pickle.load(f)
    if pruner is None:
        raise ValueError(f"Failed to load pruner from {file_path}")
    return pruner


def get_sorted_model_paths(directory):
    model_paths = glob.glob(os.path.join(directory, "*Fine*.pth"))
    model_paths.sort(key=lambda x: float(x.split("_")[-1].split(".")[0]))
    return model_paths


def extract_layer_name(name):
    # Use original logic for extracting a simplified layer name
    if name.count('.') == 1:
        return name.split('.')[0]
    elif name.count('.') == 2:
        return name.split('.')[1]
    elif name.count('.') == 3:
        return f"{name.split('.')[1]}-{name.split('.')[2]}"
    elif "stage" in name:
        return f"{name.split('.')[1]}-{name.split('.')[3]}"
    else:
        return name  # fallback


def get_layer_information(pruner, model_path, base_params=None):
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
    try:
        accuracy = pruner.metrics['accuracy_finetune']
        loss = pruner.metrics['loss_finetune']
        sparsity = pruner.metrics['step_finetune']
    except KeyError as e:
        print(f"Missing key in pruner metrics: {e}")
        return

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))


    
    axs[0].plot(1-np.array(sparsity), 1-(np.array(accuracy)/100), 'bo-', label='Accuracy')
    axs[0].set_title('Accuracy vs Sparsity')
    axs[0].set_xlabel('Sparsity')
    axs[0].set_ylabel('Accuracy')
    axs[0].grid(True)
    axs[0].legend()
    # invert x axis
    axs[0].invert_xaxis()
    axs[0].invert_yaxis()
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5, subs=[1.0, .5]))
    axs[0].yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5, subs=[1.0, .5]))
    
 

    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].plot(1-np.array(sparsity), loss, 'ro-', label='Loss')
    axs[1].set_title('Loss vs Sparsity')
    axs[1].set_xlabel('Sparsity')
    axs[1].set_ylabel('Loss')
    axs[1].grid(True)
    axs[1].legend()
    axs[1].invert_xaxis()
    axs[1].xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5, subs=[1.0, .5]))
    axs[1].yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5, subs=[1.0, .5]))


    def fractional_log_fmt(x, pos):
        from fractions import Fraction
        
        if x == 0:
            return "$1$"    

        if x == 1:
            return "$0$"
        

        expo = np.log10(x)

        frac = Fraction(expo).limit_denominator(2)  # only allow halves

    
        if frac.denominator == 1:

            return rf"$1 - 10^{{{frac.numerator}}}$"

        else:

            return rf"$1 - 10^{{{frac.numerator}/{frac.denominator}}}$"
    
    from matplotlib.ticker import FuncFormatter
    axs[0].xaxis.set_major_formatter(FuncFormatter(fractional_log_fmt))
    axs[1].xaxis.set_major_formatter(FuncFormatter(fractional_log_fmt))
    

    
    os.makedirs(path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(path, "accuracy_loss.png"), bbox_inches='tight')
    plt.close(fig)


def plot_layer_information(savedir, data):
    if not data:
        print("No data available for plotting.")
        return

    os.makedirs(savedir, exist_ok=True)
    sparsities = sorted(data.keys())
    layers = list(data[sparsities[0]]["layer_info"].keys())

    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    line_styles = ['-', '--', '-.', ':']

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

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
        axes[0].plot(1-np.array(sparsities), x1, label=layer, color=color, linestyle=style)
        axes[1].plot(1-np.array(sparsities), x2, label=layer, color=color, linestyle=style)

    axes[0].set_xscale('log')
    axes[1].set_xscale('log')
    axes[0].set_title('Zeros / Trainable Params in Layer')
    axes[0].set_xlabel('Density')
    axes[0].set_ylabel('Ratio')
    axes[0].legend()

    axes[1].set_title('Zeros / Total Trainable Params')
    axes[1].set_xlabel('Density')
    axes[1].set_ylabel('Ratio')
    axes[1].legend()

    plt.tight_layout()
    print(f"saving to {os.path.join(savedir, 'layer_info.png')}")
    plt.savefig(os.path.join(savedir, "layer_info.png"), bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    base_output_dir = "./plots"
    model_dirs = glob.glob("/scratch/jgafur/LTH_output/*")

    for model_directory in model_dirs:
        print(f"\nProcessing: {model_directory}")
        pruner_pickle_path = os.path.join(model_directory, "pruner.pkl")
        directory_name = model_directory.split("/")[-2]

        if not os.path.isfile(pruner_pickle_path):
            print(f"Missing pruner file: {pruner_pickle_path}")
            continue

        try:
            pruner = load_pickle(pruner_pickle_path)
        except Exception as e:
            print(f"Error loading pruner: {e}")
            continue

        output_dir = os.path.join(base_output_dir, os.path.basename(model_directory))
        plot_accuracy(pruner, output_dir+f"/{directory_name}")

        model_paths = get_sorted_model_paths(model_directory)
        if not model_paths:
            print("No model checkpoints found.")
            continue

        data = {}
        base_params = None

        for model_path in model_paths:
            try:
                sparsity = float("0." + model_path.split('_')[-1].split('.')[1])
                print(f"   Model: {os.path.basename(model_path)} | Sparsity: {sparsity}")
                layer_info, base_params = get_layer_information(pruner, model_path, base_params)
                data[sparsity] = {"layer_info": layer_info}
            except Exception as e:
                print(f"    Skipping {model_path}: {e}")

        plot_layer_information(output_dir+f"/{directory_name}", data)