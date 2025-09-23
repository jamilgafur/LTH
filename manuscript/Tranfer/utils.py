import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import json


# -------------------------
# Result Plotting
# -------------------------
def plot_results(params, accs, names, title, filename, infer_times=None, mem_usages=None):
    fig, axs = plt.subplots(3, 1, figsize=(16, 18))

    axs[0].bar(names, accs, color='skyblue')
    axs[0].set_title(f"{title} - Final Accuracy (%)", fontsize=14)
    axs[0].set_ylabel("Accuracy (%)")
    axs[0].grid(True)

    ax0_twin = axs[0].twinx()
    ax0_twin.plot(names, params, 'ro--', label='Trainable Parameters (log)', linewidth=2)
    ax0_twin.set_ylabel('Trainable Parameters', color='red')
    ax0_twin.set_yscale('log')
    ax0_twin.tick_params(axis='y', colors='red')
    for i, param in enumerate(params):
        ax0_twin.annotate(f'{param:,}', xy=(i, param), xytext=(0, -15),
                          textcoords='offset points', ha='center', fontsize=9, color='red')

    axs[1].bar(names, infer_times, color='orange')
    axs[1].set_title("Inference Time (avg per batch in seconds)", fontsize=14)
    axs[1].set_ylabel("Time (s)")
    axs[1].grid(True)

    mem_mb = [m / 1e6 for m in mem_usages]
    axs[2].bar(names, mem_mb, color='green')
    axs[2].set_title("FLOPs", fontsize=14)
    axs[2].set_ylabel("FLOPs (Millions)")
    axs[2].grid(True)

    for ax in axs:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"[✓] Saved plot: {filename}")

# -------------------------
# Activation capture + similarity helpers
# -------------------------
def get_conv_activations(model, loader, device, num_batches=10):
    """
    Capture per-conv-layer activation vectors: mean over batch and spatial dims -> (out_channels,)
    Returns dict: {'conv_1': tensor, 'conv_2': tensor, ...} in encounter order.
    """
    model.to(device)
    model.eval()

    conv_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_modules.append((name, module))

    accum = [None] * len(conv_modules)
    counts = [0] * len(conv_modules)
    hooks = []

    def make_hook(idx):
        def hook(module, input, output):
            with torch.no_grad():
                out = output.detach()
                mean_per_channel = out.mean(dim=(0, 2, 3)).cpu()
                if accum[idx] is None:
                    accum[idx] = mean_per_channel.clone()
                else:
                    accum[idx] += mean_per_channel
                counts[idx] += 1
        return hook

    for idx, (name, module) in enumerate(conv_modules):
        hooks.append(module.register_forward_hook(make_hook(idx)))

    with torch.no_grad():
        for i, (xb, _) in enumerate(loader):
            if i >= num_batches:
                break
            xb = xb.to(device)
            _ = model(xb)

    for h in hooks:
        h.remove()

    activations = {}
    for idx, (name, module) in enumerate(conv_modules):
        if counts[idx] > 0:
            avg = accum[idx] / counts[idx]
            activations[f"conv_{idx+1}"] = avg.clone()
        else:
            activations[f"conv_{idx+1}"] = torch.zeros(module.out_channels)

    return activations

def compute_layerwise_similarity(actsA, actsB):
    """
    Compute cosine similarity per layer between two activation dicts.
    Returns dict {layer: similarity_float}
    """
    sims = {}
    for layer in sorted(set(actsA.keys()) & set(actsB.keys()),
                        key=lambda x: int(x.split("_")[1])):
        a = actsA[layer].flatten().float()
        b = actsB[layer].flatten().float()
        if a.numel() != b.numel():
            min_len = min(a.numel(), b.numel())
            a = a[:min_len]
            b = b[:min_len]
        if a.norm() == 0 or b.norm() == 0:
            sims[layer] = 0.0
        else:
            sims[layer] = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    return sims

def save_activation_similarity_json(experiment, sim_dict):
    os.makedirs("metrics", exist_ok=True)
    fname = f"metrics/activation_similarity_{experiment.replace(' ', '_')}.json"
    with open(fname, "w") as f:
        json.dump(sim_dict, f, indent=2)
    print(f"[✓] Saved activation similarity JSON: {fname}")

def plot_activation_similarity(experiment, sim_dict):
    os.makedirs("plots", exist_ok=True)
    # choose a sample similarity dict to get layers (they should all share layer keys)
    keys = list(sim_dict.keys())
    if not keys:
        print(f"[!] No similarity data to plot for {experiment}")
        return
    sample = sim_dict[keys[0]]
    layers = sorted(sample.keys(), key=lambda x: int(x.split("_")[1]))
    x = list(range(1, len(layers) + 1))
    plt.figure(figsize=(10, 6))
    for pair_name, data in sim_dict.items():
        y = [data.get(layer, 0.0) for layer in layers]
        plt.plot(x, y, marker='o', label=pair_name)
    plt.xticks(x, layers, rotation=45)
    plt.xlabel("Conv Layer")
    plt.ylabel("Cosine Similarity")
    plt.ylim(-1.0, 1.0)
    plt.title(f"Layerwise Activation Similarity - {experiment}")
    plt.grid(True)
    plt.legend()
    filename = f"plots/activation_similarity_{experiment.replace(' ', '_')}.svg"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[✓] Saved activation similarity plot: {filename}")

def compare_activations(experiments, jf_activations, nick_activations, kevin_activations):
    print("\n=== Activation Similarity Comparison Across Workflows ===")
    for exp_name in experiments.keys():
        sims = {}
        sims["JF-Nick"] = compute_layerwise_similarity(jf_activations[exp_name], nick_activations[exp_name])
        sims["JF-Kevin"] = compute_layerwise_similarity(jf_activations[exp_name], kevin_activations[exp_name])
        sims["Nick-Kevin"] = compute_layerwise_similarity(nick_activations[exp_name], kevin_activations[exp_name])

        save_activation_similarity_json(exp_name, sims)
        plot_activation_similarity(exp_name, sims)

# -------------------------
# CIFAR-10 Data Loaders
# -------------------------
def load_cifar10(batch_size=64, num_workers=4):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    train_loader = DataLoader(
        datasets.CIFAR10('data', train=True, download=True, transform=train_transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        datasets.CIFAR10('data', train=False, transform=test_transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader

def load_tiny_imagenet(batch_size: int = 64, num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
    data_dir = '/workspace/manuscript/temp/tiny-imagenet-200/'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    val_img_dir = os.path.join(val_dir, 'images')
    val_annot_path = os.path.join(val_dir, 'val_annotations.txt')

    # Reorganize validation images into subfolders (only needs to be done once)
    if os.path.exists(val_img_dir):
        with open(val_annot_path, 'r') as f:
            for line in f:
                img_file, label = line.strip().split('\t')[:2]
                label_dir = os.path.join(val_dir, label)
                os.makedirs(label_dir, exist_ok=True)
                src = os.path.join(val_img_dir, img_file)
                dst = os.path.join(label_dir, img_file)
                if os.path.exists(src):
                    shutil.move(src, dst)
        os.rmdir(val_img_dir)

    # Define transforms
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
    ])

    transform_val = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
    ])

    # Datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


# ===============================
# Basic Counting Utilities
# ===============================

def count_zeros(tensor): 
    return torch.sum(tensor == 0).item()

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===============================
# Model Statistics
# ===============================

def layer_stats(model):
    print("\nLayer-wise zero parameter stats:\n")
    for name, param in model.named_parameters():
        if param.requires_grad:
            zeros = count_zeros(param)
            total = param.numel()
            print(f"{name}: {zeros}/{total} zeros ({100 * zeros/total:.2f}%)")


# ===============================
# Layer Collapse Helpers
# ===============================

def _find_layer_indices(named_layers, start_layer_name, end_layer_name):
    start_idx = end_idx = None
    for i, (name, _) in enumerate(named_layers):
        if name == start_layer_name:
            start_idx = i
        if name == end_layer_name:
            end_idx = i
    return start_idx, end_idx

def _simulate_input(model, section_name, start_idx):
    dummy_input = torch.randn(1, 3, 32, 32).to(next(model.parameters()).device)
    x = dummy_input

    if section_name == "features":
        for layer in list(model.features.children())[:start_idx]:
            x = layer(x)
    else:
        for layer in model.features:
            x = layer(x)
        x = torch.flatten(x, 1)
        for layer in list(model.classifier.children())[:start_idx]:
            x = layer(x)

    return dummy_input, x

def _build_collapsed_block(layer_type, in_features, out_features, output_shape):
    if layer_type == nn.Conv2d:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),   # <-- ADD THIS
            nn.MaxPool2d(kernel_size=2, stride=2),  # <-- ADD THIS
            nn.AdaptiveAvgPool2d((1, 1))
        )
    elif layer_type == nn.Linear:
        flattened_input = in_features * output_shape[-1] * output_shape[-2]
        return nn.Linear(flattened_input, out_features)
    else:
        raise NotImplementedError("Unsupported layer type for collapsing.")

def _replace_layers(named_layers, start_idx, end_idx, new_block):
    new_layers = []
    for i, (name, layer) in enumerate(named_layers):
        if i == start_idx:
            new_layers.append((f"collapsed_{named_layers[start_idx][0]}_to_{named_layers[end_idx][0]}", new_block))
        elif start_idx < i <= end_idx:
            continue  # skip collapsed layers
        elif i > end_idx and isinstance(layer, nn.MaxPool2d):
            print(f"Removing MaxPool2d after collapsed block: {name}")
            continue  # remove dangerous MaxPool2d
        else:
            new_layers.append((name, layer))
    return nn.Sequential(OrderedDict(new_layers))


# ===============================
# Main Collapse Function
# ===============================

def collapse_block(model, start_layer_name, end_layer_name):
    print(f"\nCollapsing layers from '{start_layer_name}' to '{end_layer_name}'...")
    containers = {
        "features": model.features,
        "classifier": model.classifier,
    }

    for section_name, container in containers.items():
        named_layers = list(container.named_children())
        start_idx, end_idx = _find_layer_indices(named_layers, start_layer_name, end_layer_name)

        if start_idx is not None and end_idx is not None:
            assert start_idx <= end_idx, "Start index must be <= end index"

            full_block = named_layers[start_idx:end_idx + 1]
            selected_layers = [layer for _, layer in full_block if isinstance(layer, (nn.Conv2d, nn.Linear))]

            if len(selected_layers) < 2:
                raise ValueError("Need at least 2 Conv2d or Linear layers to collapse.")
            
            layer_type = type(selected_layers[0])
            if not all(isinstance(l, layer_type) for l in selected_layers):
                raise ValueError("Cannot collapse mixed layer types.")

            dummy_input, x = _simulate_input(model, section_name, start_idx)

            in_features = x.shape[1] if layer_type == nn.Linear else selected_layers[0].in_channels
            for layer in selected_layers:
                x = layer(x)
            out_features = x.shape[1] if layer_type == nn.Linear else selected_layers[-1].out_channels

            print(f"Input shape: {dummy_input.shape} → Output shape: {x.shape}")

            collapsed_block = _build_collapsed_block(layer_type, in_features, out_features, x.shape)
            updated_container = _replace_layers(named_layers, start_idx, end_idx, collapsed_block)

            if section_name == "features":
                model.features = updated_container
            else:
                model.classifier = updated_container

            print(f"Collapsed {section_name} layers {start_layer_name} → {end_layer_name}")
            print(f"New trainable params: {count_trainable_params(model)}")
            return model

    print(f"New structure:\n{layer_stats(model)}")
    raise ValueError(f"Layer names '{start_layer_name}' or '{end_layer_name}' not found.")


# ===============================
# Cloning Utility
# ===============================

def clone_model(model, model_class):
    """Utility to clone a model and load weights to keep experiments isolated."""
    new_model = model_class()
    new_model.load_state_dict(model.state_dict())
    return new_model

def collapse_only(model_weights_1, compression_set, model_class, model_kwargs=None):
    model_kwargs = model_kwargs or {}

    model = model_class(**model_kwargs)
    checkpoint = torch.load(model_weights_1, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    for start, end in compression_set:
        model = collapse_block(model, start, end)

    return model

    """
    Args:
        model_weights_1 (str): Path to model weights (used for collapsing layers).
        compression_set (list of tuples): Each tuple is (start_layer_name, end_layer_name).
        model_class: the class of the model to instantiate.
    Returns:
        nn.Module: Model with collapsed layers.
    """
    checkpoint_1 = torch.load(model_weights_1, map_location='cpu')
    state_dict_1 = checkpoint_1['model'] if 'model' in checkpoint_1 else checkpoint_1

    reference_model = model_class()

    # First, load the state dict for the model
    try:
        reference_model.load_state_dict(state_dict_1)
    except RuntimeError as e:
        print(f"Warning: Error loading state_dict for {model_class.__name__}. Proceeding with layer collapse.")

    collapsed_layer_names = set()

    # Apply layer collapse
    for start_name, end_name in compression_set:
        reference_model = collapse_block(reference_model, start_name, end_name)
        
        start_idx = int(start_name.split('_')[1])
        end_idx = int(end_name.split('_')[1])
        for i in range(start_idx, end_idx + 1):
            collapsed_layer_names.add(f"features.conv_{i}.weight")
            collapsed_layer_names.add(f"features.conv_{i}.bias")
            collapsed_layer_names.add(f"features.bn_{i}.weight")
            collapsed_layer_names.add(f"features.bn_{i}.bias")
            collapsed_layer_names.add(f"features.bn_{i}.running_mean")
            collapsed_layer_names.add(f"features.bn_{i}.running_var")

    # Now, load the model without the collapsed layers' weights
    state_dict_1_filtered = {k: v for k, v in state_dict_1.items() if k not in collapsed_layer_names}
    reference_model.load_state_dict(state_dict_1_filtered, strict=False)  # strict=False allows missing keys

    print(f"Collapsed model structure: {reference_model}")
    print(f"New trainable params: {count_trainable_params(reference_model)}")
    return reference_model

    """
    Args:
        model_weights_1 (str): Path to model weights (used for collapsing layers).
        compression_set (list of tuples): Each tuple is (start_layer_name, end_layer_name).
        model_class: the class of the model to instantiate.
    Returns:
        nn.Module: Model with collapsed layers.
    """
    checkpoint_1 = torch.load(model_weights_1, map_location='cpu')
    state_dict_1 = checkpoint_1['model'] if 'model' in checkpoint_1 else checkpoint_1

    reference_model = model_class()
    reference_model.load_state_dict(state_dict_1)

    collapsed_layer_names = set()
    for start_name, end_name in compression_set:
        reference_model = collapse_block(reference_model, start_name, end_name)
        
        start_idx = int(start_name.split('_')[1])
        end_idx = int(end_name.split('_')[1])
        for i in range(start_idx, end_idx + 1):
            collapsed_layer_names.add(f"features.conv_{i}.weight")
            collapsed_layer_names.add(f"features.conv_{i}.bias")
            collapsed_layer_names.add(f"features.bn_{i}.weight")
            collapsed_layer_names.add(f"features.bn_{i}.bias")
            collapsed_layer_names.add(f"features.bn_{i}.running_mean")
            collapsed_layer_names.add(f"features.bn_{i}.running_var")

    return reference_model