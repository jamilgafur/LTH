import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from pyPrune.models.Vgg16 import VGG16_CIFAR10
from collections import OrderedDict

def load_cifar10(batch_size: int = 64, num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_loader = DataLoader(
        datasets.CIFAR10('data', train=True, download=True, transform=train_transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = DataLoader(
        datasets.CIFAR10('data', train=False, transform=test_transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    print("CIFAR-10 data shape: ", next(iter(train_loader))[0].shape)
    return train_loader, test_loader

def count_zeros(tensor): 
    return torch.sum(tensor == 0).item()

def layer_stats(model):
    print("\nLayer-wise zero parameter stats:\n")
    for name, param in model.named_parameters():
        if param.requires_grad:
            zeros = count_zeros(param)
            total = param.numel()
            print(f"{name}: {zeros}/{total} zeros ({100 * zeros/total:.2f}%)")

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            model = model.to(device)
            preds = model(xb)
            _, predicted = preds.max(1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)
    return 100 * correct / total

from collections import OrderedDict
from torch import nn
import torch


def collapse_block(model, start_layer_name, end_layer_name):
    containers = {
        "features": model.features,
        "classifier": model.classifier,
    }

    for section_name, container in containers.items():
        named = list(container.named_children())

        start_idx = end_idx = None
        for i, (name, _) in enumerate(named):
            if name == start_layer_name:
                start_idx = i
            if name == end_layer_name:
                end_idx = i

        if start_idx is not None and end_idx is not None:
            assert start_idx <= end_idx, "Start index must be <= end index"

            full_block = named[start_idx:end_idx + 1]

            # Only keep Conv2d or Linear layers for collapsing
            selected_layers = [layer for _, layer in full_block if isinstance(layer, (nn.Conv2d, nn.Linear))]

            if len(selected_layers) < 2:
                raise ValueError("Need at least 2 Conv2d or Linear layers to collapse.")

            layer_type = type(selected_layers[0])
            if not all(isinstance(l, layer_type) for l in selected_layers):
                raise ValueError("Cannot collapse mixed layer types.")

            # Simulate input
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

            in_features = x.shape[1] if layer_type == nn.Linear else selected_layers[0].in_channels
            for layer in selected_layers:
                x = layer(x)
            out_features = x.shape[1] if layer_type == nn.Linear else selected_layers[-1].out_channels

            print(f"Input shape: {dummy_input.shape} → Output shape: {x.shape}")

            # Construct new block
            if layer_type == nn.Conv2d:
                collapsed_block = nn.Sequential(
                    nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=0),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
            elif layer_type == nn.Linear:
                collapsed_block = nn.Linear(in_features * x.shape[-1] * x.shape[-2], out_features)
            else:
                raise NotImplementedError

            # Replace layers
            new_layers = []
            for i, (name, layer) in enumerate(named):
                if i == start_idx:
                    new_layers.append((f"collapsed_{start_layer_name}_to_{end_layer_name}", collapsed_block))
                elif start_idx < i <= end_idx:
                    continue  # skip
                elif i > end_idx and isinstance(layer, nn.MaxPool2d):
                    print(f"Removing MaxPool2d after collapsed block: {name}")
                    continue  # remove dangerous MaxPool2d
                else:
                    new_layers.append((name, layer))

            updated_container = nn.Sequential(OrderedDict(new_layers))
            if section_name == "features":
                model.features = updated_container
            else:
                model.classifier = updated_container

            print(f"Collapsed {section_name} layers {start_layer_name} → {end_layer_name}")
            print(f"New trainable params: {count_trainable_params(model)}")
            return model

    raise ValueError(f"Layer names '{start_layer_name}' or '{end_layer_name}' not found.")


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def clone_model(model):
    """Utility to clone a model and load weights to keep experiments isolated."""
    new_model = VGG16_CIFAR10()
    new_model.load_state_dict(model.state_dict())
    return new_model

def train_and_evaluate(model, train_loader, test_loader, device, epochs=10):
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = correct = total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            _, predicted = preds.max(1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)

        avg_loss = total_loss / total
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
        accuracies.append(acc)

    final_acc = evaluate(model, test_loader, device)
    print(f"\nFinal Test Accuracy: {final_acc:.2f}%")
    return accuracies, final_acc

def run_experiment(model, train_loader, test_loader, device, epochs=10, collapse_range=None):
    """
    Runs training and evaluation on the model, optionally collapsing a layer range.
    collapse_range: tuple of (start_layer_name, end_layer_name) or None for no collapse.
    Returns (param_count, final_accuracy, accuracies_per_epoch)
    """
    model = clone_model(model)  # Work on a fresh copy

    if collapse_range:
        print(f"\nCollapsing layers: {collapse_range}")
        model = collapse_block(model, *collapse_range)
    else:
        print("\nNo collapsing applied - original model")

    param_count = count_trainable_params(model)
    print(f"Trainable parameters: {param_count}")

    # Initial evaluation before training
    init_acc = evaluate(model, test_loader, device)
    print(f"Initial test accuracy: {init_acc:.2f}%")

    accuracies, final_acc = train_and_evaluate(model, train_loader, test_loader, device, epochs)

    return param_count, init_acc, final_acc, accuracies

def main():
    for run in range(1, 2):  # Just one run for demonstration
        model_path = "../structured_study/pruning_checkpoints/Vgg16_pretrain10_finetune30_steps21_batch1024_devicecuda_strategy_magnitude/checkpoint_Finetuned_0.97.pth"
        if not os.path.exists(model_path):
            print(f"Model path {model_path} not found. Exiting.")
            return

        base_model = VGG16_CIFAR10()
        base_model.load_state_dict(torch.load(model_path)['model'])
        print(f"Loaded model from {model_path}")
        # print layer stats
        layer_stats(base_model)
        train_loader, test_loader = load_cifar10()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        epochs = 2

        # Define collapse experiments dynamically:
        experiments = {
            "Last 2 Conv Layer Only ": ('conv_12', 'conv_13'),
            "Original Model": None,
            "Stage 5": ('conv_11', 'conv_13'),
            "Stage 4": ('conv_8', 'conv_10'),
            "All Conv Layers": ('conv_1', 'conv_13'),
        }

        param_counts = []
        final_accuracies = []
        epoch_accuracies = []
        exp_names = []
        init_acc = None
        for exp_name, collapse_range in experiments.items():
            print(f"\n=== Running experiment: {exp_name} ===")
            base_model = VGG16_CIFAR10()
            base_model.load_state_dict(torch.load(model_path)['model'])
            p_count, init_acc, final_acc, acc_list = run_experiment(
                base_model, train_loader, test_loader, device, epochs, collapse_range
            )
            param_counts.append(p_count)
            final_accuracies.append(final_acc)
            epoch_accuracies.append(acc_list)
            exp_names.append(exp_name)

        # Sort by parameter counts
        sorted_data = sorted(zip(param_counts, final_accuracies, exp_names), key=lambda x: x[0])
        sorted_params, sorted_final_acc, sorted_exp_names = zip(*sorted_data)

        # --- Improved plotting code starts here ---

        fig, ax1 = plt.subplots(figsize=(12, 7))

        bar_colors = ['#1f77b4', '#ff7f0e','#2ca02c']  # customize colors for bars

        # Bar plot for final accuracy
        bars = ax1.bar(sorted_exp_names, sorted_final_acc, color=bar_colors, alpha=0.7, label='Final Accuracy (%)')
        ax1.set_ylabel('Final Accuracy (%)', fontsize=14)
        ax1.set_ylim(0, 100)
        ax1.set_xlabel('Experiment', fontsize=14)
        ax1.set_title('Final Accuracy vs Trainable Parameters', fontsize=16)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Annotate bars with accuracy values
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=12)

        # Secondary axis for parameters (log scale)
        ax2 = ax1.twinx()
        ax2.plot(sorted_exp_names, sorted_params, 'ro--', label='Trainable Parameters (log scale)')
        ax2.set_ylabel('Trainable Parameters', color='red', fontsize=14)
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', colors='red')
        ax2.grid(False)
   
        # Annotate parameter counts
        for i, param in enumerate(sorted_params):
            ax2.annotate(f'{param:,}',
                        xy=(i, param),
                        xytext=(0, -15),
                        textcoords='offset points',
                        ha='center', va='top', color='red', fontsize=10)

        # Legend combining both axes
        lines_labels = [ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        ax1.legend(lines, labels, loc='upper right', fontsize=12)

        plt.tight_layout()
        plt.savefig(f"final_accuracy_vs_params_better_{run}.png")
        plt.show()
        print("Saved improved plot to final_accuracy_vs_params_better.png")

        # print all data plotted
        print("\nSummary of experiments:")
        for p, acc, name in zip(sorted_params, sorted_final_acc, sorted_exp_names):
            print(f"{name}: Params={p}, Final Acc={acc:.2f}%")

if __name__ == "__main__":
    main()
