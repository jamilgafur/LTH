import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from pyPrune.models.Vgg16 import VGG16_CIFAR10
from collections import OrderedDict
from utils import *
from collections import OrderedDict
from torch import nn
import torch

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
    model = clone_model(model, VGG16_CIFAR10)  # Work on a fresh copy
    
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
    for run in range(1, 2):
        model_path_097 = "../structured_study/pruning_checkpoints/Vgg16_pretrain10_finetune30_steps21_batch1024_devicecuda_strategy_magnitude/checkpoint_Finetuned_0.97.pth"
        model_path_000 = "../structured_study/pruning_checkpoints/Vgg16_pretrain10_finetune30_steps21_batch1024_devicecuda_strategy_magnitude/checkpoint_Original_0.00.pth"

        if not os.path.exists(model_path_097) or not os.path.exists(model_path_000):
            print("Required model weight files not found. Exiting.")
            return

        train_loader, test_loader = load_cifar10()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        epochs = 30

        # Define compression experiments
        experiments = {
            "Last 2 Conv Layer Only": ('conv_12', 'conv_13'),
            "Original Model": None,
            "Stage 5": ('conv_11', 'conv_13'),
            "Stage 4": ('conv_8', 'conv_10'),
            "Stage 4-5": ('conv_8', 'conv_13'),
            "Stage 3": ('conv_5', 'conv_7'),
            "Stage 3-5": ('conv_5', 'conv_13'),
            "Stage 2": ('conv_3', 'conv_4'),
            "Stage 2-5": ('conv_3', 'conv_13'),
            "Stage 1": ('conv_1', 'conv_2'),
            "All Conv Layers": ('conv_1', 'conv_13'),
        }

        # ----------------------------
        # Compression only
        # ----------------------------
        orig_param_counts = []
        orig_final_accuracies = []
        orig_exp_names = []

        for exp_name, collapse_range in experiments.items():
            print(f"\n=== Running Standard experiment: {exp_name} ===")
            base_model = VGG16_CIFAR10()
            base_model.load_state_dict(torch.load(model_path_097)['model'])

            p_count, init_acc, final_acc, acc_list = run_experiment(
                base_model, train_loader, test_loader, device, epochs, collapse_range
            )

            orig_param_counts.append(p_count)
            orig_final_accuracies.append(final_acc)
            orig_exp_names.append(exp_name)

        # ----------------------------
        # Kevin Experiment
        # ----------------------------
        merged_param_counts = []
        merged_final_accuracies = []
        merged_exp_names = []

        for exp_name, collapse_range in experiments.items():
            print(f"\n=== Running Merged experiment: {exp_name} ===")
            if collapse_range is None:
                print("No collapse in this experiment — skipping merge variant.")
                continue

            try:
                merged_model = collapse_only(
                    model_weights_1=model_path_000,
                    compression_set=[collapse_range],
                    model_class=VGG16_CIFAR10
                )
            except Exception as e:
                print(f"Error in collapse_only for {exp_name}: {e}")
                continue

            param_count = count_trainable_params(merged_model)
            accuracies, final_acc  = train_and_evaluate(
                merged_model, train_loader, test_loader, device, epochs
            )
            print(f"[Merged] Final Accuracy: {final_acc:.2f}%, Params: {param_count}")
            
            merged_param_counts.append(param_count)
            merged_final_accuracies.append(final_acc)
            merged_exp_names.append(exp_name)

 
        plot_results(
            orig_param_counts,
            orig_final_accuracies,
            orig_exp_names,
            "Original Compression Experiments",
            f"original_experiments_plot_{run}.svg"
        )

        plot_results(
            merged_param_counts,
            merged_final_accuracies,
            merged_exp_names,
            "Merged (collapse_and_merge) Experiments",
            f"merged_experiments_plot_{run}.svg"
        )

# ----------------------------
# Plotting
# ----------------------------
def plot_results(params, accs, names, title, filename):
    sorted_data = sorted(zip(params, accs, names), key=lambda x: x[0])

    if not sorted_data:
        print(f"[Warning] No data to plot for: {title}")
        return

    sorted_params, sorted_accs, sorted_names = zip(*sorted_data)

    fig, ax1 = plt.subplots(figsize=(14, 7))
    bar_colors = ['#1f77b4'] * len(sorted_names)

    bars = ax1.bar(sorted_names, sorted_accs, color=bar_colors, alpha=0.7, label='Final Accuracy (%)')
    ax1.set_ylabel('Final Accuracy (%)', fontsize=14)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel('Experiment', fontsize=14)
    ax1.set_title(title, fontsize=16)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=11)

    ax2 = ax1.twinx()
    ax2.plot(sorted_names, sorted_params, 'ro--', label='Trainable Parameters (log scale)')
    ax2.set_ylabel('Trainable Parameters', color='red', fontsize=14)
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', colors='red')
    ax2.grid(False)

    for i, param in enumerate(sorted_params):
        ax2.annotate(f'{param:,}',
                    xy=(i, param),
                    xytext=(0, -15),
                    textcoords='offset points',
                    ha='center', va='top', color='red', fontsize=10)

    lines_labels = [ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax1.legend(lines, labels, loc='upper right', fontsize=12)

    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Saved plot: {filename}")

    # Print summary
    print("\nSummary:")
    for p, a, n in zip(sorted_params, sorted_accs, sorted_names):
        print(f"{n}: Params={p:,}, Final Acc={a:.2f}%")

if __name__ == "__main__":
    main()
