import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import time
import json

from pyPrune.models.Vgg16 import VGG16_CIFAR10
from utils import count_trainable_params, collapse_block, collapse_only
from torchinfo import summary

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

# -------------------------
# Training and Evaluation
# -------------------------
def train_and_evaluate(model, train_loader, test_loader, device, epochs=10):
    if epochs <= 0:
        print("[Warning] Number of training epochs is zero or negative!")
        final_acc = evaluate(model, test_loader, device)
        return [], final_acc, []

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    accuracies = []
    losses = []

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
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")
        accuracies.append(acc)
        losses.append(avg_loss)

    final_acc = evaluate(model, test_loader, device)
    print(f"Final Test Accuracy: {final_acc:.2f}%")
    return accuracies, final_acc, losses

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            _, predicted = preds.max(1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)
    return 100 * correct / total

# -------------------------
# Benchmark Inference
# -------------------------
def benchmark_model(model, loader, device, num_batches=10):
    model.eval()
    model.to(device)
    times = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for i, (xb, _) in enumerate(loader):
            if i >= num_batches:
                break
            xb = xb.to(device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            _ = model(xb)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - start_time)
    avg_time = sum(times) / len(times) if times else 0
    peak_mem = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
    return avg_time, peak_mem

def describe_model(model, input_size=(1, 3, 32, 32), device='cpu'):
    print("=" * 60)
    print("🔍 Model Summary (via torchinfo)")
    print("=" * 60)
    summary(model, input_size=input_size, device=device)
    print("=" * 60)


# -------------------------
# Checkpoint and Naming
# -------------------------
def get_checkpoint_filename(workflow, exp_name, model_type, pretrain_epochs, finetune_epochs):
    exp_tag = exp_name.replace(" ", "_").replace("-", "_")
    return f"checkpoints/{workflow}_{exp_tag}_{model_type}_pre{pretrain_epochs}_ft{finetune_epochs}.pth"

def save_metrics_json(workflow, experiment, accuracy, loss, infer_time=None, mem_usage=None, param_count=None):
    os.makedirs("metrics", exist_ok=True)
    json_path = f"metrics/{workflow}_metrics.json"

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    if workflow not in data:
        data[workflow] = {}

    data[workflow][experiment] = {
        "accuracy": accuracy,
        "loss": loss,
        "inference_time": infer_time,
        "memory_usage": mem_usage,
        "trainable_params": param_count,
    }

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[✓] Saved metrics to {json_path}")

def load_metrics_json(workflow, experiment):
    json_path = f"metrics/{workflow}_metrics.json"
    if not os.path.exists(json_path):
        return [], []

    with open(json_path, "r") as f:
        data = json.load(f)
    if workflow in data and experiment in data[workflow]:
        return data[workflow][experiment]["accuracy"], data[workflow][experiment]["loss"]
    return [], []

def plot_accuracy_loss_curve(acc_list, loss_list, workflow, experiment):
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(acc_list, label='Accuracy', marker='o')
    plt.plot(loss_list, label='Loss', marker='x')
    plt.title(f'{workflow} - {experiment} Accuracy & Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    filename = f"plots/{workflow}_{experiment.replace(' ', '_')}_metrics.svg"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[✓] Saved plot: {filename}")

# -------------------------
# Experiment Runner
# -------------------------
def run_experiment(model, train_loader, test_loader, device, epochs, workflow, exp_name,
                   collapse_range=None, pretrain=0):
    ckpt_path = get_checkpoint_filename(workflow, exp_name, 'VGG16', pretrain, epochs)

    if collapse_range:
        model = collapse_block(model, *collapse_range)
    model.to(device)
    describe_model(model, input_size=(1, 3, 32, 32), device=device)

    print(f"[•] Training model: {exp_name}")
    acc_list, final_acc, loss_list = train_and_evaluate(model, train_loader, test_loader, device, epochs)
    
    # Ensure the directory for checkpoint exists
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({'model': model.state_dict()}, ckpt_path)
    
    
    plot_accuracy_loss_curve(acc_list, loss_list, workflow, exp_name)

    param_count = count_trainable_params(model)
    infer_time, mem_usage = benchmark_model(model, test_loader, device)
    save_metrics_json(
            workflow, exp_name,
            acc_list, loss_list,
            infer_time=infer_time,
            mem_usage=mem_usage,
            param_count=param_count
        )
    return param_count, final_acc, infer_time, mem_usage

# -------------------------
# Result Plotting
# -------------------------
def plot_results(params, accs, names, title, filename, infer_times=None, mem_usages=None):
    fig, axs = plt.subplots(3, 1, figsize=(16, 18))

    # Accuracy Bar Plot
    axs[0].bar(names, accs, color='skyblue')
    axs[0].set_title(f"{title} - Final Accuracy (%)", fontsize=14)
    axs[0].set_ylabel("Accuracy (%)")
    axs[0].grid(True)

    # Trainable Parameters Line Plot (log scale)
    ax0_twin = axs[0].twinx()
    ax0_twin.plot(names, params, 'ro--', label='Trainable Parameters (log)', linewidth=2)
    ax0_twin.set_ylabel('Trainable Parameters', color='red')
    ax0_twin.set_yscale('log')
    ax0_twin.tick_params(axis='y', colors='red')
    for i, param in enumerate(params):
        ax0_twin.annotate(f'{param:,}', xy=(i, param), xytext=(0, -15),
                          textcoords='offset points', ha='center', fontsize=9, color='red')

    # Inference Time
    axs[1].bar(names, infer_times, color='orange')
    axs[1].set_title("Inference Time (avg per batch in seconds)", fontsize=14)
    axs[1].set_ylabel("Time (s)")
    axs[1].grid(True)

    # Memory Usage
    mem_mb = [m / 1e6 for m in mem_usages]
    axs[2].bar(names, mem_mb, color='green')
    axs[2].set_title("Peak Memory Usage", fontsize=14)
    axs[2].set_ylabel("Memory (MB)")
    axs[2].grid(True)

    # Common settings
    for ax in axs:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"[✓] Saved plot: {filename}")

# -------------------------
# Main
# -------------------------
from collections import OrderedDict

def main():
    model_path_097 = "../structured_study/pruning_checkpoints/Vgg16_pretrain10_finetune30_steps21_batch1024_devicecuda_strategy_magnitude/checkpoint_Finetuned_0.97.pth"
    model_path_000 = "../structured_study/pruning_checkpoints/Vgg16_pretrain10_finetune30_steps21_batch1024_devicecuda_strategy_magnitude/checkpoint_Original_0.00.pth"

    if not os.path.exists(model_path_097) or not os.path.exists(model_path_000):
        print("Required model weight files not found. Exiting.")
        return

    print("Loading CIFAR-10 data...")
    train_loader, test_loader = load_cifar10()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 30
    pretrain = 10  # kept for consistency

    experiments = {
        # "Last 2 Conv Layer Only": ('conv_12', 'conv_13'),
        "Original Model": None,
        # "Stage 5": ('conv_11', 'conv_13'),
        # "Stage 4": ('conv_8', 'conv_10'),
        "Stage 4-5": ('conv_8', 'conv_13'),
        # "Stage 3": ('conv_5', 'conv_7'),
        # "Stage 3-5": ('conv_5', 'conv_13'),
        # "Stage 2": ('conv_3', 'conv_4'),
        "Stage 2-5": ('conv_3', 'conv_13'),
        # "Stage 1": ('conv_1', 'conv_2'),
        "All Conv Layers": ('conv_1', 'conv_13'),
    }

    # ------------------------
    # JF Experiment Workflow
    # ------------------------
    jf_param_counts = []
    jf_final_accuracies = []
    jf_exp_names = []
    jf_infer_times = []
    jf_mem_usages = []

    print("\n=== Running JF experiment ===")
    print(f"Loading pre-trained model from {model_path_097}...")
    base_model = VGG16_CIFAR10()
    base_model.load_state_dict(torch.load(model_path_097)['model'])

    for exp_name, collapse_range in experiments.items():
        print(f"\nRunning experiment: {exp_name}")

        # Reload fresh model for each experiment
        base_model = VGG16_CIFAR10()
        base_model.load_state_dict(torch.load(model_path_097)['model'])

        if collapse_range is not None:
            print(f"Applying compression for {exp_name}...")
            base_model = collapse_only(model_weights_1=model_path_097, compression_set=[collapse_range], model_class=VGG16_CIFAR10)

        print(f"Finetuning the compressed model...")
        param_count, final_acc, infer_time, mem_usage = run_experiment(
            base_model, train_loader, test_loader, device, epochs, "JF", exp_name, pretrain=pretrain
        )

        jf_param_counts.append(param_count)
        jf_final_accuracies.append(final_acc)
        jf_infer_times.append(infer_time)
        jf_mem_usages.append(mem_usage)
        jf_exp_names.append(exp_name)

    print("\nPlotting JF results...")
    plot_results(
        jf_param_counts, jf_final_accuracies, jf_exp_names, "JF Experiment", "jf_experiment_results.svg",
        infer_times=jf_infer_times, mem_usages=jf_mem_usages
    )

 # ------------------------
    # Nick Experiment Workflow
    # ------------------------
    nick_param_counts = []
    nick_final_accuracies = []
    nick_exp_names = []
    nick_infer_times = []
    nick_mem_usages = []

    print("\n=== Running Nick experiment ===")
    print(f"Loading untrained model from {model_path_000}...")

    for exp_name, collapse_range in experiments.items():
        print(f"\nRunning experiment: {exp_name}")

        base_model = VGG16_CIFAR10()
        base_model.load_state_dict(torch.load(model_path_000)['model'])

        # Nick workflow: First finetune
        print("First finetuning before compression...")
        param_count, final_acc, infer_time, mem_usage = run_experiment(
            base_model, train_loader, test_loader, device, epochs+pretrain, "Nick", exp_name, pretrain=pretrain
        )
        print(f"First finetuning completed. Accuracy: {final_acc:.4f}")

        if collapse_range is not None:
            print(f"Applying compression for {exp_name}...")
            temp_model_path = "temp_model.pth"
            torch.save({'model': base_model.state_dict()}, temp_model_path)
            base_model = collapse_only(model_weights_1=temp_model_path, compression_set=[collapse_range], model_class=VGG16_CIFAR10)
            os.remove(temp_model_path)

        print("Second finetuning after compression...")
        param_count, final_acc, infer_time, mem_usage = run_experiment(
            base_model, train_loader, test_loader, device, epochs, "Nick", exp_name, pretrain=pretrain
        )
        print(f"Second finetuning completed. Final accuracy: {final_acc:.4f}")

        nick_param_counts.append(param_count)
        nick_final_accuracies.append(final_acc)
        nick_infer_times.append(infer_time)
        nick_mem_usages.append(mem_usage)
        nick_exp_names.append(exp_name)

    print("\nPlotting Nick results...")
    plot_results(
        nick_param_counts, nick_final_accuracies, nick_exp_names, "Nick Experiment", "nick_experiment_results.svg",
        infer_times=nick_infer_times, mem_usages=nick_mem_usages
    )
    # ------------------------
    # Kevin Experiment Workflow
    # ------------------------
    kevin_param_counts = []
    kevin_final_accuracies = []
    kevin_exp_names = []
    kevin_infer_times = []
    kevin_mem_usages = []

    print("\n=== Running Kevin experiment ===")
    print(f"Loading untrained model from {model_path_000}...")

    for exp_name, collapse_range in experiments.items():
        print(f"\nRunning experiment: {exp_name}")

        base_model = VGG16_CIFAR10()
        base_model.load_state_dict(torch.load(model_path_000)['model'])

        if collapse_range is not None:
            print(f"Applying compression for {exp_name}...")
            base_model = collapse_only(model_weights_1=model_path_000, compression_set=[collapse_range], model_class=VGG16_CIFAR10)

        print(f"Finetuning the compressed model...")
        param_count, final_acc, infer_time, mem_usage = run_experiment(
            base_model, train_loader, test_loader, device, epochs+pretrain, "Kevin", exp_name, pretrain=pretrain
        )

        kevin_param_counts.append(param_count)
        kevin_final_accuracies.append(final_acc)
        kevin_infer_times.append(infer_time)
        kevin_mem_usages.append(mem_usage)
        kevin_exp_names.append(exp_name)

    print("\nPlotting Kevin results...")
    plot_results(
        kevin_param_counts, kevin_final_accuracies, kevin_exp_names, "Kevin Experiment", "kevin_experiment_results.svg",
        infer_times=kevin_infer_times, mem_usages=kevin_mem_usages
    )


   

if __name__ == "__main__":
    main()

