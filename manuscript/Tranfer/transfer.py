import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import time
import json
from torch.nn import functional as F
import torch
import time
from fvcore.nn import FlopCountAnalysis
from pyPrune.models.Vgg16 import VGG16_CIFAR10
from utils import count_trainable_params, collapse_block, collapse_only
from torchinfo import summary
from collections import OrderedDict
import numpy as np
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
# -------------------------
# Helper: load checkpoint into matching architecture
# -------------------------
def load_model_from_checkpoint(ckpt_path, collapse_range, device):
    """
    Load checkpoint at ckpt_path into a model whose architecture matches whether
    collapse_range is present. If collapse_range is not None, apply collapse_block
    to the fresh model before loading state_dict so shapes/keys match.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # instantiate fresh base model
    model = VGG16_CIFAR10()
    # If this experiment used collapse_range, modify architecture before loading
    if collapse_range is not None:
        # collapse_block returns the modified model (same function used in run_experiment)
        model = collapse_block(model, *collapse_range)

    # load weights
    sd = torch.load(ckpt_path)['model']
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model

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
    flops = 0
    with torch.no_grad():
        for i, (xb, _) in enumerate(loader):
            if i >= num_batches:
                break
            xb = xb.to(device)

            # Measure inference time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            output = model(xb)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - start_time)

            # Measure FLOPs (only on the first batch for simplicity)
            if i == 0:
                flops = FlopCountAnalysis(model, xb).total()

    avg_time = sum(times) / len(times) if times else 0
    return avg_time, flops
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
    return np.abs(sims)

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

# -------------------------
# Main
# -------------------------
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
        "Original Model": None,
        "Stage 4-5": ('conv_8', 'conv_13'),
        "Stage 2-5": ('conv_3', 'conv_13'),
        # "All Conv Layers": ('conv_1', 'conv_13'),
    }

    # Run JF Experiment Workflow
    jf_data = run_jf_experiment(experiments, model_path_097, train_loader, test_loader, device, epochs, pretrain)
    
    # Run Nick Experiment Workflow
    nick_data = run_nick_experiment(experiments, model_path_000, train_loader, test_loader, device, epochs, pretrain)

    # Run Kevin Experiment Workflow
    kevin_data = run_kevin_experiment(experiments, model_path_000, train_loader, test_loader, device, epochs, pretrain)

    # Compare activations across workflows
    compare_activations(experiments, jf_data, nick_data, kevin_data)


def run_jf_experiment(experiments, model_path_097, train_loader, test_loader, device, epochs, pretrain):
    jf_param_counts, jf_final_accuracies, jf_exp_names = [], [], []
    jf_infer_times, jf_mem_usages = [], []
    jf_activations = {}

    print("\n=== Running JF experiment ===")
    for exp_name, collapse_range in experiments.items():
        print(f"\nRunning JF experiment: {exp_name}")

        # prepare base_model (not strictly needed if we load checkpoint)
        base_model = VGG16_CIFAR10()
        base_model.load_state_dict(torch.load(model_path_097)['model'])
        if collapse_range is not None:
            base_model = collapse_only(
                model_weights_1=model_path_097,
                compression_set=[collapse_range],
                model_class=VGG16_CIFAR10
            )

        ckpt_path = get_checkpoint_filename("JF", exp_name, 'VGG16', pretrain, epochs)
        print(f"[•] Checking for existing checkpoint at: {ckpt_path}")

        if os.path.exists(ckpt_path):
            try:
                print(f"[i] Found existing checkpoint for JF {exp_name}: {ckpt_path}. Loading instead of re-training.")
                model_ckpt = load_model_from_checkpoint(ckpt_path, collapse_range, device)
                param_count = count_trainable_params(model_ckpt)
                final_acc = evaluate(model_ckpt, test_loader, device)
                infer_time, mem_usage = benchmark_model(model_ckpt, test_loader, device)
            except Exception as e:
                print(f"[!] Failed to load checkpoint for JF {exp_name} (will retrain): {e}")
                param_count, final_acc, infer_time, mem_usage = run_experiment(
                    base_model, train_loader, test_loader, device, epochs, "JF", exp_name, pretrain=pretrain
                )
                # after training, load the saved final model properly
                model_ckpt = load_model_from_checkpoint(ckpt_path, collapse_range, device)
        else:
            param_count, final_acc, infer_time, mem_usage = run_experiment(
                base_model, train_loader, test_loader, device, epochs, "JF", exp_name, pretrain=pretrain
            )
            model_ckpt = load_model_from_checkpoint(ckpt_path, collapse_range, device)

        # collect activations from final model
        jf_activations[exp_name] = get_conv_activations(model_ckpt, test_loader, device)

        jf_param_counts.append(param_count)
        jf_final_accuracies.append(final_acc)
        jf_infer_times.append(infer_time)
        jf_mem_usages.append(mem_usage)
        jf_exp_names.append(exp_name)

    plot_results(
        jf_param_counts, jf_final_accuracies, jf_exp_names,
        "JF Experiment", "jf_experiment_results.svg",
        infer_times=jf_infer_times, mem_usages=jf_mem_usages
    )
    return jf_activations

def run_nick_experiment(experiments, model_path_000, train_loader, test_loader, device, epochs, pretrain):
    nick_param_counts, nick_final_accuracies, nick_exp_names = [], [], []
    nick_infer_times, nick_mem_usages = [], []
    nick_activations = {}

    print("\n=== Running Nick experiment ===")
    for exp_name, collapse_range in experiments.items():
        print(f"\nRunning Nick experiment: {exp_name}")

        base_model = VGG16_CIFAR10()
        base_model.load_state_dict(torch.load(model_path_000)['model'])

        # Pre-finetune step (always run once)
        _, _, _, _ = run_experiment(
            base_model, train_loader, test_loader, device, epochs+pretrain,
            "Nick", exp_name, pretrain=pretrain
        )

        if collapse_range is not None:
            print(f"Applying compression for {exp_name}...")
            tmp_path = "temp_model.pth"
            torch.save({'model': base_model.state_dict()}, tmp_path)
            base_model = collapse_only(
                model_weights_1=tmp_path,
                compression_set=[collapse_range],
                model_class=VGG16_CIFAR10
            )
            os.remove(tmp_path)

        ckpt_path = get_checkpoint_filename("Nick", exp_name, 'VGG16', pretrain, epochs)
        print(f"[•] Checking for existing checkpoint at: {ckpt_path}")

        if os.path.exists(ckpt_path):
            try:
                print(f"[i] Found existing checkpoint for Nick {exp_name}: {ckpt_path}. Loading instead of re-training.")
                model_ckpt = load_model_from_checkpoint(ckpt_path, collapse_range, device)
                param_count = count_trainable_params(model_ckpt)
                final_acc = evaluate(model_ckpt, test_loader, device)
                infer_time, mem_usage = benchmark_model(model_ckpt, test_loader, device)
            except Exception as e:
                print(f"[!] Failed to load checkpoint for Nick {exp_name} (will retrain): {e}")
                param_count, final_acc, infer_time, mem_usage = run_experiment(
                    base_model, train_loader, test_loader, device, epochs,
                    "Nick", exp_name, pretrain=pretrain
                )
                model_ckpt = load_model_from_checkpoint(ckpt_path, collapse_range, device)
        else:
            param_count, final_acc, infer_time, mem_usage = run_experiment(
                base_model, train_loader, test_loader, device, epochs,
                "Nick", exp_name, pretrain=pretrain
            )
            model_ckpt = load_model_from_checkpoint(ckpt_path, collapse_range, device)

        nick_activations[exp_name] = get_conv_activations(model_ckpt, test_loader, device)

        nick_param_counts.append(param_count)
        nick_final_accuracies.append(final_acc)
        nick_infer_times.append(infer_time)
        nick_mem_usages.append(mem_usage)
        nick_exp_names.append(exp_name)

    plot_results(
        nick_param_counts, nick_final_accuracies, nick_exp_names,
        "Nick Experiment", "nick_experiment_results.svg",
        infer_times=nick_infer_times, mem_usages=nick_mem_usages
    )
    return nick_activations
def run_kevin_experiment(experiments, model_path_000, train_loader, test_loader, device, epochs, pretrain):
    kevin_param_counts, kevin_final_accuracies, kevin_exp_names = [], [], []
    kevin_infer_times, kevin_mem_usages = [], []
    kevin_activations = {}

    print("\n=== Running Kevin experiment ===")
    for exp_name, collapse_range in experiments.items():
        print(f"\nRunning Kevin experiment: {exp_name}")

        base_model = VGG16_CIFAR10()
        base_model.load_state_dict(torch.load(model_path_000)['model'])

        if collapse_range is not None:
            base_model = collapse_only(
                model_weights_1=model_path_000,
                compression_set=[collapse_range],
                model_class=VGG16_CIFAR10
            )

        # try multiple checkpoint names if needed
        possible_ckpts = [
            get_checkpoint_filename("Kevin", exp_name, 'VGG16', pretrain, epochs+pretrain),
            get_checkpoint_filename("Kevin", exp_name, 'VGG16', pretrain, epochs),
        ]
        ckpt_found = next((ck for ck in possible_ckpts if os.path.exists(ck)), None)

        if ckpt_found:
            try:
                print(f"[i] Found existing checkpoint for Kevin {exp_name}: {ckpt_found}. Loading instead of re-training.")
                model_ckpt = load_model_from_checkpoint(ckpt_found, collapse_range, device)
                param_count = count_trainable_params(model_ckpt)
                final_acc = evaluate(model_ckpt, test_loader, device)
                infer_time, mem_usage = benchmark_model(model_ckpt, test_loader, device)
            except Exception as e:
                print(f"[!] Failed to load checkpoint for Kevin {exp_name} (will retrain): {e}")
                param_count, final_acc, infer_time, mem_usage = run_experiment(
                    base_model, train_loader, test_loader, device, epochs+pretrain,
                    "Kevin", exp_name, pretrain=pretrain
                )
                ckpt_new = get_checkpoint_filename("Kevin", exp_name, 'VGG16', pretrain, epochs+pretrain)
                model_ckpt = load_model_from_checkpoint(ckpt_new, collapse_range, device)
        else:
            param_count, final_acc, infer_time, mem_usage = run_experiment(
                base_model, train_loader, test_loader, device, epochs+pretrain,
                "Kevin", exp_name, pretrain=pretrain
            )
            ckpt_new = get_checkpoint_filename("Kevin", exp_name, 'VGG16', pretrain, epochs+pretrain)
            model_ckpt = load_model_from_checkpoint(ckpt_new, collapse_range, device)

        kevin_activations[exp_name] = get_conv_activations(model_ckpt, test_loader, device)

        kevin_param_counts.append(param_count)
        kevin_final_accuracies.append(final_acc)
        kevin_infer_times.append(infer_time)
        kevin_mem_usages.append(mem_usage)
        kevin_exp_names.append(exp_name)

    plot_results(
        kevin_param_counts, kevin_final_accuracies, kevin_exp_names,
        "Kevin Experiment", "kevin_experiment_results.svg",
        infer_times=kevin_infer_times, mem_usages=kevin_mem_usages
    )
    return kevin_activations


def compare_activations(experiments, jf_activations, nick_activations, kevin_activations):
    print("\n=== Activation Similarity Comparison Across Workflows ===")
    for exp_name in experiments.keys():
        sims = {}
        sims["JF-Nick"] = compute_layerwise_similarity(jf_activations[exp_name], nick_activations[exp_name])
        sims["JF-Kevin"] = compute_layerwise_similarity(jf_activations[exp_name], kevin_activations[exp_name])
        sims["Nick-Kevin"] = compute_layerwise_similarity(nick_activations[exp_name], kevin_activations[exp_name])

        save_activation_similarity_json(exp_name, sims)
        plot_activation_similarity(exp_name, sims)

if __name__ == "__main__":
    main()
