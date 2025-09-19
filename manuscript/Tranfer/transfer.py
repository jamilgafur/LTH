import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from pyPrune.models.Vgg16 import VGG16_CIFAR10
from utils import *
from collections import OrderedDict
import torch.nn.functional as F

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
    if collapse_range:
        print(f"\nCollapsing layers: {collapse_range}")
        model = collapse_block(model, *collapse_range)
    else:
        print("\nNo collapsing applied - original model")
    model.to(device)
    param_count = count_trainable_params(model)
    print(f"Trainable parameters: {param_count}")

    # Initial evaluation before training
    init_acc = evaluate(model, test_loader, device)
    print(f"Initial test accuracy: {init_acc:.2f}%")

    accuracies, final_acc = train_and_evaluate(model, train_loader, test_loader, device, epochs)

    return param_count, init_acc, final_acc, accuracies

def register_hooks(model, layers_to_hook):
    activations = {}
    def hook_fn(module, input, output):
        activations[module] = output.detach()

    hooks = []
    for layer_name, layer in model.named_modules():
        if layer_name in layers_to_hook:
            hooks.append(layer.register_forward_hook(hook_fn))
    
    return activations, hooks

def compute_weight_similarity(original_weights, compressed_weights):
    similarities = {}
    for name in original_weights:
        if name in compressed_weights:
            original = original_weights[name].view(-1)
            compressed = compressed_weights[name].view(-1)
            original = original.cpu().to(torch.float32)
            compressed = compressed.cpu().to(torch.float32)
            similarity = F.cosine_similarity(original, compressed, dim=0).item()
            similarities[name] = similarity
    return similarities


def plot_weight_similarity(weight_similarities, filename):
    # Ensure weight_similarities is a flat dictionary (layer_name -> similarity_value)
    if not isinstance(weight_similarities, dict):
        raise TypeError("weight_similarities must be a dictionary")
    
    # Flatten the dictionary (if there are nested dictionaries)
    flat_similarities = {}
    for layer, similarity in weight_similarities.items():
        # if bias in layer name skip it
        if 'bias' in layer:
            continue
        if isinstance(similarity, dict):  # if similarity is a dictionary, flatten it
            for sub_layer, sub_similarity in similarity.items():
                flat_similarities[f"{layer}_{sub_layer}"] = sub_similarity
        else:
            flat_similarities[layer] = similarity

    # Prepare for plotting
    layers = list(flat_similarities.keys())
    similarities = list(flat_similarities.values())

    plt.figure(figsize=(20, 12))
    plt.barh(layers, similarities, color='green')
    plt.xlabel("Cosine Similarity", fontsize=14)
    plt.title("Weight Similarity for Each Layer (Compression)", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Saved plot: {filename}")


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
    pretrain = 10  # Not used in this script but kept for reference

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

    all_weight_similarities = OrderedDict()
    # ------------------------
    # JF Experiment Workflow
    # ------------------------
    jf_param_counts = []
    jf_final_accuracies = []
    jf_exp_names = []
    all_weight_similarities["JF"] = {}

    print("\n=== Running JF experiment ===")
    print(f"Loading pre-trained model from {model_path_097}...")
    base_model = VGG16_CIFAR10()
    base_model.load_state_dict(torch.load(model_path_097)['model'])

    for exp_name, collapse_range in experiments.items():
        print(f"\nRunning experiment: {exp_name}")
        
        original_weights = base_model.state_dict()

        if collapse_range is not None:
           print(f"Applying compression for {exp_name}...")
            base_model = collapse_only(model_weights_1=model_path_097, compression_set=[collapse_range], model_class=VGG16_CIFAR10)

        print(f"Finetuning the compressed model...")
        param_count, init_acc, final_acc, acc_list = run_experiment(base_model, train_loader, test_loader, device, epochs)

        # After finetuning, capture compressed weights
        compressed_weights = base_model.state_dict()

        # Compute weight similarities
        weight_similarities = compute_weight_similarity(original_weights, compressed_weights)
        all_weight_similarities["JF"][exp_name] = weight_similarities

        jf_param_counts.append(param_count)
        jf_final_accuracies.append(final_acc)
        jf_exp_names.append(exp_name)

    # Plot JF experiment results
    print("\nPlotting JF results...")
    # plot_weight_similarity(all_weight_similarities["JF"], "jf_weight_similarity.svg")
    plot_results(jf_param_counts, jf_final_accuracies, jf_exp_names, "JF Experiment", "jf_experiment_results.svg")


    # ------------------------
    # Kevin Experiment Workflow
    # ------------------------
    kevin_param_counts = []
    kevin_final_accuracies = []
    kevin_exp_names = []
    all_weight_similarities["Kevin"] = {}

    print("\n=== Running Kevin experiment ===")
    print(f"Loading untrained model from {model_path_000}...")
    base_model = VGG16_CIFAR10()
    base_model.load_state_dict(torch.load(model_path_000)['model'])

    for exp_name, collapse_range in experiments.items():
        print(f"\nRunning experiment: {exp_name}")
        
        original_weights = base_model.state_dict()

        if collapse_range is not None:
            print(f"Applying compression for {exp_name}...")
            base_model = collapse_only(model_weights_1=model_path_000, compression_set=[collapse_range], model_class=VGG16_CIFAR10)

        print(f"Finetuning the compressed model...")
        param_count, init_acc, final_acc, acc_list = run_experiment(base_model, train_loader, test_loader, device, epochs+pretrain)

        # After finetuning, capture compressed weights
        compressed_weights = base_model.state_dict()

        # Compute weight similarities
        weight_similarities = compute_weight_similarity(original_weights, compressed_weights)
        all_weight_similarities["Kevin"][exp_name] = weight_similarities

        kevin_param_counts.append(param_count)
        kevin_final_accuracies.append(final_acc)
        kevin_exp_names.append(exp_name)

    # Plot Kevin experiment results
    print("\nPlotting Kevin results...")
    # plot_weight_similarity(all_weight_similarities["Kevin"], "kevin_weight_similarity.svg")
    plot_results(kevin_param_counts, kevin_final_accuracies, kevin_exp_names, "Kevin Experiment", "kevin_experiment_results.svg")


    # ------------------------
    # Nick Experiment Workflow
    # ------------------------
    nick_param_counts = []
    nick_final_accuracies = []
    nick_exp_names = []
    all_weight_similarities["Nick"] = {}

    print("\n=== Running Nick experiment ===")
    print(f"Loading untrained model from {model_path_000}...")
    base_model = VGG16_CIFAR10()
    base_model.load_state_dict(torch.load(model_path_000)['model'])

    for exp_name, collapse_range in experiments.items():
        print(f"\nRunning experiment: {exp_name}")
        
        # Nick workflow: First finetune
        print("First finetuning before compression...")
        param_count, init_acc, final_acc, acc_list = run_experiment(base_model, train_loader, test_loader, device, epochs+pretrain)
        print(f"First finetuning completed. Accuracy: {final_acc:.4f}")

        original_weights = base_model.state_dict()

        if collapse_range is not None:
            print(f"Applying compression for {exp_name}...")
            temp_model_path = "temp_model.pth"
            torch.save({'model': base_model.state_dict()}, temp_model_path)
            base_model = collapse_only(model_weights_1=temp_model_path, compression_set=[collapse_range], model_class=VGG16_CIFAR10)

        print("Second finetuning after compression...")
        param_count, init_acc, final_acc, acc_list = run_experiment(base_model, train_loader, test_loader, device, epochs)
        print(f"Second finetuning completed. Final accuracy: {final_acc:.4f}")

        # After finetuning, capture compressed weights
        compressed_weights = base_model.state_dict()

        # Compute weight similarities
        weight_similarities = compute_weight_similarity(original_weights, compressed_weights)
        all_weight_similarities["Nick"][exp_name] = weight_similarities

        nick_param_counts.append(param_count)
        nick_final_accuracies.append(final_acc)
        nick_exp_names.append(exp_name)

    # Plot Nick experiment results
    print("\nPlotting Nick results...")
    # plot_weight_similarity(all_weight_similarities["Nick"], "nick_weight_similarity.svg")
    plot_results(nick_param_counts, nick_final_accuracies, nick_exp_names, "Nick Experiment", "nick_experiment_results.svg")



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