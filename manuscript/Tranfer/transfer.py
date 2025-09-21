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
from pyPrune.models.Vgg16ImageNet import VGG16_ImageNet
from utils import *
from torchinfo import summary
from collections import OrderedDict
import numpy as np

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

def load_model_from_checkpoint(ckpt_path, collapse_range, device, model_class=VGG16_CIFAR10, model_kwargs=None):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model_kwargs = model_kwargs or {}
    model = model_class(**model_kwargs)
    import pdb; pdb.set_trace()
    if collapse_range is not None:
        model = collapse_block(model, *collapse_range)

    sd = torch.load(ckpt_path)['model']
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model

# -------------------------
# Experiment Runner
# -------------------------
def run_experiment(model_class=VGG16_CIFAR10, model_kwargs=None,
                   train_loader=None, test_loader=None, device='cpu',
                   epochs=10, workflow='default', exp_name='experiment',
                   collapse_range=None, pretrain=0,data_shape=(1,3,32,32)):
    
    model_kwargs = model_kwargs or {}
    model = model_class(**model_kwargs)

    ckpt_path = get_checkpoint_filename(workflow, exp_name, model_class.__name__, pretrain, epochs)

    if collapse_range:
        model = collapse_block(model, *collapse_range)

    model.to(device)
    describe_model(model, input_size=data_shape, device=device)

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


def run_jf_experiment(experiments, model_path_097, train_loader, test_loader, device,
                      epochs, pretrain, model_class=VGG16_CIFAR10, model_kwargs=None,data_shape=None):
    model_kwargs = model_kwargs or {}
    jf_param_counts, jf_final_accuracies, jf_exp_names = [], [], []
    jf_infer_times, jf_mem_usages = [], []
    jf_activations = {}

    print("\n=== Running JF experiment ===")
    for exp_name, collapse_range in experiments.items():
        print(f"\nRunning JF experiment: {exp_name}")

        base_model = model_class(**model_kwargs)
        base_model.load_state_dict(torch.load(model_path_097)['model'])

        if collapse_range is not None:
            base_model = collapse_only(
                model_weights_1=model_path_097,
                compression_set=[collapse_range],
                model_class=model_class,
                model_kwargs=model_kwargs
            )

        param_count, final_acc, infer_time, mem_usage = run_experiment(
            model_class=model_class, model_kwargs=model_kwargs,
            train_loader=train_loader, test_loader=test_loader, device=device,
            epochs=epochs, workflow="JF", exp_name=exp_name, pretrain=pretrain,data_shape=data_shape )

        jf_activations[exp_name] = get_conv_activations(base_model, test_loader, device)

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

def run_nick_experiment(experiments, model_path_000, train_loader, test_loader, device,
                        epochs, pretrain, model_class=VGG16_CIFAR10, model_kwargs=None,data_shape=None):
    model_kwargs = model_kwargs or {}
    nick_param_counts, nick_final_accuracies, nick_exp_names = [], [], []
    nick_infer_times, nick_mem_usages = [], []
    nick_activations = {}

    print("\n=== Running Nick experiment ===")
    for exp_name, collapse_range in experiments.items():
        print(f"\nRunning Nick experiment: {exp_name}")

        base_model = model_class(**model_kwargs)
        base_model.load_state_dict(torch.load(model_path_000)['model'])

        _, _, _, _ = run_experiment(
            model_class=model_class, model_kwargs=model_kwargs,
            train_loader=train_loader, test_loader=test_loader, device=device,
            epochs=epochs+pretrain, workflow="Nick", exp_name=exp_name, pretrain=pretrain,data_shape=data_shape
        )

        if collapse_range is not None:
            tmp_path = "temp_model.pth"
            torch.save({'model': base_model.state_dict()}, tmp_path)
            base_model = collapse_only(
                model_weights_1=tmp_path,
                compression_set=[collapse_range],
                model_class=model_class,
                model_kwargs=model_kwargs
            )
            os.remove(tmp_path)

        param_count, final_acc, infer_time, mem_usage = run_experiment(
            model_class=model_class, model_kwargs=model_kwargs,
            train_loader=train_loader, test_loader=test_loader, device=device,
            epochs=epochs, workflow="Nick", exp_name=exp_name, pretrain=pretrain,data_shape=data_shape
        )

        nick_activations[exp_name] = get_conv_activations(base_model, test_loader, device)

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

def run_kevin_experiment(experiments, model_path_000, train_loader, test_loader, device,
                         epochs, pretrain, model_class=VGG16_CIFAR10, model_kwargs=None,data_shape=None):
    model_kwargs = model_kwargs or {}
    kevin_param_counts, kevin_final_accuracies, kevin_exp_names = [], [], []
    kevin_infer_times, kevin_mem_usages = [], []
    kevin_activations = {}

    print("\n=== Running Kevin experiment ===")
    for exp_name, collapse_range in experiments.items():
        print(f"\nRunning Kevin experiment: {exp_name}")

        base_model = model_class(**model_kwargs)
        base_model.load_state_dict(torch.load(model_path_000)['model'])

        if collapse_range is not None:
            base_model = collapse_only(
                model_weights_1=model_path_000,
                compression_set=[collapse_range],
                model_class=model_class,
                model_kwargs=model_kwargs
            )

     
        param_count, final_acc, infer_time, mem_usage = run_experiment(
            model_class=model_class, model_kwargs=model_kwargs,
            train_loader=train_loader, test_loader=test_loader, device=device,
            epochs=epochs + pretrain, workflow="Kevin", exp_name=exp_name, pretrain=pretrain,data_shape=data_shape
        )

        kevin_activations[exp_name] = get_conv_activations(base_model, test_loader, device)

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

def main(model_path_097, model_path_000, experiments=None, epochs=0, pretrain=0,
         model_class=VGG16_CIFAR10, model_kwargs=None,dataset="Cifar10"):
    import os
    import torch

    if not os.path.exists(model_path_097) or not os.path.exists(model_path_000):
        print("Required model weight files not found. Exiting.")
        return
    if dataset=="Cifar10":
        print("Loading CIFAR-10 data...")
        train_loader, test_loader = load_cifar10()
        input_size = (1, 3, 32, 32)
        default_num_classes = 10
    elif dataset=="TinyImageNet":
        print("Loading Tiny ImageNet data...")
        train_loader, test_loader = load_tiny_imagenet()
        input_size = (1, 3, 64, 64)
        default_num_classes = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_kwargs = model_kwargs or {'num_classes': default_num_classes}  # default

    jf_data = run_jf_experiment(experiments, model_path_097, train_loader, test_loader, device,
                                epochs, pretrain, model_class=model_class, model_kwargs=model_kwargs,data_shape=input_size)

    nick_data = run_nick_experiment(experiments, model_path_000, train_loader, test_loader, device,
                                    epochs, pretrain, model_class=model_class, model_kwargs=model_kwargs,data_shape=input_size)

    kevin_data = run_kevin_experiment(experiments, model_path_000, train_loader, test_loader, device,
                                      epochs, pretrain, model_class=model_class, model_kwargs=model_kwargs,data_shape=input_size)

    compare_activations(experiments, jf_data, nick_data, kevin_data)


if __name__ == "__main__":
    style = "Cifar10"
    if style == "Cifar10":
        model_path_097 = "../structured_study/pruning_checkpoints/Vgg16_pretrain10_finetune30_steps21_batch1024_devicecuda_strategy_magnitude/checkpoint_Finetuned_0.97.pth"
        model_path_000 = "../structured_study/pruning_checkpoints/Vgg16_pretrain10_finetune30_steps21_batch1024_devicecuda_strategy_magnitude/checkpoint_Original_0.00.pth"

        experiments = {
                # "Original Model": None,
                "Stage 4-5": ('conv_8', 'conv_13'),
                # "Stage 2-5": ('conv_3', 'conv_13'),
                # "All Conv Layers": ('conv_1', 'conv_13'),
            }
        main(model_path_097, model_path_000, experiments=experiments, epochs=0, pretrain=0,
            model_class=VGG16_CIFAR10, model_kwargs={'num_classes': 10},dataset="Cifar10")
    elif style == "TinyImageNet":
        model_path_097 = "../structured_study/pruning_checkpoints/Vgg16ImageNet_pretrain10_finetune30_steps21_batch512_devicecuda_strategy_magnitude/checkpoint_Finetuned_0.74.pth"
        model_path_000 = "../structured_study/pruning_checkpoints/Vgg16ImageNet_pretrain10_finetune30_steps21_batch512_devicecuda_strategy_magnitude/checkpoint_Original_0.00.pth"

        experiments = {
                # "Original Model": None,
                "Stage 4-5": ('conv_8', 'conv_13'),
                # "Stage 2-5": ('conv_3', 'conv_13'),
                # "All Conv Layers": ('conv_1', 'conv_13'),
            }
        main(model_path_097, model_path_000, experiments=experiments, epochs=0, pretrain=0,
            model_class=VGG16_ImageNet, model_kwargs={'num_classes': 200},dataset="TinyImageNet")