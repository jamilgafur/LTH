import os
import torch
import json
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from collections import OrderedDict
import torch.nn as nn
from pyPrune.models.Vgg16 import VGG16
from utils import *
from filemanager import *
from collapse import collapse_only
from trainer import train_and_evaluate
from plots import plot_accuracy_loss_curve, plot_results
from pyPrune.utils import *

def run_experiment(model, model_kwargs=None,
                   train_loader=None, test_loader=None, device='cpu',
                   epochs=10, workflow='default', exp_name='experiment',
                   collapse_range=None, data_shape=(1,3,32,32),
                   save_path="./runs", post_compress_epochs=False):
    
    # Checkpoint directory & filename
    ckpt_dir = os.path.join(save_path, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, get_checkpoint_filename(workflow, exp_name, model.__class__.__name__, epochs))

    model.to(device)
    describe_model(model, input_size=(1, model_kwargs['input_channels'], *data_shape), device=device)

    print(f"[•] Training model: {exp_name}")

    # === Check if metrics JSON already exists for this experiment ===
    metrics_dir = os.path.join(save_path, "metrics")
    json_path = os.path.join(metrics_dir, f"{workflow}_metrics.json")

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            all_metrics = json.load(f)

        if workflow in all_metrics and exp_name in all_metrics[workflow]:
            print(f"[✓] Found existing results for '{exp_name}' in {json_path}, skipping experiment.")
            result = all_metrics[workflow][exp_name]
            return result

    # === Run training and evaluation ===
    data = train_and_evaluate(model, train_loader, test_loader, device, epochs, post_compress_epochs=post_compress_epochs)

    # Save checkpoint
    torch.save({'model': model.state_dict()}, ckpt_path)

    # Save plots
    plots_dir = os.path.join(save_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_accuracy_loss_curve(
        data['accuracies'], data['losses'], workflow, exp_name,
        save_dir=plots_dir
    )

    param_count = count_trainable_params(model)
    infer_time, mem_usage = benchmark_model(model, test_loader, device)

    # Update metrics and save to JSON
    os.makedirs(metrics_dir, exist_ok=True)
    data.update({
        "param_count": param_count,
        "inference_time": infer_time,
        "memory_usage": mem_usage,
        "final_accuracy": data["accuracies"][-1] if data["accuracies"] else 0,
    })

    save_metrics_json(
        f"{workflow}/{model.__class__.__name__}_postcomp_{post_compress_epochs}", exp_name,
        data, base_dir=metrics_dir
    )

    # Save final model
    final_path = os.path.join(ckpt_dir, f"final_{os.path.basename(ckpt_path)}")
    torch.save({'model': model.state_dict()}, final_path)
    del model

    return {
        "param_count": param_count,
        "final_accuracy": data["accuracies"][-1] if data["accuracies"] else 0,
        "infer_time": infer_time,
        "mem_usage": mem_usage
    }

def run_jf_experiment(experiments, model_path_097, train_loader, test_loader, device, epochs, pretrain, model_class=VGG16, model_kwargs=None, data_shape=None, save_path="./runs", post_compress_epochs=False):

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
                model_kwargs=model_kwargs,
                input_shape=(1, model_kwargs['input_channels'], *data_shape)
            )

        data = run_experiment(
            model=base_model,
            model_kwargs=model_kwargs,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            workflow="JF",
            exp_name=exp_name,
            data_shape=data_shape,
            save_path=save_path,
            post_compress_epochs=post_compress_epochs
        )
        param_count = data["param_count"]
        final_acc = data["final_accuracy"]
        infer_time = data["infer_time"]
        mem_usage = data["mem_usage"]
        
        jf_param_counts.append(param_count)
        jf_final_accuracies.append(final_acc)
        jf_infer_times.append(infer_time)
        jf_mem_usages.append(mem_usage)
        jf_exp_names.append(exp_name)

    plot_dir = os.path.join(save_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_results(
        jf_param_counts, jf_final_accuracies, jf_exp_names,
        "JF Experiment", os.path.join(plot_dir, "jf_experiment_results.svg"),
        infer_times=jf_infer_times, mem_usages=jf_mem_usages
    )
    return jf_activations

def run_kevin_experiment(experiments, model_path_000, train_loader, test_loader, device, epochs, model_class=VGG16, model_kwargs=None, data_shape=None, save_path="./runs", post_compress_epochs=False):

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
                model_kwargs=model_kwargs,
                input_shape=(1, model_kwargs['input_channels'], *data_shape)
            )

        # Run the experiment
        data = run_experiment(
            model=base_model,
            model_kwargs=model_kwargs,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            workflow="Kevin",
            exp_name=exp_name,
            data_shape=data_shape,
            save_path=save_path,
            post_compress_epochs=post_compress_epochs
        )

        # Merge results
        kevin_param_counts.append(data["param_count"])
        kevin_final_accuracies.append(data["final_accuracy"])
        kevin_infer_times.append(data["infer_time"])
        kevin_mem_usages.append(data["mem_usage"])
        kevin_exp_names.append(exp_name)

    # Plot results
    plot_dir = os.path.join(save_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_results(
        kevin_param_counts, kevin_final_accuracies, kevin_exp_names,
        "Kevin Experiment", os.path.join(plot_dir, "kevin_experiment_results.svg"),
        infer_times=kevin_infer_times, mem_usages=kevin_mem_usages
    )

    return {
        "param_counts": kevin_param_counts,
        "final_accuracies": kevin_final_accuracies,
        "infer_times": kevin_infer_times,
        "mem_usages": kevin_mem_usages
    }

def run_nick_experiment(experiments, model_path_000, train_loader, test_loader, device,
                        epochs, pretrain, model_class=VGG16, model_kwargs=None, data_shape=None,
                        save_path="./runs", post_compress_epochs=False):

    model_kwargs = model_kwargs or {}
    nick_param_counts, nick_final_accuracies, nick_exp_names = [], [], []
    nick_infer_times, nick_mem_usages = [], []

    print("\n=== Running Nick experiment ===")
    for exp_name, collapse_range in experiments.items():
        print(f"\nRunning Nick experiment: {exp_name}")

        # 1. Load initial model weights (model_path_000)
        base_model = model_class(**model_kwargs)
        base_model.load_state_dict(torch.load(model_path_000)['model'])
        base_model.to(device)
        # 2. Train the model for (pretrain + epochs) epochs with initial weights
        print(f"Training for {pretrain + epochs} epochs (initial weights)...")
        train_and_evaluate(base_model, train_loader, test_loader, device, pretrain + epochs, post_compress_epochs=False)

        # 3. Apply compression (collapse) if collapse_range is provided
        if collapse_range is not None:
            print("Applying compression...")
            tmp_path = os.path.join(save_path, "temp_model.pth")
            torch.save({'model': base_model.state_dict()}, tmp_path)  # Save the current model weights
            base_model = collapse_only(
                model_weights_1=tmp_path,
                compression_set=[collapse_range],
                model_class=model_class,
                model_kwargs=model_kwargs,
                input_shape=(1, model_kwargs['input_channels'], *data_shape)
            )
            base_model.to(device)
            os.remove(tmp_path)  # Remove temporary model file after collapse

        # 4. Fine-tune the compressed model (train again for the remaining epochs)
        print(f"Fine-tuning for {epochs} epochs after compression...")
        data = run_experiment(
            model=base_model,
            model_kwargs=model_kwargs,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            workflow="Nick",
            exp_name=exp_name,
            data_shape=data_shape,
            save_path=save_path,
            post_compress_epochs=post_compress_epochs
        )

        # Merge the fine-tuned model's results (data)
        nick_param_counts.append(data["param_count"])
        nick_final_accuracies.append(data["final_accuracy"])
        nick_infer_times.append(data["infer_time"])
        nick_mem_usages.append(data["mem_usage"])
        nick_exp_names.append(f"{exp_name}_finetuned")

    # Plot results
    plot_dir = os.path.join(save_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_results(
        nick_param_counts, nick_final_accuracies, nick_exp_names,
        "Nick Experiment", os.path.join(plot_dir, "nick_experiment_results.svg"),
        infer_times=nick_infer_times, mem_usages=nick_mem_usages
    )

    return {
        "param_counts": nick_param_counts,
        "final_accuracies": nick_final_accuracies,
        "infer_times": nick_infer_times,
        "mem_usages": nick_mem_usages
    }
