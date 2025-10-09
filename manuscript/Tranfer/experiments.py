import os
import torch
import json
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from collections import OrderedDict
import torch.nn as nn
from pyPrune.models.Vgg16 import VGG16  # Ensure this import matches your project structure
from utils import *
from filemanager import *
from collapse import collapse_only
from trainer import train_and_evaluate
from plots import plot_accuracy_loss_curve, plot_results
from pyPrune.utils import *
import glob

def run_experiment(model, model_kwargs=None, train_loader=None, test_loader=None, device='cuda',
                   epochs=10, workflow='default', exp_name='experiment', collapse_range=None,
                   data_shape=(1, 3, 32, 32), save_path="./runs", post_compress_epochs=False):
    
    # Paths
    ckpt_dir = os.path.join(save_path, "checkpoints")
    metrics_dir = os.path.join(save_path, "metrics")
    plots_dir = os.path.join(save_path, "plots")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, get_checkpoint_filename(workflow, exp_name, model.__class__.__name__, epochs))
    model.to(device)

    describe_model(model, loader=train_loader, device=device)

    # Check if metrics already exist
    glob_path = os.path.join(metrics_dir, f"{workflow}/*metrics.json")
    json_paths = glob.glob(glob_path)
    data = None
    if json_paths:
        json_path = json_paths[0]
        with open(json_path, "r") as f:
            all_metrics = json.load(f)
            if exp_name in all_metrics.get(list(all_metrics.keys())[0], {}):
                print(f"[✓] Found existing results for '{exp_name}' in {json_path}, skipping training.")
                data = all_metrics[list(all_metrics.keys())[0]][exp_name]
                plot_accuracy_loss_curve(data['accuracies'], data['losses'], workflow, exp_name, save_dir=plots_dir)

    # Train if needed
    if data is None:
        print(f"[•] Training model: {exp_name}")
        data = train_and_evaluate(model, train_loader, test_loader, device, epochs, post_compress_epochs=post_compress_epochs)
    else:
        print(f"[✓] Skipping training for '{exp_name}' as results already exist.")

    # Save model checkpoint
    torch.save({'model': model.state_dict()}, ckpt_path)

    # Save plots
    plot_accuracy_loss_curve(data['accuracies'], data['losses'], workflow, exp_name, save_dir=plots_dir)

    # Benchmarking
    param_count = count_trainable_params(model)
    infer_time, mem_usage = benchmark_model(model, test_loader, device)

    data.update({
        "param_count": param_count,
        "inference_time": infer_time,
        "memory_usage": mem_usage,
        "final_accuracy": data["accuracies"][-1] if data["accuracies"] else 0,
    })

    save_metrics_json(f"{workflow}/{model.__class__.__name__}_postcomp_{post_compress_epochs}", exp_name, data, base_dir=metrics_dir)

    # Final save
    final_path = os.path.join(ckpt_dir, f"final_{os.path.basename(ckpt_path)}")
    torch.save({'model': model.state_dict()}, final_path)

    del model
    return data


def run_jf_experiment(experiments, model_path_097, train_loader, test_loader, device, epochs, pretrain,
                      model_class=VGG16, model_kwargs=None, data_shape=None, save_path="./runs",
                      post_compress_epochs=False):
    
    model_kwargs = model_kwargs or {}
    print("\n=== Running JF experiment ===")
    exp_name, collapse_range = list(experiments.items())[0]
    print(f"\nRunning JF experiment: {exp_name}")

    base_model = model_class(**model_kwargs)
    base_model.load_state_dict(torch.load(model_path_097)['model'])

    if collapse_range is not None:
        base_model = collapse_only(
            model_weights_1=model_path_097,
            compression_set=[collapse_range],
            model_class=model_class,
            model_kwargs=model_kwargs,
            input_shape=model_kwargs['one_batch'].shape,
            device=device
        )

    data = run_experiment(base_model, model_kwargs, train_loader, test_loader, device, epochs,
                          workflow="JF", exp_name=exp_name, data_shape=data_shape,
                          save_path=save_path, post_compress_epochs=post_compress_epochs)

    # Return model for further pruning
    return base_model

def run_kevin_experiment(experiments, model_path_000, train_loader, test_loader, device, epochs,
                         model_class=VGG16, model_kwargs=None, data_shape=None, save_path="./runs",
                         post_compress_epochs=False):
    
    model_kwargs = model_kwargs or {}
    print("\n=== Running Kevin experiment ===")
    exp_name, collapse_range = list(experiments.items())[0]
    print(f"\nRunning Kevin experiment: {exp_name}")

    base_model = model_class(**model_kwargs)
    base_model.load_state_dict(torch.load(model_path_000)['model'])

    if collapse_range is not None:
        tmp_path = os.path.join(save_path, "temp_model.pth")
        torch.save({'model': base_model.state_dict()}, tmp_path)
        base_model = collapse_only(
            model_weights_1=tmp_path,
            compression_set=[collapse_range],
            model_class=model_class,
            model_kwargs=model_kwargs,
            input_shape=model_kwargs['one_batch'].shape,
            device=device
        )
        os.remove(tmp_path)

    data = run_experiment(base_model, model_kwargs, train_loader, test_loader, device, epochs,
                          workflow="Kevin", exp_name=exp_name, data_shape=data_shape,
                          save_path=save_path, post_compress_epochs=post_compress_epochs)

    return base_model

def run_nick_experiment(experiments, model_path_000, train_loader, test_loader, device, epochs, pretrain,
                        model_class=VGG16, model_kwargs=None, data_shape=None, save_path="./runs",
                        post_compress_epochs=False):
    
    model_kwargs = model_kwargs or {}
    print("\n=== Running Nick experiment ===")
    exp_name, collapse_range = list(experiments.items())[0]
    print(f"\nRunning Nick experiment: {exp_name}")

    base_model = model_class(**model_kwargs)
    base_model.load_state_dict(torch.load(model_path_000)['model'])
    base_model.to(device)

    # Check pretraining metrics
    metrics_dir = os.path.join(save_path, "metrics")
    plots_dir = os.path.join(save_path, "plots")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    data = None
    glob_path = os.path.join(metrics_dir, f"Nick/*metrics.json")
    json_paths = glob.glob(glob_path)
    if json_paths:
        json_path = json_paths[0]
        with open(json_path, "r") as f:
            all_metrics = json.load(f)
            if exp_name in all_metrics.get(list(all_metrics.keys())[0], {}):
                print(f"[✓] Found existing pretraining results for '{exp_name}' in {json_path}")
                data = all_metrics[list(all_metrics.keys())[0]][exp_name]
                plot_accuracy_loss_curve(data['accuracies'], data['losses'], "Nick", exp_name, save_dir=plots_dir)

    if data is None:
        print(f"Training for {pretrain + epochs} epochs (initial)...")
        data = train_and_evaluate(base_model, train_loader, test_loader, device, pretrain + epochs, post_compress_epochs=False)
        save_metrics_json(f"Nick/{model_class.__name__}_pretrain", exp_name, data, base_dir=metrics_dir)

    # Apply compression
    if collapse_range is not None:
        tmp_path = os.path.join(save_path, "temp_model.pth")
        torch.save({'model': base_model.state_dict()}, tmp_path)
        base_model = collapse_only(
            model_weights_1=tmp_path,
            compression_set=[collapse_range],
            model_class=model_class,
            model_kwargs=model_kwargs,
            input_shape=model_kwargs['one_batch'].shape,
            device=device
        )
        os.remove(tmp_path)

    # Fine-tune
    print(f"Fine-tuning for {epochs} epochs...")
    data = run_experiment(base_model, model_kwargs, train_loader, test_loader, device, epochs,
                          workflow="Nick", exp_name=exp_name, data_shape=data_shape,
                          save_path=save_path, post_compress_epochs=post_compress_epochs)

    return base_model
