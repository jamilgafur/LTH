import gc
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, List

def get_pruneable_named_parameters(model: torch.nn.Module, prunable_layers: Tuple) -> Tuple[List[str], List[torch.nn.Parameter]]:
    names = []
    params = []
    last_layer = list(model.modules())[-1]
    for name, param in model.named_parameters():
        module_name = name.rsplit('.', 1)[0]
        module = dict(model.named_modules()).get(module_name, None)
        if module != last_layer and 'weight' in name and module and any(isinstance(module, layer) for layer in prunable_layers):
            names.append(name)
            params.append(param)
    return names, params

def get_pruneable_named_modules(model: torch.nn.Module, prunable_layers: Tuple) -> Tuple[List[str], List[torch.nn.Module]]:
    names = []
    modules = []
    last_layer = list(model.modules())[-1]
    for name, module in model.named_modules():
        if module != last_layer and any(isinstance(module, layer) for layer in prunable_layers):
            names.append(name)
            modules.append(module)
    return names, modules

def get_pruneable_modules(model: torch.nn.Module, prunable_layers: Tuple) -> List[torch.nn.Module]:
    modules = []
    last_layer = list(model.modules())[-1]
    for module in model.modules():
        if module != last_layer and any(isinstance(module, layer) for layer in prunable_layers):
            modules.append(module)
    return modules

def clean_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_loss_accuracy_sparsity(pruner) -> None:
    metrics = pruner.metrics
    accuracy = metrics['accuracy']
    loss = metrics['loss']
    sparsity = metrics['sparsity']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(sparsity, accuracy, 'g-', label='Accuracy', linewidth=2)
    ax1.scatter(sparsity, accuracy, c='g', marker='o', s=50)
    ax1.set_ylabel('Accuracy', color='g', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left', fontsize=12)
    
    ax2.plot(sparsity, loss, 'b-', label='Loss', linewidth=2)
    ax2.scatter(sparsity, loss, c='b', marker='x', s=50)
    ax2.set_xlabel('Sparsity', fontsize=14)
    ax2.set_ylabel('Loss', color='b', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left', fontsize=12)
    
    plt.suptitle('Loss and Accuracy vs. Sparsity', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_path = pruner.save_dir + '/sparsity_vs_loss_and_accuracy.png'
    plt.savefig(plot_path, dpi=300)
    plt.show()
