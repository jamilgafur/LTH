import matplotlib.pyplot as plt
import gc
import torch
import random
import numpy as np
import torch


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_loss_accuracy_sparsity(pruner):
    metrics = pruner.metrics
    accuracy = metrics['accuracy']
    loss = metrics['loss']
    sparsity = metrics['sparsity']
    
    # 2x1 subplots: one for Accuracy, one for Loss
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot Accuracy
    ax1.plot(sparsity, accuracy, 'g-', label='Accuracy', linewidth=2)
    ax1.scatter(sparsity, accuracy, c='g', marker='o', s=50)
    ax1.set_ylabel('Accuracy', color='g', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left', fontsize=12)

    # Plot Loss
    ax2.plot(sparsity, loss, 'b-', label='Loss', linewidth=2)
    ax2.scatter(sparsity, loss, c='b', marker='x', s=50)
    ax2.set_xlabel('Sparsity', fontsize=14)
    ax2.set_ylabel('Loss', color='b', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left', fontsize=12)

    # Set title and adjust layout
    plt.suptitle('Loss and Accuracy vs Sparsity', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title

    # Save and show plot
    plt.savefig('sparsity_vs_loss_and_accuracy.png', dpi=300)
    plt.show()