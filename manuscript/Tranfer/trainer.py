import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np

import logging
import matplotlib
import matplotlib.pyplot as plt

# Set the logging level for matplotlib to WARNING or ERROR to suppress debug/info messages
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def get_sample_for_each_class(dataloader, num_classes):
    class_samples = {i: None for i in range(num_classes)}  # Initialize for valid classes
    for inputs, targets in dataloader:
        for i, target in enumerate(targets):
            target = target.item()
            # Check if target is within the expected range of classes
            if target >= num_classes:
                print(f"Warning: Found an unexpected class {target} in the dataset. Ignoring.")
                continue  # Skip invalid class labels
            if class_samples[target] is None:
                class_samples[target] = inputs[i]
            # Check if all classes have at least one sample
            if all(sample is not None for sample in class_samples.values()):
                return class_samples
    return class_samples


def initialize_model_and_data(args):
    """Initialize the model and dataset based on the provided arguments."""
    model_class = args.model
    dataset = args.dataset
    model_kwargs = {}

    # Ensure InceptionNet is not used with JF experiments
    if model_class == InceptionNet and args.JF:
        raise ValueError("JF experiments are not supported for InceptionNet.")

    train_loader, test_loader, input_size, input_channels, num_classes = load_dataset(dataset, model_class)
    model_kwargs["num_classes"] = num_classes
    model_kwargs["one_batch"] = next(iter(load_dataset(dataset, model_class)[0]))[0]

    return train_loader, test_loader, model_class, model_kwargs, input_size, input_channels, num_classes

def train_one_epoch(model, train_loader, optimizer, device):
    """Train the model for one epoch."""
    total_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for inputs, targets in tqdm.tqdm(train_loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy

def evaluate_model(model, test_loader, device):
    """Evaluate the model on the test dataset."""
    total_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()
    model.to(device)

    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy

# Helper functions
def create_optimizer_scheduler(model, learning_rate=1e-3):
    """Creates an optimizer and scheduler for model training."""
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return optimizer, scheduler

def save_ckascores_to_json(cka_scores, layer_names, save_dir):
    """Save CKA scores and corresponding layer names to a JSON file."""
    
    # Extract string representations of the layers for serialization
    layer_names_serializable = [
        (cka_score, str(start_layer), str(end_layer))
        for cka_score, start_layer, end_layer in layer_names
    ]
    
    data = {
        "cka_scores": cka_scores,
        "layer_names": layer_names_serializable
    }
    
    print(data)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "cka_scores.json"), "w") as f:
        json.dump(data, f, indent=4)

def get_activations(model, group, dataloader, device):
    activations = []

    def hook_fn(module, input, output):
        flattened_output = output.view(output.size(0), -1)
        summed_output = flattened_output.sum(dim=1, keepdim=True)
        activations.append(summed_output)

    hooks = [module.register_forward_hook(hook_fn) for _, module in group]

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            model.to(device)
            model(inputs)

    for hook in hooks:
        hook.remove()

    print("Calculating activation variance...")
    for idx, act in enumerate(activations):
        variance = torch.var(act, dim=0)
        print(f"Activation {idx} variance: {variance.mean().item()}")

    return torch.cat(activations, dim=0)

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import gc

def calculate_central_kernel_alignment(model, test_dataloader, save_dir, grouping, device):
    """
    Memory-efficient CKA calculation using streaming HSIC components.
    """
    model.to(device)
    model.eval()

    # 1. Collect layers of interest
    all_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            all_layers.append((name, module))

    # 2. Group layers
    grouped_layers = [all_layers[i:i + grouping] for i in range(0, len(all_layers), grouping)]
    
    cka_scores = []
    layer_names_metadata = []

    # 3. Compute CKA between adjacent groups using Streaming Logic
    # This prevents storing all activations for the whole dataset at once
    for idx in range(len(grouped_layers) - 1):
        group_a = grouped_layers[idx]
        group_b = grouped_layers[idx + 1]
        
        print(f"Comparing Group {idx} and Group {idx+1}...")
        
        score = compute_cka_streaming(model, group_a, group_b, test_dataloader, device)
        cka_scores.append(score)
        layer_names_metadata.append((score, group_a[0][0], group_b[-1][0]))

    # 4. Visualization Phase (using single samples per class)
    # We use a lower number of classes if the dataset is huge to keep the plot readable
    num_classes_to_plot = len(set(test_dataloader.dataset.targets)) if hasattr(test_dataloader.dataset, 'targets') else 10
    class_samples = get_sample_for_each_class(test_dataloader, num_classes_to_plot)
    
    plot_activations_for_classes_efficient(grouped_layers, class_samples, model, save_path=save_dir+"/activations_plot.png")
    
    return cka_scores, layer_names_metadata

def compute_cka_streaming(model, group1, group2, dataloader, device):
    """
    Computes Linear CKA batch-by-batch to avoid OutOfMemory errors.
    """
    sum_hsic_kl = 0.0
    sum_hsic_kk = 0.0
    sum_hsic_ll = 0.0
    
    with torch.no_grad():
        for inputs, _ in tqdm.tqdm(dataloader, desc="Streaming HSIC", leave=False):
            inputs = inputs.to(device)
            
            # Extract activations for this specific batch
            # We take the output of the LAST layer in each group as the representative feature
            act1 = get_group_output_batch(model, group1, inputs).view(inputs.size(0), -1)
            act2 = get_group_output_batch(model, group2, inputs).view(inputs.size(0), -1)
            
            # Compute N x N Gram matrices
            K = torch.mm(act1, act1.T)
            L = torch.mm(act2, act2.T)
            
            # Centering matrix H = I - 1/n
            n = K.shape[0]
            H = torch.eye(n, device=device) - torch.ones((n, n), device=device) / n
            K_c = H @ K @ H
            L_c = H @ L @ H
            
            # Accumulate HSIC components (trace(K'L'))
            sum_hsic_kl += torch.trace(K_c @ L_c).item()
            sum_hsic_kk += torch.trace(K_c @ K_c).item()
            sum_hsic_ll += torch.trace(L_c @ L_c).item()

            # Cleanup batch memory
            del act1, act2, K, L, K_c, L_c
            torch.cuda.empty_cache()

    # Final CKA Calculation
    cka_score = sum_hsic_kl / (np.sqrt(sum_hsic_kk * sum_hsic_ll) + 1e-8)
    return max(0.0, min(1.0, cka_score))

def get_group_output_batch(model, group, inputs):
    """Hooks into the last layer of a group to get the batch activations."""
    activations = []
    def hook_fn(module, input, output):
        activations.append(output)
    
    # We only need the hook on the very last layer of the group
    last_layer_name, last_layer_module = group[-1]
    handle = last_layer_module.register_forward_hook(hook_fn)
    
    model(inputs)
    handle.remove()
    return activations[0]

def plot_activations_for_classes_efficient(group_list, class_samples, model, save_path="activations_plot.png"):
    """
    Visualizes the mean 'energy' of activations to show where the network is looking.
    """
    num_samples = len(class_samples)
    num_groups = len(group_list)
    device = next(model.parameters()).device
    
    fig, axes = plt.subplots(num_samples, num_groups + 1, figsize=(20, num_samples * 3))
    model.eval()

    for row_idx, (class_idx, img_tensor) in enumerate(class_samples.items()):
        if img_tensor is None: continue
        
        # Plot Original Image
        ax_img = axes[row_idx, 0]
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
        # Simple min-max scaling for visualization
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        ax_img.imshow(img_np)
        ax_img.set_title(f"Class {class_idx}")
        ax_img.axis('off')

        # Plot Activation Heatmaps for each Group
        for col_idx, group in enumerate(group_list):
            with torch.no_grad():
                # Get the output of the last layer in this group
                raw_act = get_group_output_batch(model, group, img_tensor.unsqueeze(0).to(device))
                act = raw_act.squeeze(0).detach().cpu()
                
                # Spatial mapping: Average across channels
                if act.ndim == 3: # (C, H, W)
                    heatmap = torch.mean(act, dim=0).numpy()
                else: # (D,) for Linear layers
                    # Reshape flattened vector into a square for visual context
                    val = act.view(-1).numpy()
                    side = int(np.ceil(np.sqrt(val.size)))
                    heatmap = np.pad(val, (0, side**2 - val.size)).reshape(side, side)

            # Normalize the heatmap for better color contrast
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            ax_feat = axes[row_idx, col_idx + 1]
            ax_feat.imshow(heatmap, cmap='viridis', interpolation='bilinear')
            ax_feat.axis('off')
            if row_idx == 0:
                ax_feat.set_title(f"Group {col_idx}")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Visualization saved to {save_path}")
    plt.close()
    gc.collect()

def get_sample_for_each_class(dataloader, num_classes):
    """Helper to pick one representative image per class."""
    samples = {}
    for inputs, targets in dataloader:
        for i in range(len(targets)):
            label = targets[i].item()
            if label not in samples and label < num_classes:
                samples[label] = inputs[i]
            if len(samples) >= num_classes:
                return dict(sorted(samples.items()))
    return dict(sorted(samples.items()))