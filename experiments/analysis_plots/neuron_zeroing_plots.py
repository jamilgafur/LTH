#!/usr/bin/env python3

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pyPrune.utils import plot_loss_accuracy_sparsity, set_seed, lr_lambda

def load_metrics(json_file):
    """
    Load the metrics from a json checkpoint file.
    
    Args:
        json_file (str): File path to a JSON file.
        
    Returns:
        dict: The loaded JSON data.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

def merge_neuron_metrics(metrics):
    """
    Merge neuron accuracy drops with corresponding sparsity metrics.
    
    Both lists are merged on 'layer_name' and 'neuron_index'. Each merged record contains:
      - layer: layer name
      - neuron_index: index of the neuron
      - accuracy_drop: the measured accuracy drop (interpreted as loss)
      - sparsity: the sparsity metric for the neuron
      
    Args:
        metrics (dict): The loaded metrics data.
    
    Returns:
        list of dict: Merged neuron data.
    """
    acc_drops = metrics.get("neuron_accuracy_drops", [])
    sparsity_metrics = metrics.get("sparsity_metrics", [])
    
    # Create a lookup dictionary for sparsity using (layer_name, neuron_index) as key.
    sparsity_dict = {(d['layer_name'], d['neuron_index']): d['sparsity'] 
                     for d in sparsity_metrics}
    
    merged_data = []
    for d in acc_drops:
        key = (d['layer_name'], d['neuron_index'])
        if key in sparsity_dict:
            merged_data.append({
                "layer": d['layer_name'],
                "neuron_index": d['neuron_index'],
                "accuracy_drop": d['accuracy_drop'],
                "sparsity": sparsity_dict[key]
            })
        else:
            print(f"No sparsity data for {key}")
    return merged_data

# --- Plotting functions with optional "step" parameter to update filename ---

def plot_histogram_accuracy_drop(data, output_dir, step=None):
    accuracy_drops = [d['accuracy_drop'] for d in data]
    plt.figure(figsize=(10, 6))
    plt.hist(accuracy_drops, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Loss Drop")
    plt.ylabel("Count of Neurons")
    plt.title("Histogram of Neuron Loss Drops" + (f" at Step {step}" if step else ""))
    filename = "histogram_Loss_drop" + (f"_step_{step}" if step else "") + ".png"
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved histogram of Loss drops to {out_path}")

def plot_histogram_by_layer(data, output_dir, step=None):
    layers = list(set(d['layer'] for d in data))
    plt.figure(figsize=(10, 6))
    for layer in layers:
        layer_data = [d['accuracy_drop'] for d in data if d['layer'] == layer]
        plt.hist(layer_data, bins=30, alpha=0.5, label=layer)
    plt.xlabel("Loss Drop")
    plt.ylabel("Count of Neurons")
    plt.title("Histogram of Neuron Loss Drops by Layer" + (f" at Step {step}" if step else ""))
    plt.legend(title="Layer")
    filename = "histogram_Loss_drop_by_layer" + (f"_step_{step}" if step else "") + ".png"
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved histogram of accuracy drops by layer to {out_path}")

def plot_scatter_sparsity_accuracy(data, output_dir, step=None):
    layers = sorted(set(d['layer'] for d in data))
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.get_cmap('tab10', len(layers))
    for i, layer in enumerate(layers):
        x = [d['sparsity'] for d in data if d['layer'] == layer]
        y = [d['accuracy_drop'] for d in data if d['layer'] == layer]
        plt.scatter(x, y, color=cmap(i), label=layer, alpha=0.7)
    plt.xlabel("Sparsity")
    plt.ylabel("Loss Drop")
    plt.title("Scatter Plot of Sparsity vs. Loss Drop by Layer" + (f" at Step {step}" if step else ""))
    plt.legend(title="Layer")
    plt.xticks(rotation=45)
    filename = "scatter_sparsity_vs_Loss_drop" + (f"_step_{step}" if step else "") + ".png"
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved scatter plot of sparsity vs. Loss drop to {out_path}")

def plot_boxplot_accuracy_by_layer(data, output_dir, step=None):
    layer_dict = defaultdict(list)
    for d in data:
        layer_dict[d['layer']].append(d['accuracy_drop'])
    layers = sorted(layer_dict.keys())
    data_to_plot = [layer_dict[layer] for layer in layers]
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot, labels=layers)
    plt.xlabel("Layer")
    plt.ylabel("Loss Drop")
    plt.title("Boxplot of Loss Drop by Layer" + (f" at Step {step}" if step else ""))
    filename = "boxplot_loss_drop_by_layer" + (f"_step_{step}" if step else "") + ".png"
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved boxplot of accuracy drop by layer to {out_path}")
    return (layers, data_to_plot)

def plot_2d_histogram(data, output_dir, step=None):
    x = [d['sparsity'] for d in data]
    y = [d['accuracy_drop'] for d in data]
    plt.figure(figsize=(10, 6))
    plt.hist2d(x, y, bins=30, cmap='viridis')
    plt.xlabel("Sparsity")
    plt.ylabel("Loss Drop")
    plt.title("2D Histogram of Sparsity vs. Loss Drop" + (f" at Step {step}" if step else ""))
    plt.colorbar(label="Count")
    filename = "2d_histogram_sparsity_vs_loss_drop" + (f"_step_{step}" if step else "") + ".png"
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved 2D histogram of sparsity vs. loss drop to {out_path}")

def plot_boxplot_accuracy_by_sparsity_for_layer(data, output_dir, step=None):
    """
    For each layer, group neurons by (rounded) sparsity and create a boxplot of accuracy drops.
    """
    layers = sorted(set(d['layer'] for d in data))
    for layer in layers:
        layer_data = [d for d in data if d['layer'] == layer]
        sparsity_dict = defaultdict(list)
        for d in layer_data:
            key = round(d['sparsity'], 2)
            sparsity_dict[key].append(d['accuracy_drop'])
        sorted_sparsity = sorted(sparsity_dict.keys())
        data_to_plot = [sparsity_dict[s] for s in sorted_sparsity]
        
        plt.figure(figsize=(10, 6))
        plt.boxplot(data_to_plot, labels=[str(s) for s in sorted_sparsity])
        plt.xlabel("Sparsity")
        plt.ylabel("Loss Drop")
        plt.title(f"Boxplot of Loss Drop by Sparsity for Layer {layer}" + (f" at Step {step}" if step else ""))
        filename = f"boxplot_loss_drop_by_sparsity_{layer}" + (f"_step_{step}" if step else "") + ".png"
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path)
        plt.close()
        print(f"Saved boxplot by sparsity for layer {layer} at step {step} to {out_path}")

def plot_violinplot_accuracy_by_sparsity_for_layer(data, output_dir, step=None):
    """
    For each layer, group neurons by (rounded) sparsity and create a violin plot of accuracy drops.
    """
    layers = sorted(set(d['layer'] for d in data))
    for layer in layers:
        layer_data = [d for d in data if d['layer'] == layer]
        sparsity_dict = defaultdict(list)
        for d in layer_data:
            key = round(d['sparsity'], 2)
            sparsity_dict[key].append(d['accuracy_drop'])
        sorted_sparsity = sorted(sparsity_dict.keys())
        data_to_plot = [sparsity_dict[s] for s in sorted_sparsity]
        
        plt.figure(figsize=(10, 6))
        positions = range(1, len(data_to_plot)+1)
        plt.violinplot(data_to_plot, positions=positions, showmedians=True)
        plt.xticks(positions, [str(s) for s in sorted_sparsity])
        plt.xlabel("Sparsity")
        plt.ylabel("Loss Drop")
        plt.title(f"Violin Plot of Loss Drop by Sparsity for Layer {layer}" + (f" at Step {step}" if step else ""))
        filename = f"violinplot_loss_drop_by_sparsity_{layer}" + (f"_step_{step}" if step else "") + ".png"
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path)
        plt.close()
        print(f"Saved violin plot by sparsity for layer {layer} at step {step} to {out_path}")

def plot_violinplot_accuracy_drop_by_step_for_layer(agg_by_layer_step, output_dir):
    """
    For each layer, create a violin plot of accuracy drop distributions across different step levels.
    
    agg_by_layer_step: dict mapping each layer to another dict mapping step -> list of accuracy drops.
    """
    for layer, step_data in agg_by_layer_step.items():
        # Sort steps numerically (assuming they can be converted to float)
        steps = sorted(step_data.keys(), key=lambda x: float(x))
        data_to_plot = [step_data[s] for s in steps]
        plt.figure(figsize=(10, 6))
        positions = range(1, len(data_to_plot)+1)
        plt.violinplot(data_to_plot, positions=positions, showmedians=True)
        plt.xticks(positions, steps)
        plt.xlabel("Step")
        plt.ylabel("Loss Drop")
        plt.title(f"Violin Plot of Loss Drop across Steps for Layer {layer}")
        out_path = os.path.join(output_dir, f"violinplot_loss_drop_by_step_for_layer_{layer}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved aggregated violin plot by step for layer {layer} to {out_path}")

# --- Main function ---
def main():
    model_names = ["LeNet", "ResNet20", "Vgg16"]
 
    
    for model_name in model_names:        
        metrics_files_glob = f"/scratch/jgafur/LTH_output/LeNet_pretrain*_finetune*_steps21_batch*_devicecuda"
        for metrics_files_path in glob.glob(metrics_files_glob):
            base_dir = os.path.join(".", f"plots/{metrics_files_path.split("/")[-1]}/NeuronZeroing")
            steps_dir = os.path.join(base_dir, "steps")
            combined_dir = os.path.join(base_dir, "combined")
            os.makedirs(steps_dir, exist_ok=True)
            os.makedirs(combined_dir, exist_ok=True)
            print(f"Processing metrics files in: {metrics_files_path}")
            csv_data = {}
            # Aggregate data across steps: agg_by_layer_step[layer][step] = list of accuracy drops
            agg_by_layer_step = defaultdict(lambda: defaultdict(list))
            
            for metrics_file in glob.glob(metrics_files_path + "/neuronZeroing_accuracy/*.json"):
                print(f"Processing: {metrics_file}")
                # Extract step from file name (e.g., assuming filename format ..._step.{step}.json)
                step = "0."+metrics_file.split("/")[-1].split("_")[-1].split(".")[1]
                print(f"Working on step: {step}")
                merged_data = merge_neuron_metrics(load_metrics(metrics_file))
                
                if not merged_data:
                    print("No merged neuron data available. Skipping this file.")
                    continue
                
                # Accumulate data for the top-level aggregated plots.
                for d in merged_data:
                    agg_by_layer_step[d['layer']][step].append(d['accuracy_drop'])
                
                # Generate per-step plots (saved in steps_dir) with the step in the filename.
                plot_histogram_accuracy_drop(merged_data, steps_dir, step=step)
                plot_histogram_by_layer(merged_data, steps_dir, step=step)
                plot_scatter_sparsity_accuracy(merged_data, steps_dir, step=step)
                csv_data[step] = plot_boxplot_accuracy_by_layer(merged_data, steps_dir, step=step)
                plot_2d_histogram(merged_data, steps_dir, step=step)
                
                # Per-layer plots by sparsity for this step.
                plot_boxplot_accuracy_by_sparsity_for_layer(merged_data, steps_dir, step=step)
                plot_violinplot_accuracy_by_sparsity_for_layer(merged_data, steps_dir, step=step)
            
            # Save CSV for boxplot data (if desired)
            csv_file = os.path.join(base_dir, "boxplot_accuracy_drop_by_layer.csv")
            with open(csv_file, 'w') as f:
                f.write("step,pruning_layer,average_accuracy_drop,std_accuracy,min_accuracy,max_accuracy\n")
                for step, (layers, data_list) in csv_data.items():
                    for layer, values in zip(layers, data_list):
                        f.write(f"{np.round(float(step),4)},{layer},{np.mean(values)},{np.std(values)},{np.min(values)},{np.max(values)}\n")
            try:
                import pandas as pd
                x = pd.read_csv(csv_file)
                x.to_latex(csv_file.replace(".csv", ".tex"), index=False)
            except ImportError:
                print("Pandas not installed; skipping CSV to LaTeX conversion.")
            
            # Generate top-level aggregated violin plots across steps (saved in combined_dir).
            plot_violinplot_accuracy_drop_by_step_for_layer(agg_by_layer_step, combined_dir)

if __name__ == "__main__":
    main()
