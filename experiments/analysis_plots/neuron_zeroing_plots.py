#!/usr/bin/env python3

import json
import os
import pickle
import logging
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import glob
def load_metrics(json_file):
    """
    Load the metrics from a json checkpoint file.
    
    Args:
        json file
        
    Returns:
        dict: The "metrics" dictionary extracted from the checkpoint.
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
            logging.warning(f"No sparsity data for {key}")
    return merged_data

def plot_histogram_accuracy_drop(data, output_dir):
    """
    Plot a histogram of accuracy drop values   across all neurons.
    
    Args:
        data (list of dict): Merged neuron data.
        output_dir (str): Directory to save the plot.
    """
    accuracy_drops = [d['accuracy_drop'] for d in data]
    plt.figure(figsize=(10, 6))
    plt.hist(accuracy_drops, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Accuracy Drop")
    plt.ylabel("Count of Neurons")
    plt.title("Histogram of Neuron Accuracy Drops")
    out_path = os.path.join(output_dir, "histogram_accuracy_drop.png")
    plt.savefig(out_path)
    plt.close()
    logging.info(f"Saved histogram of accuracy drops to {out_path}")

def plot_histogram_by_layer(data, output_dir):
    """
    Plot a histogram of accuracy drop values for each layer (color-coded).
    
    Args:
        data (list of dict): Merged neuron data.
        output_dir (str): Directory to save the plot.
    """
    layers = list(set(d['layer'] for d in data))
    plt.figure(figsize=(10, 6))
    for layer in layers:
        layer_data = [d['accuracy_drop'] for d in data if d['layer'] == layer]
        plt.hist(layer_data, bins=30, alpha=0.5, label=layer)
    plt.xlabel("Accuracy Drop")
    plt.ylabel("Count of Neurons")
    plt.title("Histogram of Neuron Accuracy Drops by Layer")
    plt.legend(title="Layer")
    out_path = os.path.join(output_dir, "histogram_accuracy_drop_by_layer.png")
    plt.savefig(out_path)
    plt.close()
    logging.info(f"Saved histogram of accuracy drops by layer to {out_path}")

def plot_scatter_sparsity_accuracy(data, output_dir):
    """
    Create a scatter plot of sparsity vs. accuracy drop, with points color-coded by layer.
    
    Args:
        data (list of dict): Merged neuron data.
        output_dir (str): Directory to save the plot.
    """
    layers = sorted(set(d['layer'] for d in data))
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.get_cmap('tab10', len(layers))
    for i, layer in enumerate(layers):
        x = [d['sparsity'] for d in data if d['layer'] == layer]
        y = [d['accuracy_drop'] for d in data if d['layer'] == layer]
        plt.scatter(x, y, color=cmap(i), label=layer, alpha=0.7)
    plt.xlabel("Sparsity")
    plt.ylabel("Accuracy Drop  ")
    plt.title("Scatter Plot of Sparsity vs. Accuracy Drop by Layer")
    plt.legend(title="Layer")
    plt.xticks(rotation=45)
    out_path = os.path.join(output_dir, "scatter_sparsity_vs_accuracy_drop.png")
    plt.savefig(out_path)
    plt.close()
    logging.info(f"Saved scatter plot of sparsity vs. accuracy drop to {out_path}")

def plot_boxplot_accuracy_by_layer(data, output_dir):
    """
    Create a boxplot of accuracy drop values grouped by layer.
    
    Args:
        data (list of dict): Merged neuron data.
        output_dir (str): Directory to save the plot.
    """
    layer_dict = defaultdict(list)
    for d in data:
        layer_dict[d['layer']].append(d['accuracy_drop'])
    layers = sorted(layer_dict.keys())
    data_to_plot = [layer_dict[layer] for layer in layers]
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot, labels=layers)
    plt.xlabel("Layer")
    plt.ylabel("Accuracy Drop")
    plt.title("Boxplot of Accuracy Drop by Layer")
    out_path = os.path.join(output_dir, "boxplot_accuracy_drop_by_layer.png")
    plt.savefig(out_path)
    plt.close()
    logging.info(f"Saved boxplot of accuracy drop by layer to {out_path}")
    return (layers, data_to_plot)
def plot_2d_histogram(data, output_dir):
    """
    Create a 2D histogram (heatmap) of sparsity vs. accuracy drop.
    
    Args:
        data (list of dict): Merged neuron data.
        output_dir (str): Directory to save the plot.
    """
    x = [d['sparsity'] for d in data]
    y = [d['accuracy_drop'] for d in data]
    plt.figure(figsize=(10, 6))
    plt.hist2d(x, y, bins=30, cmap='viridis')
    plt.xlabel("Sparsity")
    plt.ylabel("Accuracy Drop  ")
    plt.title("2D Histogram of Sparsity vs. Accuracy Drop")
    plt.colorbar(label="Count")
    out_path = os.path.join(output_dir, "2d_histogram_sparsity_vs_accuracy_drop.png")
    plt.savefig(out_path)
    plt.close()
    logging.info(f"Saved 2D histogram of sparsity vs. accuracy drop to {out_path}")

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    model_names = ["LeNet", "ResNet20", "Vgg16"]
    pretrain = "3"
    finetune = "3"
    steps = "5"
    batch = "128" 
    # List of model names to process (update as needed)
    for model_name in model_names:
        # Update the file paths as needed.
        # Note: The pickle file is assumed to be named "neuron_zeroing.pkl"
        metrics_files =  f"/scratch/jgafur/LTH_output/{model_name}_pretrain{pretrain}_finetune{finetune}_steps{steps}_batch{batch}_devicecuda/"
        print(f"globing: {metrics_files}")
        csv_data = {}
        try:
            for metrics_file in glob.glob(metrics_files):
                step = metrics_file.split("/")[-1].split("_")[-1].split(".")[1]
                logging.info(f"Loading metrics from {metrics_file}")
                merged_data = merge_neuron_metrics(load_metrics(metrics_file))
                
                if not merged_data:
                    logging.error("No merged neuron data available. Exiting.")
                    continue
                
                # Generate plots
                plots_dir = os.path.join(".", f"plots/{model_name}/NeuronZeroing/0.{step}",)
                os.makedirs(plots_dir, exist_ok=True)
                plot_histogram_accuracy_drop(merged_data, plots_dir)
                plot_histogram_by_layer(merged_data, plots_dir)
                plot_scatter_sparsity_accuracy(merged_data, plots_dir)
                csv_data[step] = plot_boxplot_accuracy_by_layer(merged_data, plots_dir)
                plot_2d_histogram(merged_data, plots_dir)
                
                logging.info(f"All plots for {model_name} have been generated and saved in {plots_dir}.")
        except Exception as e:
            print(f"Error: {e}")
            continue
        # Save the boxplot data to a CSV file [step, pruning_layer, average_accuracy_drop, std_accuracy, min_accuracy, max_accuracy]
        csv_file = os.path.join(".", f"plots/{model_name}/NeuronZeroing/boxplot_accuracy_drop_by_layer.csv")
        with open(csv_file, 'w') as f:
            f.write("step,pruning_layer,average_accuracy_drop,std_accuracy,min_accuracy,max_accuracy\n")
            for step, (layers, data) in csv_data.items():
                for layer, values in zip(layers, data):
                    f.write(f"{np.round(float("."+step),4)},{layer},{np.mean(values)},{np.std(values)},{np.min(values)},{np.max(values)}\n")
        import pandas as pd
        x = pd.read_csv(csv_file)
        x.to_latex(csv_file.replace(".csv",".tex"), index=False)
if __name__ == "__main__":
    main()
