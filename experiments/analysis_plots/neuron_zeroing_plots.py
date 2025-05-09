#!/usr/bin/env python3

import os
import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_pickle(pkl_file):
    with open(pkl_file, 'rb') as file:
        return pickle.load(file)

def merge_neuron_metrics(metrics):
    acc_drops = metrics.get("neuron_accuracy_drops", [])
    sparsity_metrics = metrics.get("sparsity_metrics", [])
    sparsity_dict = {(d['layer_name'], d['neuron_index']): d['sparsity'] for d in sparsity_metrics}
    plt.colormaps.get_cmap('tab10')
    merged = []
    for d in acc_drops:
        key = (d['layer_name'], d['neuron_index'])
        if key in sparsity_dict:
            merged.append({
                "layer": d['layer_name'],
                "neuron_index": d['neuron_index'],
                "accuracy_drop": d['accuracy_drop'],
                "sparsity": sparsity_dict[key]
            })
        else:
            print(f"⚠️ Missing sparsity data for {key}")
    return merged

# -- Plotting Functions --

def plot_violinplot_accuracy_drop_by_step_for_layer(agg_by_layer_step, output_dir):
    layers = sorted(agg_by_layer_step.keys())  # Get sorted list of layers
    n_layers = len(layers)
    
    # Create subplots with enough space for each layer
    fig, axs = plt.subplots(n_layers, 1, figsize=(10, 6 * n_layers))  # One plot per layer
    if n_layers == 1:  # In case there's only one layer, axs is not a list
        axs = [axs]

    # Plot each layer in its respective subplot
    for i, layer in enumerate(layers):
        step_dict = agg_by_layer_step[layer]
        steps = sorted(step_dict, key=lambda x: float(x))
        axs[i].violinplot([step_dict[s] for s in steps], positions=range(1, len(steps) + 1), showmedians=True)
        axs[i].set_xticks(range(1, len(steps) + 1))
        axs[i].set_xticklabels(steps)
        axs[i].set_xlabel("Step")
        axs[i].set_ylabel("Loss Drop")
        axs[i].set_title(f"Violin Plot Across Steps for Layer {layer}")
    
    # Adjust layout to prevent overlapping and save the plot
    plt.tight_layout()
    save_plot(plt, output_dir, "violinplot_loss_drop_by_step_for_all_layers.png")

def save_plot(plt_obj, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    plt_obj.tight_layout()
    plt_obj.savefig(path)
    plt_obj.close()
    print(f"✅ Saved plot to {path}")

# --- Main ----
def main():
    metrics_dirs = glob.glob("/scratch/jgafur/LTH_output/*LeNet_pretrain20_finetune5_steps21_batch128_devicecuda_strategy_magnitud*/")
    for metrics_dir in metrics_dirs:
        print(f"\n📂 Processing directory: {metrics_dir}")
        model_id = os.path.normpath(metrics_dir).split(os.sep)[-1]
        base_dir = os.path.join("plots", model_id, "NeuronZeroing")
        steps_dir = os.path.join(base_dir, "steps")
        combined_dir = os.path.join(base_dir, "combined")
        os.makedirs(combined_dir, exist_ok=True)

        csv_data = {}
        agg_by_layer_step = defaultdict(lambda: defaultdict(list))

        # Gather and group pkl files by step
        pkl_files = glob.glob(os.path.join(metrics_dir, "neuronZeroing_accuracy", "*.pkl"))
        step_to_files = defaultdict(list)
        for pkl_file in pkl_files:
            step = os.path.basename(pkl_file).split("_")[-1].replace(".pkl", "")
            step_to_files[step].append(pkl_file)

        for step, files in step_to_files.items():
            print(f"\n🔢 Processing step: {step}")
            try:
                combined_metrics = {"neuron_accuracy_drops": [], "sparsity_metrics": []}

                for f in files:
                    data = load_pickle(f)
                    combined_metrics["neuron_accuracy_drops"].extend(data['metrics'].get("neuron_accuracy_drops", []))
                    combined_metrics["sparsity_metrics"].extend(data['metrics'].get("sparsity_metrics", []))

                merged_data = merge_neuron_metrics(combined_metrics)
                if not merged_data:
                    print("⚠️ Empty merged data; skipping.")
                    continue

                for d in merged_data:
                    agg_by_layer_step[d['layer']][step].append(d['accuracy_drop'])

                print(f"📈 Step {step} contains {len(merged_data)} data points.")

            except Exception as e:
                print(f"❌ Error processing step {step}: {e}")

        # Save summary stats to CSV
        csv_path = os.path.join(base_dir, "boxplot_accuracy_drop_by_layer.csv")
        with open(csv_path, 'w') as f:
            f.write("step,pruning_layer,avg,std,min,max\n")
            for step, (layers, values_list) in csv_data.items():
                for layer, values in zip(layers, values_list):
                    f.write(f"{step},{layer},{np.mean(values):.4f},{np.std(values):.4f},{np.min(values):.4f},{np.max(values):.4f}\n")

        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            df.to_latex(csv_path.replace(".csv", ".tex"), index=False)
        except ImportError:
            print("📄 Pandas not installed — skipping LaTeX export.")

        # Plot all layers on one figure with multiple subplots
        plot_violinplot_accuracy_drop_by_step_for_layer(agg_by_layer_step, combined_dir)

if __name__ == "__main__":
    main()
