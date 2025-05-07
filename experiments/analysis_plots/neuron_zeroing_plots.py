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

def plot_histogram_accuracy_drop(data, output_dir, step=None):
    drops = [d['accuracy_drop'] for d in data]
    plt.figure(figsize=(10, 6))
    plt.hist(drops, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Loss Drop")
    plt.ylabel("Neuron Count")
    plt.title(f"Histogram of Loss Drops{' at Step ' + str(step) if step else ''}")
    save_plot(plt, output_dir, f"histogram_Loss_drop_step_{step}.png" if step else "histogram_Loss_drop.png")

def plot_histogram_by_layer(data, output_dir, step=None):
    layers = sorted(set(d['layer'] for d in data))
    plt.figure(figsize=(10, 6))
    for layer in layers:
        plt.hist([d['accuracy_drop'] for d in data if d['layer'] == layer], bins=30, alpha=0.5, label=layer)
    plt.xlabel("Loss Drop")
    plt.ylabel("Neuron Count")
    plt.title(f"Loss Drops by Layer{' at Step ' + str(step) if step else ''}")
    plt.legend(title="Layer")
    save_plot(plt, output_dir, f"histogram_Loss_drop_by_layer_step_{step}.png" if step else "histogram_Loss_drop_by_layer.png")

def plot_scatter_sparsity_accuracy(data, output_dir, step=None):
    layers = sorted(set(d['layer'] for d in data))
    plt.figure(figsize=(10, 6))
    cmap = plt.colormaps.get_cmap('tab10')
    for i, layer in enumerate(layers):
        x = [d['sparsity'] for d in data if d['layer'] == layer]
        y = [d['accuracy_drop'] for d in data if d['layer'] == layer]
        plt.scatter(x, y, color=cmap(i), label=layer, alpha=0.7)
    plt.xlabel("Sparsity")
    plt.ylabel("Loss Drop")
    plt.title(f"Sparsity vs Loss Drop by Layer{' at Step ' + str(step) if step else ''}")
    plt.legend()
    save_plot(plt, output_dir, f"scatter_sparsity_vs_Loss_drop_step_{step}.png" if step else "scatter_sparsity_vs_Loss_drop.png")

def plot_boxplot_accuracy_by_layer(data, output_dir, step=None):
    layer_dict = defaultdict(list)
    for d in data:
        layer_dict[d['layer']].append(d['accuracy_drop'])

    layers = sorted(layer_dict)
    data_to_plot = [layer_dict[layer] for layer in layers]
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot, tick_labels=layers)

    plt.xlabel("Layer")
    plt.ylabel("Loss Drop")
    plt.title(f"Boxplot of Loss Drop by Layer{' at Step ' + str(step) if step else ''}")
    save_plot(plt, output_dir, f"boxplot_loss_drop_by_layer_step_{step}.png" if step else "boxplot_loss_drop_by_layer.png")
    return layers, data_to_plot

def plot_2d_histogram(data, output_dir, step=None):
    x = [d['sparsity'] for d in data]
    y = [d['accuracy_drop'] for d in data]
    plt.figure(figsize=(10, 6))
    plt.hist2d(x, y, bins=30, cmap='viridis')
    plt.colorbar(label="Count")
    plt.xlabel("Sparsity")
    plt.ylabel("Loss Drop")
    plt.title(f"2D Histogram of Sparsity vs Loss Drop{' at Step ' + str(step) if step else ''}")
    save_plot(plt, output_dir, f"2d_histogram_sparsity_vs_loss_drop_step_{step}.png" if step else "2d_histogram_sparsity_vs_loss_drop.png")

def plot_boxplot_accuracy_by_sparsity_for_layer(data, output_dir, step=None):
    for layer in sorted(set(d['layer'] for d in data)):
        layer_data = [d for d in data if d['layer'] == layer]
        spar_dict = defaultdict(list)
        for d in layer_data:
            spar_dict[round(d['sparsity'], 2)].append(d['accuracy_drop'])

        sorted_keys = sorted(spar_dict)
        plt.figure(figsize=(10, 6))
        plt.boxplot([spar_dict[k] for k in sorted_keys], labels=[str(k) for k in sorted_keys])
        plt.xlabel("Sparsity")
        plt.ylabel("Loss Drop")
        plt.title(f"Boxplot by Sparsity for Layer {layer} {'at Step ' + str(step) if step else ''}")
        save_plot(plt, output_dir, f"boxplot_loss_drop_by_sparsity_{layer}_step_{step}.png" if step else f"boxplot_loss_drop_by_sparsity_{layer}.png")

def plot_violinplot_accuracy_by_sparsity_for_layer(data, output_dir, step=None):
    for layer in sorted(set(d['layer'] for d in data)):
        layer_data = [d for d in data if d['layer'] == layer]
        spar_dict = defaultdict(list)
        for d in layer_data:
            spar_dict[round(d['sparsity'], 2)].append(d['accuracy_drop'])

        sorted_keys = sorted(spar_dict)
        plt.figure(figsize=(10, 6))
        positions = range(1, len(sorted_keys) + 1)
        plt.violinplot([spar_dict[k] for k in sorted_keys], positions=positions, showmedians=True)
        plt.xticks(positions, [str(k) for k in sorted_keys])
        plt.xlabel("Sparsity")
        plt.ylabel("Loss Drop")
        plt.title(f"Violin Plot by Sparsity for Layer {layer} {'at Step ' + str(step) if step else ''}")
        save_plot(plt, output_dir, f"violinplot_loss_drop_by_sparsity_{layer}_step_{step}.png")

def plot_violinplot_accuracy_drop_by_step_for_layer(agg_by_layer_step, output_dir):
    for layer, step_dict in agg_by_layer_step.items():
        steps = sorted(step_dict, key=lambda x: float(x))
        plt.figure(figsize=(10, 6))
        plt.violinplot([step_dict[s] for s in steps], positions=range(1, len(steps) + 1), showmedians=True)
        plt.xticks(range(1, len(steps) + 1), steps)
        plt.xlabel("Step")
        plt.ylabel("Loss Drop")
        plt.title(f"Violin Plot Across Steps for Layer {layer}")
        save_plot(plt, output_dir, f"violinplot_loss_drop_by_step_for_layer_{layer}.png")

def save_plot(plt_obj, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    plt_obj.tight_layout()
    plt_obj.savefig(path)
    plt_obj.close()
    print(f"✅ Saved plot to {path}")

# --- Main ---
def main():
    metrics_dirs = glob.glob("/scratch/jgafur/LTH_output/*LeNet_pretrain1_finetune1_steps2_batch128_devicecuda_strategy_brain-damage*")

    for metrics_dir in metrics_dirs:
        print(f"\n📂 Processing directory: {metrics_dir}")
        model_id = os.path.normpath(metrics_dir).split(os.sep)[-1]
        base_dir = os.path.join("plots", model_id, "NeuronZeroing")
        steps_dir = os.path.join(base_dir, "steps")
        combined_dir = os.path.join(base_dir, "combined")
        os.makedirs(steps_dir, exist_ok=True)
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
                    combined_metrics["neuron_accuracy_drops"].extend(data.get("neuron_accuracy_drops", []))
                    combined_metrics["sparsity_metrics"].extend(data.get("sparsity_metrics", []))

                merged_data = merge_neuron_metrics(combined_metrics)
                if not merged_data:
                    print("⚠️ Empty merged data; skipping.")
                    continue

                for d in merged_data:
                    agg_by_layer_step[d['layer']][step].append(d['accuracy_drop'])

                print(f"📈 Step {step} contains {len(merged_data)} data points.")
                plot_histogram_accuracy_drop(merged_data, steps_dir, step)
                plot_histogram_by_layer(merged_data, steps_dir, step)
                plot_scatter_sparsity_accuracy(merged_data, steps_dir, step)
                csv_data[step] = plot_boxplot_accuracy_by_layer(merged_data, steps_dir, step)
                plot_2d_histogram(merged_data, steps_dir, step)
                plot_boxplot_accuracy_by_sparsity_for_layer(merged_data, steps_dir, step)
                plot_violinplot_accuracy_by_sparsity_for_layer(merged_data, steps_dir, step)

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

        plot_violinplot_accuracy_drop_by_step_for_layer(agg_by_layer_step, combined_dir)

if __name__ == "__main__":
    main()
