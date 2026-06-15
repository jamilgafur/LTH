import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import matplotlib.cm as cm
import re
import glob

def sanitize_filename(text):
    """Make a string safe for use in filenames."""
    text = str(text).strip()
    text = text.replace('.', 'p')
    return re.sub(r'[^\w\-]', '_', text)

def plot_metrics_from_csv(df, model_name, batch_size):
    print(f"plot_metrics_from_csv: {df.columns}")
    print(f"plot_metrics_from_csv: {df.head()}")
    
    fig, axs = plt.subplots(2, 4, figsize=(20, 8), sharex=True)
    axs = np.array(axs)

    color_map = plt.get_cmap('tab10')
    thresholds = sorted(df["Threshold"].unique())
    colors = {f"Threshold={th}": color_map(i) for i, th in enumerate(thresholds)}

    metric_titles = {
        "Peak_Memory_MB": "Peak Memory (MB)",
        "Time_s": "Time (s)",
        "CPU_Energy_kWh": "CPU Energy (kWh)",
        "GPU_Energy_kWh": "GPU Energy (kWh)",
    }

    metrics = ["Time_s", "CPU_Energy_kWh", "GPU_Energy_kWh", "Peak_Memory_MB"]

    for i, device_key in enumerate(["cpu", "cuda"]):
        for j, metric in enumerate(metrics):
            ax = axs[i, j] if axs.ndim == 2 else axs[j]

            for threshold in thresholds:
                metric_data = df[(df["Device"] == device_key) & (df["Threshold"] == threshold)].copy()

                # Safely parse stringified lists to actual lists
                metric_values = metric_data[metric].apply(ast.literal_eval).apply(lambda x: [float(i) for i in x])

                sparsity = metric_data["Sparsity"]
                medians = metric_values.apply(np.median)
                q25s = metric_values.apply(lambda x: np.percentile(x, 25))
                q75s = metric_values.apply(lambda x: np.percentile(x, 75))

                label = f"Threshold={threshold}"
                color = colors[label]

                ax.plot(sparsity, medians, marker='o', linestyle='-', label=label, color=color)
                ax.fill_between(sparsity, q25s, q75s, alpha=0.2, color=color)

            if i == 0:
                ax.set_title(metric_titles[metric], fontsize=14)
            if j == 0:
                ax.set_ylabel(device_key.upper(), fontsize=12)

            ax.grid(True)

    for ax in axs[-1, :]:
        ax.set_xlabel("Sparsity")

    for j in range(axs.shape[1]):
        axs[0, j].legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.suptitle(f"{model_name} Performance Metrics (Batch Size: {batch_size})", fontsize=18, y=1.02)

    model_name_clean = sanitize_filename(model_name)
    plot_path = f"./plots/{model_name_clean}_{batch_size}_metrics_plot"
    plt.savefig(f"{plot_path}.png", dpi=300)
    plt.savefig(f"{plot_path}.svg", dpi=300)
    plt.close()


def combine_csv_files(directory_path, model_name, batch_size):
    import glob

    # Match files like: ResNet20_64batchsize_thresholds_0p5_performance_metrics.csv
    pattern = f"{model_name}_{batch_size}batchsize_thresholds_*_performance_metrics.csv"
    full_pattern = os.path.join(directory_path, pattern)

    csv_files = glob.glob(full_pattern)

    if not csv_files:
        print(f"⚠️ No matching CSVs found for {model_name} with batch size {batch_size}")
        return None

    combined_data = pd.DataFrame()

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        combined_data = pd.concat([combined_data, df], ignore_index=True)

    combined_data = combined_data.drop_duplicates()

    output_csv = os.path.join(directory_path, f"{model_name}_{batch_size}batchsize_allthresholds_metrics.csv")
    combined_data.to_csv(output_csv, index=False)
    print(f"✅ Combined metrics saved to {output_csv}")
    
    return combined_data


def process_model_and_batch_size(directory_path, model_name, batch_size):
    combined_data = combine_csv_files(directory_path, model_name, batch_size)

    if combined_data is not None:
        print(f"Plotting for model: {model_name}, batch size: {batch_size}")
        plot_metrics_from_csv(combined_data, model_name, batch_size)


def main():
    import re
    import glob

    directory_path = "./plots/"
    files = glob.glob(os.path.join(directory_path, "*batchsize_thresholds_*_performance_metrics.csv"))

    models_batches = set()

    # Match: Model_BatchSizebatchsize_thresholds_Threshold_performance_metrics.csv
    pattern = re.compile(r"^(.*?)_(\d+)batchsize_thresholds_.*_performance_metrics\.csv$")

    for f in files:
        base = os.path.basename(f)
        match = pattern.match(base)
        if match:
            model, batch = match.groups()
            models_batches.add((model.rstrip('_'), batch))

    if not models_batches:
        print("No valid model/batch combinations found in the CSV files.")
    else:
        print(f"Found models and batch sizes: {models_batches}")

    for model_name, batch_size in sorted(models_batches):
        print(f"Processing {model_name} with batch size {batch_size}...")
        process_model_and_batch_size(directory_path, model_name, batch_size)


if __name__ == "__main__":
    main()
