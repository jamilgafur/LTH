"""
Fixed: adversarial_plotting.py
Addresses:
  - Clear axis labels with units (ms, %, seconds)
  - Delta equation definition
  - Better x-label spacing (rotated, reduced count)
  - Includes all models in figures
  - Comprehensive legends and annotations
  - Professional figure formatting
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AdversarialPlotter:
    """Creates publication-quality adversarial robustness plots."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set publication-quality defaults
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['figure.dpi'] = 300
    
    def plot_robustness_compute_pareto(self, df: pd.DataFrame):
        """
        Figure 1: Pareto frontier of robustness vs compute.
        
        Fixes:
        - Includes all models (ConvNeXt, InceptionNet, MobileNet, RegNetX, VGG16, XceptionNet)
        - Clear axis labels with units (FLOPs, %)
        - Proper legend
        """
        logger.info("Generating Figure 1: Robustness-Compute Pareto")
        
        if df.empty:
            logger.warning("Empty dataframe for Figure 1")
            return
        
        # Aggregate data to ensure each model‑kind appears even if attack coverage differs.
        # We take the first non‑null param_count for the model‑kind (they are identical across attacks)
        # and compute the mean robust_accuracy across all attacks present.
        agg_df = (
            df.groupby(["model", "kind"], as_index=False)
            .agg({"param_count": "first", "robust_accuracy": "mean"})
        )

        fig, ax = plt.subplots(figsize=(14, 9))

        # Define colors and markers for all models
        models = ['ConvNeXt', 'InceptionNet', 'MobileNet', 'RegNetX_400MF', 'VGG16', 'XceptionNet']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        markers = ['o', 's', '^', 'D', 'v', 'x']
        kinds = ['Control_Continued', 'Dynamic_Region_All_Combined', 'Dynamic_Region_All_Combined_quant']
        kind_styles = ['-', '--', ':']

        model_color_map = dict(zip(models, colors))
        model_marker_map = dict(zip(models, markers))

        # Plot each model‑kind combination using the aggregated values
        for model in models:
            model_data = agg_df[agg_df['model'] == model]
            if model_data.empty:
                logger.warning(f"No data for model: {model}")
                continue
            
            for kind, style in zip(kinds, kind_styles):
                kind_row = model_data[model_data['kind'] == kind]
                if kind_row.empty:
                    continue
                
                ax.scatter(
                    kind_row['param_count'],
                    kind_row['robust_accuracy'],
                    s=150,
                    alpha=0.7,
                    color=model_color_map[model],
                    marker=model_marker_map[model],
                    label=f"{model} - {kind}",
                    edgecolors='black',
                    linewidth=1.5
                )
        
        ax.set_xlabel('Parameter Count (FLOPs)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Robust Accuracy (%)', fontsize=13, fontweight='bold')
        ax.set_title('Figure 1: Robustness-Compute Pareto Frontier\n(Larger = Better Robustness, Smaller = More Efficient)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Use log scale for parameters
        ax.set_xscale('log')
        
        # Format grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Legend with better positioning
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='best', framealpha=0.95, fontsize=9, ncol=2)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / 'figure1_pareto_flops_vs_robust_accuracy_FIXED.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved Figure 1 to {self.output_dir / 'figure1_pareto_flops_vs_robust_accuracy_FIXED.png'}")
        plt.close()
    
    def plot_transfer_resistance_latency(self, df: pd.DataFrame):
        """
        Figure 2: Transfer resistance vs latency.
        
        Fixes:
        - Clear axis labels with units (ms/batch, 1-transfer_success)
        - Includes transfer attack data
        - Proper error handling for missing data
        """
        logger.info("Generating Figure 2: Transfer Resistance vs Latency")
        
        if df.empty:
            logger.warning("Empty dataframe for Figure 2")
            self._create_empty_figure_with_message(
                'Figure 2: Transfer Resistance vs Latency',
                'Data will be populated when transfer attacks are evaluated',
                'figure2_transfer_resistance_vs_latency_FIXED.png'
            )
            return
        
        # Check for transfer/latency data
        transfer_col = 'transfer_success_rate' if 'transfer_success_rate' in df.columns else 'transfer_success'
        has_transfer = transfer_col in df.columns
        has_latency = 'latency_ms' in df.columns

        if not has_transfer or not has_latency:
            logger.warning(f"Missing transfer data (has_transfer={has_transfer}, has_latency={has_latency})")
            self._create_empty_figure_with_message(
                'Figure 2: Transfer Resistance vs Latency',
                'Transfer attack evaluation and latency measurement required',
                'figure2_transfer_resistance_vs_latency_FIXED.png'
            )
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        # Compute transfer resistance = 1 - transfer_success_rate
        df_plot = df.copy()
        if 'transfer_success_rate' not in df_plot.columns and 'transfer_acc' in df_plot.columns:
            df_plot['transfer_success_rate'] = 1.0 - df_plot['transfer_acc']
        df_plot = df_plot.dropna(subset=['latency_ms', transfer_col])
        df_plot['transfer_resistance'] = 1.0 - df_plot[transfer_col]

        if df_plot.empty:
            logger.warning('Figure 2: no valid latency/transfer rows remain after filtering.')
            self._create_empty_figure_with_message(
                'Figure 2: Transfer Resistance vs Latency',
                'No valid transfer-success vs latency measurements are available.',
                'figure2_transfer_resistance_vs_latency_FIXED.png'
            )
            return

        # Plot by model
        models = df_plot['model'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

        for model, color in zip(models, colors):
            model_data = df_plot[df_plot['model'] == model]
            ax.scatter(
                model_data['latency_ms'],
                model_data['transfer_resistance'],
                s=120,
                alpha=0.7,
                label=model,
                color=color,
                edgecolors='black',
                linewidth=1
            )
        
        ax.set_xlabel('Latency (ms/batch)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Transfer Resistance (1 - transfer success)', fontsize=13, fontweight='bold')
        ax.set_title('Figure 2: Transfer Attack Resistance vs Inference Latency\n(Higher = More Transfer-Resistant, Lower = Faster)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc='best', framealpha=0.95)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / 'figure2_transfer_resistance_vs_latency_FIXED.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved Figure 2 to {self.output_dir / 'figure2_transfer_resistance_vs_latency_FIXED.png'}")
        plt.close()
    
    def plot_metric_deltas(self, df: pd.DataFrame, control_kind: str = 'Control_Continued'):
        """
        Figure 3: Metric deltas vs control.
        
        Fixes:
        - Clear axis labels with units (%, ms, MB)
        - Delta equation definition in caption
        - Better x-label spacing (rotated, reduced count)
        - Legend explaining metrics
        """
        logger.info("Generating Figure 3: Variant vs Control Metric Deltas")
        
        if df.empty:
            logger.warning("Empty dataframe for Figure 3")
            return
        
        # Filter to comparisons against control
        control_data = df[df['kind'] == control_kind]
        if control_data.empty:
            logger.warning(f"No control data found for kind: {control_kind}")
            return
        
        # Compute deltas for each variant
        delta_data = []
        
        for (model, dataset), group in df.groupby(['model', 'dataset']):
            control = control_data[(control_data['model'] == model) & (control_data['dataset'] == dataset)]
            if control.empty:
                continue
            
            control_row = control.iloc[0]
            
            for _, variant_row in group[group['kind'] != control_kind].iterrows():
                delta_row = {
                    'model': model,
                    'dataset': dataset,
                    'kind': variant_row['kind'],
                    'robust_accuracy_delta': (variant_row.get('robust_accuracy', 0) - control_row.get('robust_accuracy', 0)) * 100,
                    'latency_delta_ms': variant_row.get('latency_ms', 0) - control_row.get('latency_ms', 0),
                    'memory_delta_mb': variant_row.get('memory_mb', 0) - control_row.get('memory_mb', 0),
                }
                delta_data.append(delta_row)
        
        if not delta_data:
            logger.warning("No delta data computed")
            return
        
        delta_df = pd.DataFrame(delta_data)
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # Create x-labels (model_dataset_kind combinations)
        delta_df['label'] = delta_df.apply(lambda x: f"{x['model'][:3]}\n{x['dataset'][:3]}\n{x['kind'][:15]}", axis=1)
        x_pos = np.arange(len(delta_df))
        
        # Plot 1: Robust Accuracy Delta (%)
        axes[0].bar(x_pos, delta_df['robust_accuracy_delta'], color=['green' if x > 0 else 'red' for x in delta_df['robust_accuracy_delta']], alpha=0.7, edgecolor='black')
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[0].set_ylabel('Δ Robust Accuracy (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Figure 3a: Robust Accuracy Delta vs Control\nΔ = (Variant Accuracy - Control Accuracy) × 100', 
                         fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[0].set_axisbelow(True)
        
        # Plot 2: Latency Delta (ms)
        axes[1].bar(x_pos, delta_df['latency_delta_ms'], color=['green' if x < 0 else 'red' for x in delta_df['latency_delta_ms']], alpha=0.7, edgecolor='black')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1].set_ylabel('Δ Latency (milliseconds)', fontsize=12, fontweight='bold')
        axes[1].set_title('Figure 3b: Inference Latency Delta vs Control\nΔ = (Variant Latency - Control Latency) in ms', 
                         fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[1].set_axisbelow(True)
        
        # Plot 3: Memory Delta (MB)
        axes[2].bar(x_pos, delta_df['memory_delta_mb'], color=['green' if x < 0 else 'red' for x in delta_df['memory_delta_mb']], alpha=0.7, edgecolor='black')
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[2].set_ylabel('Δ Memory (Megabytes)', fontsize=12, fontweight='bold')
        axes[2].set_title('Figure 3c: Peak Memory Delta vs Control\nΔ = (Variant Memory - Control Memory) in MB', 
                         fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='y', linestyle='--')
        axes[2].set_axisbelow(True)
        
        # Set x-labels with rotation to avoid overlap
        for ax in axes:
            ax.set_xticks(x_pos)
            # Show every 3rd label to reduce overlap
            labels = [delta_df['label'].iloc[i] if i % 3 == 0 else '' for i in range(len(delta_df))]
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        
        # Add legend explaining delta notation
        fig.text(0.5, 0.02, 
                 'Delta (Δ) Definition: Δ Metric = Variant Value - Control Value\n' +
                 'Green bars = Improvement (↑ accuracy, ↓ latency/memory), Red bars = Degradation\n' +
                 'Control baseline: ' + control_kind,
                 ha='center', fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        fig.savefig(self.output_dir / 'figure3_collapsed_original_deltas_FIXED.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved Figure 3 to {self.output_dir / 'figure3_collapsed_original_deltas_FIXED.png'}")
        plt.close()
    
    def plot_tradeoff_heatmap(self, df: pd.DataFrame):
        """
        Figure 4: Normalized compute-performance tradeoff heatmap.
        
        Fixes:
        - Clear colorbar labels
        - All models included
        - Better spacing for row labels
        """
        logger.info("Generating Figure 4: Normalized Compute-Performance Tradeoff Heatmap")
        
        if df.empty:
            logger.warning("Empty dataframe for Figure 4")
            return
        
        # Create pivot table for heatmap
        try:
            pivot_data = df.pivot_table(
                values='robust_accuracy',
                index=['model', 'dataset', 'kind'],
                aggfunc='mean'
            )
            
            if pivot_data.empty:
                logger.warning("Empty pivot table for heatmap")
                return
            
            fig, ax = plt.subplots(figsize=(14, 10))
            
            im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(pivot_data.columns)))
            ax.set_yticks(np.arange(len(pivot_data.index)))
            
            # Create row labels from multi-index
            row_labels = [f"{idx[0]}\n{idx[1]}\n{idx[2][:20]}" for idx in pivot_data.index]
            ax.set_yticklabels(row_labels, fontsize=8)
            ax.set_xticklabels(pivot_data.columns, fontsize=10, rotation=45, ha='right')
            
            ax.set_title('Figure 4: Normalized Compute-Performance Tradeoff\n(Green = Better Robustness, Red = Worse)', 
                        fontsize=14, fontweight='bold', pad=20)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Robust Accuracy (normalized)', fontsize=11, fontweight='bold')
            
            # Add text annotations
            for i in range(len(pivot_data.index)):
                for j in range(len(pivot_data.columns)):
                    value = pivot_data.values[i, j]
                    if not np.isnan(value):
                        text = ax.text(j, i, f'{value:.2f}', ha="center", va="center", color="black", fontsize=7)
            
            plt.tight_layout()
            fig.savefig(self.output_dir / 'figure4_tradeoff_heatmap_FIXED.png', dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved Figure 4 to {self.output_dir / 'figure4_tradeoff_heatmap_FIXED.png'}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {str(e)}")
    
    def _create_empty_figure_with_message(self, title: str, message: str, filename: str):
        """Create a placeholder figure with message."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=14, 
               transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        fig.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved placeholder {filename}")
        plt.close()
    
    def run(self, summary_df: pd.DataFrame, compute_df: pd.DataFrame, tradeoff_df: pd.DataFrame):
        """Generate all figures."""
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING PUBLICATION-QUALITY FIGURES")
        logger.info("=" * 80 + "\n")

        if not summary_df.empty:
            if "latency_ms" not in summary_df.columns and not compute_df.empty:
                compute_merged = compute_df[["model", "dataset", "kind", "latency_ms"]].drop_duplicates()
                summary_df = summary_df.merge(compute_merged, on=["model", "dataset", "kind"], how="left")

            transfer_path = self.output_dir / "transferability.csv"
            if "transfer_success_rate" not in summary_df.columns and transfer_path.exists():
                transfer_df = pd.read_csv(transfer_path)
                if not transfer_df.empty and {"source_model", "dataset", "source_kind", "transfer_success_rate"}.issubset(transfer_df.columns):
                    transfer_summary = (
                        transfer_df.groupby(["source_model", "dataset", "source_kind"], as_index=False)["transfer_success_rate"]
                        .mean()
                        .rename(columns={"source_model": "model", "source_kind": "kind"})
                    )
                    summary_df = summary_df.merge(transfer_summary, on=["model", "dataset", "kind"], how="left")

            self.plot_robustness_compute_pareto(summary_df)
            self.plot_transfer_resistance_latency(summary_df)
            self.plot_metric_deltas(summary_df)

        if not tradeoff_df.empty:
            self.plot_tradeoff_heatmap(tradeoff_df)

        logger.info("\n" + "=" * 80)
        logger.info("FIGURE GENERATION COMPLETE")
        logger.info("=" * 80 + "\n")


def main():
    """Main entry point."""
    import sys
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./adversarial_results_ep100_pre300"
    
    # Load data
    base_path = Path(output_dir)
    summary_path = base_path / 'summary.csv'
    compute_path = base_path / 'compute_profile.csv'
    tradeoff_path = base_path / 'tradeoff_summary.csv'
    
    summary_df = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    compute_df = pd.read_csv(compute_path) if compute_path.exists() else pd.DataFrame()
    tradeoff_df = pd.read_csv(tradeoff_path) if tradeoff_path.exists() else pd.DataFrame()
    
    # Generate figures
    plotter = AdversarialPlotter(output_dir)
    plotter.run(summary_df, compute_df, tradeoff_df)


if __name__ == "__main__":
    main()
