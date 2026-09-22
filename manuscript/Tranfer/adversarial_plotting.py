"""
Fixed adversarial_plotting.py
Works with individual summary_*.csv files organized by model-dataset-attack
Generates 4 publication-quality figures
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import glob

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
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

    def load_all_summaries(self) -> pd.DataFrame:
        """Load and merge all summary_*.csv files."""
        logger.info("Loading all summary CSV files...")
        
        summary_files = sorted(glob.glob(str(self.output_dir / "summary_*.csv")))
        if not summary_files:
            logger.warning("No summary_*.csv files found!")
            return pd.DataFrame()
        
        dfs = []
        for file in summary_files:
            try:
                df = pd.read_csv(file)
                # Extract model, dataset, attack from filename
                filename = Path(file).stem  # e.g., "summary_ConvNeXt_Cifar10_APGD"
                parts = filename.replace("summary_", "").split("_")
                
                # Parse filename: summary_Model_Dataset_Attack
                if len(parts) >= 3:
                    model = parts[0]  # ConvNeXt
                    dataset = parts[1]  # Cifar10, Cifar100, TinyImageNet
                    attack = "_".join(parts[2:])  # APGD, BIM, CW, etc.
                    
                    df['model'] = model
                    df['dataset'] = dataset
                    df['attack'] = attack
                    
                    dfs.append(df)
                    logger.info(f"  ✓ Loaded {Path(file).name}: {len(df)} rows ({model}, {dataset}, {attack})")
            except Exception as e:
                logger.warning(f"  ✗ Failed to load {file}: {e}")
        
        if not dfs:
            logger.error("No summary files could be loaded!")
            return pd.DataFrame()
        
        merged = pd.concat(dfs, ignore_index=True)
        logger.info(f"\n✓ Total merged rows: {len(merged)}")
        logger.info(f"✓ Columns: {merged.columns.tolist()}\n")
        return merged

    def plot_attack_success_by_model(self, df: pd.DataFrame):
        """Figure 1: Attack success rates by model and dataset."""
        logger.info("Generating Figure 1: Attack Success Rates by Model and Dataset")
        
        if df.empty or 'model' not in df.columns:
            logger.warning("Cannot generate Figure 1: missing data")
            return
        
        # Group by model and compute mean robust accuracy
        model_stats = df.groupby('model').agg({
            'robust_accuracy': ['mean', 'std', 'count'],
            'attack_success_rate': ['mean', 'std']
        }).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: Robust Accuracy by Model
        models = model_stats['model'].values
        robust_mean = model_stats[('robust_accuracy', 'mean')].values
        robust_std = model_stats[('robust_accuracy', 'std')].values
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        ax1.bar(models, robust_mean, yerr=robust_std, capsize=5, alpha=0.7, 
                color=colors[:len(models)], edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Robust Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax1.set_title('Figure 1a: Robust Accuracy by Model\n(Error bars = ±1 std dev)', 
                      fontsize=13, fontweight='bold')
        ax1.set_ylim([0, 105])
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax1.set_axisbelow(True)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Attack Success Rate by Model
        attack_mean = model_stats[('attack_success_rate', 'mean')].values
        attack_std = model_stats[('attack_success_rate', 'std')].values
        
        ax2.bar(models, attack_mean, yerr=attack_std, capsize=5, alpha=0.7, 
                color=colors[:len(models)], edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax2.set_title('Figure 1b: Attack Success Rate by Model\n(Error bars = ±1 std dev)', 
                      fontsize=13, fontweight='bold')
        ax2.set_ylim([0, 105])
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax2.set_axisbelow(True)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / 'Figure_1_attack_success_by_model.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved Figure 1: Figure_1_attack_success_by_model.png\n")
        plt.close()

    def plot_robustness_by_dataset(self, df: pd.DataFrame):
        """Figure 2: Robustness by dataset."""
        logger.info("Generating Figure 2: Robustness by Dataset and Model")
        
        if df.empty or 'dataset' not in df.columns:
            logger.warning("Cannot generate Figure 2: missing data")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Group by dataset and model
        dataset_model = df.groupby(['dataset', 'model'])['robust_accuracy'].agg(['mean', 'std']).reset_index()
        
        # Pivot for grouped bar chart
        pivot_mean = dataset_model.pivot(index='dataset', columns='model', values='mean')
        pivot_std = dataset_model.pivot(index='dataset', columns='model', values='std')
        
        # Create grouped bar chart
        x = np.arange(len(pivot_mean.index))
        width = 0.13
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, (col, color) in enumerate(zip(pivot_mean.columns, colors)):
            offset = (i - len(pivot_mean.columns)/2) * width
            ax.bar(x + offset, pivot_mean[col], width, label=col, 
                   yerr=pivot_std[col], capsize=3, alpha=0.8, color=color, edgecolor='black', linewidth=1)
        
        ax.set_ylabel('Robust Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax.set_title('Figure 2: Robustness Across Datasets and Models\n(Error bars = ±1 std dev)', 
                     fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot_mean.index)
        ax.set_ylim([0, 105])
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / 'Figure_2_robustness_by_dataset.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved Figure 2: Figure_2_robustness_by_dataset.png\n")
        plt.close()

    def plot_attack_comparison(self, df: pd.DataFrame):
        """Figure 3: Attack method comparison."""
        logger.info("Generating Figure 3: Attack Method Comparison")
        
        if df.empty or 'attack' not in df.columns:
            logger.warning("Cannot generate Figure 3: missing attack column")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Group by attack
        attack_stats = df.groupby('attack').agg({
            'robust_accuracy': ['mean', 'std'],
            'attack_success_rate': ['mean', 'std']
        }).reset_index()
        
        attacks = attack_stats['attack'].values
        x = np.arange(len(attacks))
        width = 0.35
        
        # Plot 1: Robust Accuracy by Attack
        robust_mean = attack_stats[('robust_accuracy', 'mean')].values
        robust_std = attack_stats[('robust_accuracy', 'std')].values
        
        ax1.bar(x, robust_mean, width, label='Robust Accuracy', 
                yerr=robust_std, capsize=5, alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Robust Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Attack Method', fontsize=12, fontweight='bold')
        ax1.set_title('Figure 3a: Robust Accuracy by Attack Method\n(Error bars = ±1 std dev)', 
                      fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(attacks, rotation=45, ha='right')
        ax1.set_ylim([0, 105])
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax1.set_axisbelow(True)
        
        # Plot 2: Attack Success Rate by Attack
        attack_mean = attack_stats[('attack_success_rate', 'mean')].values
        attack_std = attack_stats[('attack_success_rate', 'std')].values
        
        ax2.bar(x, attack_mean, width, label='Attack Success Rate', 
                yerr=attack_std, capsize=5, alpha=0.7, color='coral', edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Attack Method', fontsize=12, fontweight='bold')
        ax2.set_title('Figure 3b: Attack Success Rate by Method\n(Error bars = ±1 std dev)', 
                      fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(attacks, rotation=45, ha='right')
        ax2.set_ylim([0, 105])
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax2.set_axisbelow(True)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / 'Figure_3_attack_comparison.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved Figure 3: Figure_3_attack_comparison.png\n")
        plt.close()

    def plot_heatmap_model_dataset_attack(self, df: pd.DataFrame):
        """Figure 4: Heatmap of robustness by model and dataset."""
        logger.info("Generating Figure 4: Robustness Heatmap (Model × Dataset)")
        
        if df.empty or 'model' not in df.columns or 'dataset' not in df.columns:
            logger.warning("Cannot generate Figure 4: missing data")
            return
        
        # Create pivot table: rows=model, cols=dataset, values=mean robust_accuracy
        pivot = df.pivot_table(values='robust_accuracy', index='model', columns='dataset', aggfunc='mean')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, fontsize=11, fontweight='bold')
        ax.set_yticklabels(pivot.index, fontsize=11, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Robust Accuracy (%)', fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.values[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f'{value:.1f}%', ha="center", va="center", 
                                 color="black", fontsize=11, fontweight='bold')
        
        ax.set_title('Figure 4: Robustness Heatmap (Model × Dataset)\nGreen = Better, Red = Worse', 
                     fontsize=13, fontweight='bold', pad=20)
        ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / 'Figure_4_robustness_heatmap.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved Figure 4: Figure_4_robustness_heatmap.png\n")
        plt.close()

    def run(self):
        """Generate all figures."""
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING PUBLICATION-QUALITY FIGURES")
        logger.info("=" * 80 + "\n")
        
        # Load all data
        df = self.load_all_summaries()
        
        if df.empty:
            logger.error("✗ No data loaded. Cannot generate figures.")
            return
        
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Models: {df['model'].unique().tolist()}")
        logger.info(f"Datasets: {df['dataset'].unique().tolist()}")
        logger.info(f"Attacks: {df['attack'].unique().tolist()}\n")
        
        # Generate figures
        self.plot_attack_success_by_model(df)
        self.plot_robustness_by_dataset(df)
        self.plot_attack_comparison(df)
        self.plot_heatmap_model_dataset_attack(df)
        
        logger.info("=" * 80)
        logger.info("✓ FIGURE GENERATION COMPLETE")
        logger.info("=" * 80 + "\n")


def main():
    """Main entry point."""
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    plotter = AdversarialPlotter(output_dir)
    plotter.run()


if __name__ == "__main__":
    main()