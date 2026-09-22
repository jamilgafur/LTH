"""
Fixed adversarial_plotting.py
Works with individual summary_*.csv files organized by model-dataset-attack
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import glob

logger = logging.getLogger(__name__)

class AdversarialPlotter:
    """Creates plots from individual attack summary CSVs."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set publication-quality defaults
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11
        plt.rcParams['dpi'] = 300

    def load_all_summaries(self) -> pd.DataFrame:
        """Load and merge all summary_*.csv files."""
        logger.info("Loading all summary CSV files...")
        
        summary_files = glob.glob(str(self.output_dir / "summary_*.csv"))
        if not summary_files:
            logger.warning("No summary_*.csv files found!")
            return pd.DataFrame()
        
        dfs = []
        for file in summary_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
                logger.info(f"  Loaded {Path(file).name}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"  Failed to load {file}: {e}")
        
        if not dfs:
            return pd.DataFrame()
        
        merged = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total merged rows: {len(merged)}")
        return merged

    def plot_attack_success_by_model(self, df: pd.DataFrame):
        """Figure 1: Attack success rates by model."""
        logger.info("Generating Figure 1: Attack Success Rates by Model")
        
        if df.empty or 'model' not in df.columns:
            logger.warning("Cannot generate Figure 1: missing data")
            return
        
        # Group by model and compute mean robust accuracy
        model_stats = df.groupby('model').agg({
            'robust_accuracy': ['mean', 'std'],
            'attack_success_rate': ['mean', 'std']
        }).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Robust Accuracy by Model
        models = model_stats['model'].values
        robust_mean = model_stats[('robust_accuracy', 'mean')].values
        robust_std = model_stats[('robust_accuracy', 'std')].values
        
        ax1.bar(models, robust_mean, yerr=robust_std, capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_ylabel('Robust Accuracy (%)', fontweight='bold')
        ax1.set_xlabel('Model', fontweight='bold')
        ax1.set_title('Figure 1a: Robust Accuracy by Model', fontweight='bold')
        ax1.set_ylim([0, 100])
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Attack Success Rate by Model
        attack_mean = model_stats[('attack_success_rate', 'mean')].values
        attack_std = model_stats[('attack_success_rate', 'std')].values
        
        ax2.bar(models, attack_mean, yerr=attack_std, capsize=5, alpha=0.7, color='coral', edgecolor='black')
        ax2.set_ylabel('Attack Success Rate (%)', fontweight='bold')
        ax2.set_xlabel('Model', fontweight='bold')
        ax2.set_title('Figure 1b: Attack Success Rate by Model', fontweight='bold')
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / 'Figure_1_attack_success_by_model.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved Figure 1")
        plt.close()

    def plot_robustness_by_dataset(self, df: pd.DataFrame):
        """Figure 2: Robustness by dataset."""
        logger.info("Generating Figure 2: Robustness by Dataset")
        
        if df.empty or 'dataset' not in df.columns:
            logger.warning("Cannot generate Figure 2: missing data")
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Group by dataset and model
        dataset_model = df.groupby(['dataset', 'model'])['robust_accuracy'].mean().unstack()
        
        dataset_model.plot(kind='bar', ax=ax, width=0.8, edgecolor='black')
        ax.set_ylabel('Robust Accuracy (%)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Dataset', fontweight='bold', fontsize=12)
        ax.set_title('Figure 2: Robustness Across Datasets and Models', fontweight='bold', fontsize=13)
        ax.set_ylim([0, 100])
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / 'Figure_2_robustness_by_dataset.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved Figure 2")
        plt.close()

    def plot_attack_comparison(self, df: pd.DataFrame):
        """Figure 3: Attack method comparison."""
        logger.info("Generating Figure 3: Attack Method Comparison")
        
        if df.empty or 'attack' not in df.columns:
            logger.warning("Cannot generate Figure 3: missing attack column")
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Group by attack
        attack_stats = df.groupby('attack').agg({
            'robust_accuracy': 'mean',
            'attack_success_rate': 'mean'
        }).reset_index()
        
        x = np.arange(len(attack_stats))
        width = 0.35
        
        ax.bar(x - width/2, attack_stats['robust_accuracy'], width, label='Robust Accuracy (%)', 
               alpha=0.7, color='steelblue', edgecolor='black')
        ax.bar(x + width/2, attack_stats['attack_success_rate'], width, label='Attack Success Rate (%)', 
               alpha=0.7, color='coral', edgecolor='black')
        
        ax.set_ylabel('Percentage (%)', fontweight='bold', fontsize=12)
        ax.set_xlabel('Attack Method', fontweight='bold', fontsize=12)
        ax.set_title('Figure 3: Attack Method Comparison', fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(attack_stats['attack'], rotation=45, ha='right')
        ax.set_ylim([0, 100])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / 'Figure_3_attack_comparison.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved Figure 3")
        plt.close()

    def plot_heatmap_model_dataset(self, df: pd.DataFrame):
        """Figure 4: Heatmap of robustness by model and dataset."""
        logger.info("Generating Figure 4: Robustness Heatmap")
        
        if df.empty or 'model' not in df.columns or 'dataset' not in df.columns:
            logger.warning("Cannot generate Figure 4: missing data")
            return
        
        # Create pivot table
        pivot = df.pivot_table(values='robust_accuracy', index='model', columns='dataset', aggfunc='mean')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Robust Accuracy (%)', fontweight='bold')
        
        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.values[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f'{value:.1f}', ha="center", va="center", 
                                 color="black", fontsize=10, fontweight='bold')
        
        ax.set_title('Figure 4: Robustness Heatmap (Model × Dataset)', fontweight='bold', fontsize=13)
        ax.set_xlabel('Dataset', fontweight='bold')
        ax.set_ylabel('Model', fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / 'Figure_4_robustness_heatmap.png', dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved Figure 4")
        plt.close()

    def run(self):
        """Generate all figures."""
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING FIGURES")
        logger.info("=" * 80 + "\n")
        
        # Load all data
        df = self.load_all_summaries()
        
        if df.empty:
            logger.error("No data loaded. Cannot generate figures.")
            return
        
        logger.info(f"\nDataFrame shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}\n")
        
        # Generate figures
        self.plot_attack_success_by_model(df)
        self.plot_robustness_by_dataset(df)
        self.plot_attack_comparison(df)
        self.plot_heatmap_model_dataset(df)
        
        logger.info("\n" + "=" * 80)
        logger.info("FIGURE GENERATION COMPLETE")
        logger.info("=" * 80 + "\n")


def main():
    """Main entry point."""
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    plotter = AdversarialPlotter(output_dir)
    plotter.run()


if __name__ == "__main__":
    main()