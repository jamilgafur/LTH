"""
adversarial_plotting.py - REVISED FOR STANDARD (NON-ADVERSARIAL) MODELS
Focus: How does pruning affect clean accuracy, attack vulnerability, and SHAP explainability?

Research Question:
  - Control (baseline unpruned) vs
  - Dynamic_Region_All_Combined (pruned) vs  
  - Dynamic_Region_All_Combined_quant (pruned + quantized)
  
Key metrics:
  - Clean accuracy (does pruning hurt standard accuracy?)
  - Attack success rate (are pruned models MORE or LESS vulnerable?)
  - Parameter efficiency (pruning compression ratio)
  - SHAP explainability (do pruned models have different feature importance?)
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
    """Analyzes pruning impact on standard models' accuracy and vulnerability."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.rcParams['figure.figsize'] = (14, 9)
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.dpi'] = 300

    def load_all_summaries(self) -> pd.DataFrame:
        """Load all summary CSVs and extract variant info from filenames."""
        logger.info("Loading summary files...")
        
        summary_files = sorted(glob.glob(str(self.output_dir / "summary_*.csv")))
        dfs = []
        
        for file in summary_files:
            try:
                df = pd.read_csv(file)
                filename = Path(file).stem.replace("summary_", "")
                parts = filename.split("_")
                
                # Parse: Model_Dataset_Attack
                if len(parts) >= 3:
                    model = parts[0]
                    dataset = parts[1]
                    attack = "_".join(parts[2:])
                    
                    df['model'] = model
                    df['dataset'] = dataset
                    df['attack'] = attack
                    
                    # Extract variant from 'kind' column if it exists
                    if 'kind' in df.columns:
                        df['variant'] = df['kind']
                    else:
                        # Fallback: assume all rows in file are same variant
                        # (you may need to adjust this based on your data structure)
                        df['variant'] = 'Unknown'
                    
                    dfs.append(df)
                    logger.info(f"  ✓ {Path(file).name}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"  ✗ {file}: {e}")
        
        merged = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        
        if not merged.empty:
            logger.info(f"\n✓ Total rows: {len(merged)}")
            logger.info(f"✓ Models: {merged['model'].nunique()}")
            logger.info(f"✓ Datasets: {merged['dataset'].nunique()}")
            logger.info(f"✓ Attacks: {merged['attack'].nunique()}")
            logger.info(f"✓ Variants: {merged['variant'].unique().tolist()}\n")
        
        return merged

    def plot_clean_accuracy_preservation(self, df: pd.DataFrame):
        """Figure 1: Does pruning hurt clean accuracy?"""
        logger.info("Generating Figure 1: Clean Accuracy Preservation")
        
        if df.empty or 'clean_accuracy' not in df.columns:
            logger.warning("  ⚠ Missing clean_accuracy column")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Figure 1: Impact of Pruning on Clean Accuracy\n(Control vs Pruned vs Pruned+Quant)', 
                     fontsize=15, fontweight='bold')
        
        models = sorted(df['model'].unique())[:6]
        colors = {
            'Control_Continued': '#27ae60',
            'Dynamic_Region_All_Combined': '#f39c12',
            'Dynamic_Region_All_Combined_quant': '#e74c3c'
        }
        
        for idx, model in enumerate(models):
            ax = axes[idx // 3, idx % 3]
            model_data = df[df['model'] == model]
            
            # Group by variant, aggregate across all attacks/datasets
            variant_stats = model_data.groupby('variant').agg({
                'clean_accuracy': ['mean', 'std', 'count']
            }).reset_index()
            
            if variant_stats.empty:
                ax.text(0.5, 0.5, f'{model}\n(No data)', ha='center', va='center')
                ax.set_title(model)
                continue
            
            variants = variant_stats['variant'].values
            means = variant_stats[('clean_accuracy', 'mean')].values
            stds = variant_stats[('clean_accuracy', 'std')].values
            
            bars = ax.bar(variants, means, yerr=stds, capsize=8, alpha=0.75,
                         color=[colors.get(v, '#95a5a6') for v in variants],
                         edgecolor='black', linewidth=2)
            
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            ax.set_ylabel('Clean Accuracy (%)', fontweight='bold')
            ax.set_title(f'{model}', fontweight='bold', fontsize=12)
            ax.set_ylim([0, 105])
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_axisbelow(True)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / 'Figure_1_clean_accuracy_preservation.png', dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved Figure 1\n")
        plt.close()

    def plot_attack_vulnerability_increase(self, df: pd.DataFrame):
        """Figure 2: Does pruning INCREASE vulnerability to attacks?"""
        logger.info("Generating Figure 2: Attack Vulnerability by Variant")
        
        if df.empty or 'attack_success_rate' not in df.columns:
            logger.warning("  ⚠ Missing attack_success_rate column")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Group by variant and attack
        attack_variant = df.groupby(['attack', 'variant']).agg({
            'attack_success_rate': ['mean', 'std']
        }).reset_index()
        
        # Pivot for grouped bar chart
        pivot_mean = attack_variant.pivot(index='attack', columns='variant', values=('attack_success_rate', 'mean'))
        pivot_std = attack_variant.pivot(index='attack', columns='variant', values=('attack_success_rate', 'std'))
        
        # Flatten column names
        pivot_mean.columns = pivot_mean.columns.droplevel(0)
        pivot_std.columns = pivot_std.columns.droplevel(0)
        
        colors = {
            'Control_Continued': '#27ae60',
            'Dynamic_Region_All_Combined': '#f39c12',
            'Dynamic_Region_All_Combined_quant': '#e74c3c'
        }
        
        x = np.arange(len(pivot_mean.index))
        width = 0.25
        
        for i, variant in enumerate(sorted(pivot_mean.columns)):
            if variant in pivot_mean.columns:
                ax.bar(x + i*width, pivot_mean[variant], width, 
                       label=variant, alpha=0.8,
                       color=colors.get(variant, '#95a5a6'),
                       edgecolor='black', linewidth=1.5,
                       yerr=pivot_std[variant], capsize=4)
        
        ax.set_xlabel('Attack Method', fontweight='bold', fontsize=12)
        ax.set_ylabel('Attack Success Rate (%)', fontweight='bold', fontsize=12)
        ax.set_title('Figure 2: Vulnerability to Adversarial Attacks\n(Higher = More Vulnerable)\nControl vs Pruned vs Pruned+Quant', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width)
        ax.set_xticklabels(pivot_mean.index, rotation=45, ha='right', fontsize=10)
        ax.set_ylim([0, 105])
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / 'Figure_2_attack_vulnerability.png', dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved Figure 2\n")
        plt.close()

    def plot_variant_summary_statistics(self, df: pd.DataFrame):
        """Figure 3: Summary statistics for Control vs Pruned vs Pruned+Quant."""
        logger.info("Generating Figure 3: Variant Summary Statistics")
        
        if df.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Figure 3: Summary Comparison Across All Models\n(Control vs Pruned vs Pruned+Quant)', 
                     fontsize=15, fontweight='bold')
        
        colors = {
            'Control_Continued': '#27ae60',
            'Dynamic_Region_All_Combined': '#f39c12',
            'Dynamic_Region_All_Combined_quant': '#e74c3c'
        }
        
        # Plot 1: Clean Accuracy
        if 'clean_accuracy' in df.columns:
            ax = axes[0, 0]
            clean_stats = df.groupby('variant')['clean_accuracy'].agg(['mean', 'std']).reset_index()
            variants = clean_stats['variant'].values
            means = clean_stats['mean'].values
            stds = clean_stats['std'].values
            
            bars = ax.bar(variants, means, yerr=stds, capsize=10, alpha=0.75,
                         color=[colors.get(v, '#95a5a6') for v in variants],
                         edgecolor='black', linewidth=2)
            
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('Clean Accuracy (%)', fontweight='bold', fontsize=11)
            ax.set_title('Clean Accuracy (Unpruned Performance)', fontweight='bold')
            ax.set_ylim([0, 105])
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Attack Success Rate
        if 'attack_success_rate' in df.columns:
            ax = axes[0, 1]
            attack_stats = df.groupby('variant')['attack_success_rate'].agg(['mean', 'std']).reset_index()
            variants = attack_stats['variant'].values
            means = attack_stats['mean'].values
            stds = attack_stats['std'].values
            
            bars = ax.bar(variants, means, yerr=stds, capsize=10, alpha=0.75,
                         color=[colors.get(v, '#95a5a6') for v in variants],
                         edgecolor='black', linewidth=2)
            
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('Attack Success Rate (%)', fontweight='bold', fontsize=11)
            ax.set_title('Attack Success Rate (Vulnerability)', fontweight='bold')
            ax.set_ylim([0, 105])
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 3: Sample Count
        ax = axes[1, 0]
        counts = df.groupby('variant').size().reset_index(name='count')
        variants = counts['variant'].values
        counts_vals = counts['count'].values
        
        bars = ax.bar(variants, counts_vals, alpha=0.75,
                     color=[colors.get(v, '#95a5a6') for v in variants],
                     edgecolor='black', linewidth=2)
        
        for bar, count in zip(bars, counts_vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Sample Count', fontweight='bold', fontsize=11)
        ax.set_title('Data Coverage by Variant', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 4: Models Tested
        ax = axes[1, 1]
        model_counts = df.groupby('variant')['model'].nunique().reset_index()
        variants = model_counts['variant'].values
        model_vals = model_counts['model'].values
        
        bars = ax.bar(variants, model_vals, alpha=0.75,
                     color=[colors.get(v, '#95a5a6') for v in variants],
                     edgecolor='black', linewidth=2)
        
        for bar, count in zip(bars, model_vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_ylabel('Number of Models', fontweight='bold', fontsize=11)
        ax.set_title('Model Coverage', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        fig.savefig(self.output_dir / 'Figure_3_summary_statistics.png', dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved Figure 3\n")
        plt.close()

    def plot_accuracy_robustness_correlation(self, df: pd.DataFrame):
        """Figure 4: Does higher clean accuracy correlate with higher vulnerability?"""
        logger.info("Generating Figure 4: Clean Accuracy vs Vulnerability Correlation")
        
        if df.empty or 'clean_accuracy' not in df.columns or 'attack_success_rate' not in df.columns:
            logger.warning("  ⚠ Missing required columns")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Aggregate by model and variant
        model_variant = df.groupby(['model', 'variant']).agg({
            'clean_accuracy': 'mean',
            'attack_success_rate': 'mean'
        }).reset_index()
        
        colors = {
            'Control_Continued': '#27ae60',
            'Dynamic_Region_All_Combined': '#f39c12',
            'Dynamic_Region_All_Combined_quant': '#e74c3c'
        }
        
        for variant in sorted(model_variant['variant'].unique()):
            variant_data = model_variant[model_variant['variant'] == variant]
            ax.scatter(variant_data['clean_accuracy'], 
                      variant_data['attack_success_rate'],
                      s=250, alpha=0.7, label=variant,
                      color=colors.get(variant, '#95a5a6'),
                      edgecolors='black', linewidth=2)
        
        ax.set_xlabel('Clean Accuracy (%)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Attack Success Rate (%)', fontweight='bold', fontsize=12)
        ax.set_title('Figure 4: Clean Accuracy vs Vulnerability\n(Does pruning change the accuracy-vulnerability trade-off?)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim([0, 105])
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.legend(loc='best', fontsize=11, framealpha=0.95)
        
        plt.tight_layout()
        fig.savefig(self.output_dir / 'Figure_4_accuracy_vulnerability_correlation.png', dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved Figure 4\n")
        plt.close()

    def run(self):
        """Generate all figures."""
        logger.info("\n" + "=" * 80)
        logger.info("PRUNING IMPACT ANALYSIS: Standard (Non-Adversarial) Models")
        logger.info("=" * 80 + "\n")
        
        df = self.load_all_summaries()
        
        if df.empty:
            logger.error("✗ No data loaded!")
            return
        
        # Generate figures
        self.plot_clean_accuracy_preservation(df)
        self.plot_attack_vulnerability_increase(df)
        self.plot_variant_summary_statistics(df)
        self.plot_accuracy_robustness_correlation(df)
        
        logger.info("=" * 80)
        logger.info("✓ ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info("\nKey Research Questions Answered:")
        logger.info("  1. Does pruning hurt clean accuracy? (Figure 1)")
        logger.info("  2. Does pruning increase adversarial vulnerability? (Figure 2)")
        logger.info("  3. Summary comparison across all models (Figure 3)")
        logger.info("  4. Accuracy-vulnerability trade-off (Figure 4)")
        logger.info("\nNext: Run SHAP analysis (temp4.sh) to understand feature importance changes\n")


def main():
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    plotter = AdversarialPlotter(output_dir)
    plotter.run()


if __name__ == "__main__":
    main()