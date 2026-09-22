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


VARIANT_COLORS = {
    'Control_Continuted': '#27ae60',
    'Control_Continued': '#27ae60',
    'Dynamic_Region_All_Combined': '#f39c12',
    'Dynamic_Region_All_Combined_quant': '#e74c3c'
}

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

    def _normalize_summary_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize plotting inputs to the current summary schema."""
        if df.empty:
            return df

        normalized = df.copy()

        if 'clean_accuracy' not in normalized.columns and 'clean_acc' in normalized.columns:
            normalized['clean_accuracy'] = normalized['clean_acc']

        if 'variant' not in normalized.columns and 'kind' in normalized.columns:
            normalized['variant'] = normalized['kind']

        # -----------------------------------------------------------------
        # NOTE ABOUT UNITS
        # -----------------------------------------------------------------
        # The raw CSV stores accuracies and attack success rates as **fractions**
        # in the range [0, 1] (e.g. 0.42 for 42%).  All plotting functions label
        # the y‑axis as a percentage and set limits assuming values up to 100.
        # We convert only once when the values are still in the raw [0, 1] range.
        # If a column is already percentage-scaled then we leave it untouched.
        # -----------------------------------------------------------------
        def _convert_if_fraction(series: pd.Series, eps: float = 1e-9) -> pd.Series:
            if series.empty:
                return series
            non_null = pd.to_numeric(series, errors='coerce').dropna()
            if non_null.empty:
                return series
            if (non_null >= 0).all() and (non_null <= 1.0 + eps).all():
                return non_null.mul(100.0)
            return series

        if 'clean_accuracy' in normalized.columns:
            normalized['clean_accuracy'] = _convert_if_fraction(normalized['clean_accuracy'])

        if 'attack_success_rate' in normalized.columns:
            normalized['attack_success_rate'] = _convert_if_fraction(normalized['attack_success_rate'])

        return normalized

    def load_all_summaries(self) -> pd.DataFrame:
        """Load all summary CSVs and extract variant info from filenames."""
        logger.info("Loading summary files...")
        
        # Search recursively for any summary_*.csv files under the output directory.
        # This works whether the user points to the results folder itself or its
        # parent directory.
        summary_files = sorted(glob.glob(str(self.output_dir / "**/summary_*.csv"), recursive=True))
        dfs = []
        
        for file in summary_files:
            try:
                df = pd.read_csv(file)
                filename = Path(file).stem.replace("summary_", "")
                parts = filename.split("_")

                # Prefer the canonical columns already stored in each summary shard.
                # Only fall back to filename parsing when a column is missing.
                if 'model' not in df.columns and len(parts) >= 1:
                    df['model'] = parts[0]
                if 'dataset' not in df.columns and len(parts) >= 2:
                    df['dataset'] = parts[1]
                if 'attack' not in df.columns and len(parts) >= 3:
                    df['attack'] = "_".join(parts[2:])
                if 'variant' not in df.columns:
                    if 'kind' in df.columns:
                        df['variant'] = df['kind']
                    else:
                        df['variant'] = 'Unknown'

                df = self._normalize_summary_df(df)
                dfs.append(df)
                logger.info(f"  ✓ {Path(file).name}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"  ✗ {file}: {e}")
        
        merged = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        merged = self._normalize_summary_df(merged)
        
        if not merged.empty:
            logger.info(f"\n✓ Total rows: {len(merged)}")
            logger.info(f"✓ Models: {merged['model'].nunique()}")
            logger.info(f"✓ Datasets: {merged['dataset'].nunique()}")
            logger.info(f"✓ Attacks: {merged['attack'].nunique()}")
            logger.info(f"✓ Variants: {merged['variant'].unique().tolist()}\n")
        
        return merged

    @staticmethod
    def _control_variant_name(variant_values):
        """Return the control variant if present, else the first variant in a stable order."""
        variant_values = [str(v) for v in variant_values if pd.notna(v)]
        if not variant_values:
            return None
        for preferred in ['Control_Continuted', 'Control_Continued']:
            if preferred in variant_values:
                return preferred
        return sorted(set(variant_values))[0]

    def plot_clean_accuracy_preservation(self, df: pd.DataFrame):
        """Figure 1: Change in clean accuracy relative to the control variant."""
        logger.info("Generating Figure 1: Clean Accuracy Preservation")
        df = self._normalize_summary_df(df)

        if df.empty or 'clean_accuracy' not in df.columns:
            logger.warning("  ⚠ Missing clean_accuracy column")
            return

        control_variant = self._control_variant_name(df['variant'].dropna().unique())
        if control_variant is None:
            logger.warning("  ⚠ No variant information found for baseline comparison")
            return

        model_variant_stats = df.groupby(['model', 'variant'], as_index=False).agg(
            clean_accuracy_mean=('clean_accuracy', 'mean'),
            clean_accuracy_std=('clean_accuracy', 'std'),
        )
        control_by_model = (
            model_variant_stats[model_variant_stats['variant'] == control_variant]
            [['model', 'clean_accuracy_mean']]
            .rename(columns={'clean_accuracy_mean': 'control_clean_accuracy'})
        )
        plot_data = model_variant_stats.merge(control_by_model, on='model', how='left')
        plot_data['clean_accuracy_delta_from_control'] = (
            plot_data['clean_accuracy_mean'] - plot_data['control_clean_accuracy']
        )
        # For the control variant the delta is zero and the error bar should be zero as well.
        plot_data.loc[plot_data['variant'] == control_variant, 'clean_accuracy_std'] = 0.0
        plot_data.to_csv(self.output_dir / 'Figure_1_clean_accuracy_data.csv', index=False)

        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        fig.suptitle('Figure 1: Change in Clean Accuracy vs Control\n(Control vs Pruned vs Pruned+Quant)', 
                     fontsize=15, fontweight='bold')

        models = sorted(df['model'].unique())[:6]
        colors = VARIANT_COLORS

        for idx, model in enumerate(models):
            ax = axes[idx // 3, idx % 3]
            model_data = plot_data[plot_data['model'] == model].copy()
            if model_data.empty:
                ax.text(0.5, 0.5, f'{model}\n(No data)', ha='center', va='center')
                ax.set_title(model)
                continue

            variants = model_data['variant'].values
            deltas = model_data['clean_accuracy_delta_from_control'].values
            stds = model_data['clean_accuracy_std'].fillna(0).values

            bars = ax.bar(variants, deltas, alpha=0.75,
                         color=[colors.get(v, '#95a5a6') for v in variants],
                         edgecolor='black', linewidth=2)

            for bar, delta in zip(bars, deltas):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                         f'{delta:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

            ax.axhline(0, color='black', linewidth=1.0)
            ax.set_ylabel('Change vs Control (%)', fontweight='bold')
            ax.set_title(f'{model}', fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_axisbelow(True)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

        fig.savefig(self.output_dir / 'Figure_1_clean_accuracy_preservation.png', dpi=100)
        logger.info(f"  ✓ Saved Figure 1\n")
        plt.close()

    def plot_attack_vulnerability_increase(self, df: pd.DataFrame):
        """Figure 2: Change in attack success rate relative to the control variant, broken out per model."""
        logger.info("Generating Figure 2: Attack Vulnerability by Variant (per model)")
        df = self._normalize_summary_df(df)

        if df.empty or 'attack_success_rate' not in df.columns:
            logger.warning("  ⚠ Missing attack_success_rate column")
            return

        control_variant = self._control_variant_name(df['variant'].dropna().unique())
        if control_variant is None:
            logger.warning("  ⚠ No variant information found for baseline comparison")
            return

        # Compute delta ASR per model, attack, variant
        attack_variant = df.groupby(['model', 'attack', 'variant'], as_index=False).agg(
            mean_attack_success_rate=('attack_success_rate', 'mean'),
            std_attack_success_rate=('attack_success_rate', 'std'),
        )
        control_by_model_attack = (
            attack_variant[attack_variant['variant'] == control_variant]
            [['model', 'attack', 'mean_attack_success_rate']]
            .rename(columns={'mean_attack_success_rate': 'control_attack_success_rate'})
        )
        plot_data = attack_variant.merge(control_by_model_attack, on=['model', 'attack'], how='left')
        plot_data['asr_delta_from_control'] = (
            plot_data['mean_attack_success_rate'] - plot_data['control_attack_success_rate']
        )
        plot_data.to_csv(self.output_dir / 'Figure_2_attack_vulnerability_data.csv', index=False)

        # Create a grid of sub‑plots – one per model (max 6 models → 2×3 grid)
        models = sorted(df['model'].unique())
        n_models = len(models)
        ncols = 3
        nrows = (n_models + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
        colors = VARIANT_COLORS

        for idx, model in enumerate(models):
            ax = axes[idx // ncols, idx % ncols]
            model_data = plot_data[plot_data['model'] == model]
            if model_data.empty:
                ax.text(0.5, 0.5, f'{model}\n(No data)', ha='center', va='center')
                ax.set_title(model)
                continue

            pivot_mean = model_data.pivot(index='attack', columns='variant', values='asr_delta_from_control')
            pivot_std = model_data.pivot(index='attack', columns='variant', values='std_attack_success_rate')
            x = np.arange(len(pivot_mean.index))
            width = 0.25
            for i, variant in enumerate(sorted(pivot_mean.columns)):
                if variant in pivot_mean.columns:
                          ax.bar(x + i * width, pivot_mean[variant], width,
                              label=variant, alpha=0.8,
                              color=colors.get(variant, '#95a5a6'),
                              edgecolor='black', linewidth=1.5)
            ax.axhline(0, color='black', linewidth=1.0)
            ax.set_xlabel('Attack', fontsize=10)
            ax.set_ylabel('Δ ASR (pp)', fontsize=10)
            ax.set_title(model, fontsize=11)
            ax.set_xticks(x + width)
            ax.set_xticklabels(pivot_mean.index, rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.set_axisbelow(True)
            if idx == 0:
                ax.legend(loc='upper right', fontsize=8, framealpha=0.95)

        # Hide any unused sub‑plots
        total_axes = nrows * ncols
        for empty_idx in range(n_models, total_axes):
            fig.delaxes(axes[empty_idx // ncols, empty_idx % ncols])

        plt.tight_layout()
        fig.savefig(self.output_dir / 'Figure_2_attack_vulnerability.png', dpi=120)
        logger.info(f"  ✓ Saved Figure 2\n")
        plt.close()

    def plot_variant_summary_statistics(self, df: pd.DataFrame):
        """Figure 3: Summary statistics for Control vs Pruned vs Pruned+Quant, split per model."""
        logger.info("Generating Figure 3: Variant Summary Statistics (per model)")
        df = self._normalize_summary_df(df)

        if df.empty:
            return

        colors = VARIANT_COLORS
        models = sorted(df['model'].unique())
        n_models = len(models)
        ncols = 3
        nrows = (n_models + ncols - 1) // ncols

        # Create a grid where each cell will contain a 2×2 block of sub‑plots for a model.
        # We'll generate a separate figure for each model to keep layout simple.
        for model in models:
            model_df = df[df['model'] == model]
            if model_df.empty:
                continue

            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(f'Figure 3: Summary for {model}\n(Control vs Pruned vs Pruned+Quant)',
                         fontsize=15, fontweight='bold')

            # Plot 1: Clean Accuracy
            if 'clean_accuracy' in model_df.columns:
                ax = axes[0, 0]
                clean_stats = model_df.groupby('variant')['clean_accuracy'].agg(['mean', 'std']).reset_index()
                variants = clean_stats['variant'].values
                means = clean_stats['mean'].values
                stds = clean_stats['std'].values
                bars = ax.bar(variants, means, alpha=0.75,
                               color=[colors.get(v, '#95a5a6') for v in variants],
                               edgecolor='black', linewidth=2)
                for bar, mean in zip(bars, means):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')
                ax.set_ylabel('Clean Accuracy (%)', fontweight='bold', fontsize=11)
                ax.set_title('Clean Accuracy', fontweight='bold')
                ax.set_ylim([0, 105])
                ax.grid(True, alpha=0.3, axis='y', linestyle='--')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Plot 2: Attack Success Rate
            if 'attack_success_rate' in model_df.columns:
                ax = axes[0, 1]
                attack_stats = model_df.groupby('variant')['attack_success_rate'].agg(['mean', 'std']).reset_index()
                variants = attack_stats['variant'].values
                means = attack_stats['mean'].values
                stds = attack_stats['std'].values
                bars = ax.bar(variants, means, alpha=0.75,
                               color=[colors.get(v, '#95a5a6') for v in variants],
                               edgecolor='black', linewidth=2)
                for bar, mean in zip(bars, means):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')
                ax.set_ylabel('Attack Success Rate (%)', fontweight='bold', fontsize=11)
                ax.set_title('Attack Success Rate', fontweight='bold')
                ax.set_ylim([0, 105])
                ax.grid(True, alpha=0.3, axis='y', linestyle='--')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Plot 3: Sample Count
            ax = axes[1, 0]
            counts = model_df.groupby('variant').size().reset_index(name='count')
            variants = counts['variant'].values
            counts_vals = counts['count'].values
            bars = ax.bar(variants, counts_vals, alpha=0.75,
                         color=[colors.get(v, '#95a5a6') for v in variants],
                         edgecolor='black', linewidth=2)
            for bar, count in zip(bars, counts_vals):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{count:,}', ha='center', va='bottom', fontweight='bold')
            ax.set_ylabel('Sample Count', fontweight='bold', fontsize=11)
            ax.set_title('Data Coverage', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Plot 4: Models Tested (will always be 1 for a single model, but kept for consistency)
            ax = axes[1, 1]
            model_counts = model_df.groupby('variant')['model'].nunique().reset_index()
            variants = model_counts['variant'].values
            model_vals = model_counts['model'].values
            bars = ax.bar(variants, model_vals, alpha=0.75,
                         color=[colors.get(v, '#95a5a6') for v in variants],
                         edgecolor='black', linewidth=2)
            for bar, cnt in zip(bars, model_vals):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{cnt}', ha='center', va='bottom', fontweight='bold')
            ax.set_ylabel('Number of Models', fontweight='bold', fontsize=11)
            ax.set_title('Model Coverage', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            plt.tight_layout()
            fig_path = self.output_dir / f'Figure_3_{model}_summary_statistics.png'
            fig.savefig(fig_path, dpi=120)
            logger.info(f"  ✓ Saved Figure 3 for {model}\n")
            plt.close()

    def plot_accuracy_robustness_correlation(self, df: pd.DataFrame):
        """Figure 4: Correlation between clean‑accuracy change and ASR change vs control, broken out per model."""
        logger.info("Generating Figure 4: Clean Accuracy vs Vulnerability Correlation (per model)")
        df = self._normalize_summary_df(df)

        if df.empty or 'clean_accuracy' not in df.columns or 'attack_success_rate' not in df.columns:
            logger.warning("  ⚠ Missing required columns")
            return

        control_variant = self._control_variant_name(df['variant'].dropna().unique())
        if control_variant is None:
            logger.warning("  ⚠ No variant information found for control comparison")
            return

        model_variant = df.groupby(['model', 'variant'], as_index=False).agg({
            'clean_accuracy': 'mean',
            'attack_success_rate': 'mean'
        })
        control_by_model = (
            model_variant[model_variant['variant'] == control_variant]
            [['model', 'clean_accuracy', 'attack_success_rate']]
            .rename(columns={
                'clean_accuracy': 'control_clean_accuracy',
                'attack_success_rate': 'control_attack_success_rate',
            })
        )
        plot_data = model_variant.merge(control_by_model, on='model', how='left')
        plot_data['clean_accuracy_delta_from_control'] = (
            plot_data['clean_accuracy'] - plot_data['control_clean_accuracy']
        )
        plot_data['asr_delta_from_control'] = (
            plot_data['attack_success_rate'] - plot_data['control_attack_success_rate']
        )
        plot_data = plot_data[plot_data['variant'] != control_variant].copy()
        plot_data.to_csv(self.output_dir / 'Figure_4_accuracy_vulnerability_data.csv', index=False)

        if plot_data.empty:
            logger.warning("  ⚠ Figure 4 has no non‑control variant data to plot")
            return

        # Create a grid of sub‑plots – one per model (max 6 → 2×3)
        models = sorted(df['model'].unique())
        n_models = len(models)
        ncols = 3
        nrows = (n_models + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
        colors = VARIANT_COLORS

        for idx, model in enumerate(models):
            ax = axes[idx // ncols, idx % ncols]
            model_data = plot_data[plot_data['model'] == model]
            if model_data.empty:
                ax.text(0.5, 0.5, f'{model}\n(No data)', ha='center', va='center')
                ax.set_title(model)
                continue
            for variant in sorted(model_data['variant'].unique()):
                variant_data = model_data[model_data['variant'] == variant]
                ax.scatter(variant_data['clean_accuracy_delta_from_control'],
                           variant_data['asr_delta_from_control'],
                           s=150, alpha=0.7, label=variant,
                           color=colors.get(variant, '#95a5a6'),
                           edgecolors='black', linewidth=1.5)
            ax.axhline(0, color='black', linewidth=1.0, alpha=0.8)
            ax.axvline(0, color='black', linewidth=1.0, alpha=0.8)
            ax.set_xlabel('Δ Clean Acc (pp)', fontsize=10)
            ax.set_ylabel('Δ ASR (pp)', fontsize=10)
            ax.set_title(model, fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            if idx == 0:
                ax.legend(loc='best', fontsize=8, framealpha=0.95)

        # Hide any unused axes
        total_axes = nrows * ncols
        for empty_idx in range(n_models, total_axes):
            fig.delaxes(axes[empty_idx // ncols, empty_idx % ncols])

        plt.tight_layout()
        fig.savefig(self.output_dir / 'Figure_4_accuracy_vulnerability_correlation.png', dpi=120)
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