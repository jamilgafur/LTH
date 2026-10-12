import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Canonical architectures expected in the benchmark suite
CANONICAL_MODELS = ['ConvNeXt', 'InceptionNet', 'MobileNet', 'RegNetX_400MF', 'VGG16', 'XceptionNet']

def load_and_compute_deltas():
    """Reads CSVs, skips empty files, and computes deltas."""
    csv_files = glob.glob("*.csv")
    if not csv_files:
        print("No CSV files found.")
        return None, []
    
    df_list = []
    for f in csv_files:
        if os.path.getsize(f) == 0:
            print(f"Skipping empty file (0 bytes): {f}")
            continue
        
        try:
            temp_df = pd.read_csv(f)
            if not temp_df.empty:
                df_list.append(temp_df)
        except pd.errors.EmptyDataError:
            print(f"Skipping unparsable/empty CSV data: {f}")
            
    if not df_list:
        print("No valid data found in any of the CSV files.")
        return None, []
        
    df = pd.concat(df_list, ignore_index=True)
    df = df[df['model'] != 'model'].reset_index(drop=True)

    # Convert numeric columns
    num_cols = [col for col in df.columns if col not in ['model', 'dataset', 'model_before']]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['pre_epochs'] = df['pre_epochs'].astype(int)
    df['post_epochs'] = df['post_epochs'].astype(int)
    
    # 1. Accuracy Deltas
    df['delta_acc_imp'] = df['acc_unstructured'] - df['acc_original_after']
    df['delta_acc_collapse'] = df['acc_collapsed'] - df['acc_original_after']
    
    # 2. Energy Savings
    df['energy_saved_imp'] = df['energy_joules_original'] - df['energy_joules_unstructured']
    df['energy_saved_collapse'] = df['energy_joules_original'] - df['energy_joules_collapsed']
    
    # 3. Adversarial Deltas
    df['delta_fgsm_imp'] = df['adv_acc_FGSM_unstructured'] - df['adv_acc_FGSM_original']
    df['delta_fgsm_collapse'] = df['adv_acc_FGSM_collapsed'] - df['adv_acc_FGSM_original']
    df['delta_pgd_imp'] = df['adv_acc_PGD_unstructured'] - df['adv_acc_PGD_original']
    df['delta_pgd_collapse'] = df['adv_acc_PGD_collapsed'] - df['adv_acc_PGD_original']
    
    # Ensure all canonical models plus any additional models found are in the master list
    all_models = sorted(list(set(df['model'].dropna().unique()).union(CANONICAL_MODELS)))
    return df, all_models

def highlight_missing_models(ax, all_models, missing_models):
    """Draws red indicators and colors x-tick labels red for models missing data."""
    for idx, model in enumerate(all_models):
        if model in missing_models:
            # Highlight missing slot with a light red span and bold text
            ax.axvspan(idx - 0.38, idx + 0.38, color='red', alpha=0.12, zorder=0)
            ax.text(idx, 0, "MISSING", color='red', fontsize=10, fontweight='bold',
                    ha='center', va='center', rotation=90, zorder=3)

    # Color the corresponding x-tick labels red
    plt.draw()
    for tick, model in zip(ax.get_xticklabels(), all_models):
        if model in missing_models:
            tick.set_color('red')
            tick.set_fontweight('bold')
        else:
            tick.set_color('black')

def plot_bar_chart(df, all_models, columns, labels, title, ylabel, filename, zero_line=False, df_annotations=None, anno_fmt="{val:.2f}"):
    """Helper function to plot standard grouped bar charts with fixed x-axis and missing indicators."""
    plt.figure(figsize=(13, 7))
    sns.set_theme(style="whitegrid")
    
    # Reindex against all_models to ensure exact same x-axis order everywhere
    df_reindexed = df.set_index('model').reindex(all_models)
    missing_models = set(df_reindexed[df_reindexed[columns].isna().all(axis=1)].index)
    
    df_plot = df_reindexed[columns]
    df_plot.columns = labels
    
    ax = df_plot.plot(kind='bar', figsize=(13, 6), width=0.7, colormap='Set2')
    
    if zero_line:
        plt.axhline(0, color='black', linewidth=1.2, linestyle='--')

    # Format values for existing bars
    if df_annotations is not None:
        df_anno = df_reindexed[df_annotations]
        anno_vals = df_anno.values.flatten('F')
        
        ymin, ymax = ax.get_ylim()
        yrange = (ymax - ymin) if (ymax - ymin) > 0 else 1.0
        ax.set_ylim(ymin - (yrange * 0.15) if ymin < 0 else 0, ymax + (yrange * 0.20))

        for i, p in enumerate(ax.patches):
            val = anno_vals[i]
            height = p.get_height()
            
            if pd.isna(val) or pd.isna(height) or height == 0:
                continue
            
            text_str = anno_fmt.format(val=val)
            xytext = (0, 5) if height >= 0 else (0, -15)
            ax.annotate(text_str, 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom' if height >= 0 else 'top',
                        xytext=xytext, textcoords='offset points', 
                        fontsize=9, fontweight='bold', color='#333333')

    highlight_missing_models(ax, all_models, missing_models)

    plt.title(title, fontsize=14, pad=15)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Modification Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved '{filename}'")
    plt.close()

def plot_adversarial_subplots(df, all_models, title, ylabel, filename):
    """Plots FGSM and PGD adversarial attacks side-by-side with fixed x-axes and missing indicators."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    sns.set_theme(style="whitegrid")

    df_reindexed = df.set_index('model').reindex(all_models)
    
    # Identify missing models
    adv_cols = ['delta_fgsm_imp', 'delta_fgsm_collapse', 'delta_pgd_imp', 'delta_pgd_collapse']
    missing_models = set(df_reindexed[df_reindexed[adv_cols].isna().all(axis=1)].index)

    # Subplot 0: FGSM
    df_fgsm = df_reindexed[['delta_fgsm_imp', 'delta_fgsm_collapse']]
    df_fgsm.columns = ['IMP (Unstructured)', 'Collapsed']
    df_fgsm_anno = df_reindexed[['adv_acc_FGSM_unstructured', 'adv_acc_FGSM_collapsed']]

    df_fgsm.plot(kind='bar', ax=axes[0], width=0.7, colormap='Set2')
    axes[0].set_title('FGSM Attack', fontsize=13, fontweight='bold')
    axes[0].axhline(0, color='black', linewidth=1.2, linestyle='--')
    axes[0].set_ylabel(ylabel, fontsize=12)
    axes[0].set_xlabel('Model', fontsize=12)

    # Subplot 1: PGD
    df_pgd = df_reindexed[['delta_pgd_imp', 'delta_pgd_collapse']]
    df_pgd.columns = ['IMP (Unstructured)', 'Collapsed']
    df_pgd_anno = df_reindexed[['adv_acc_PGD_unstructured', 'adv_acc_PGD_collapsed']]

    df_pgd.plot(kind='bar', ax=axes[1], width=0.7, colormap='Set2')
    axes[1].set_title('PGD Attack', fontsize=13, fontweight='bold')
    axes[1].axhline(0, color='black', linewidth=1.2, linestyle='--')
    axes[1].set_xlabel('Model', fontsize=12)

    # Scale y-limits for annotations
    ymin, ymax = axes[0].get_ylim()
    yrange = (ymax - ymin) if (ymax - ymin) > 0 else 1.0
    axes[0].set_ylim(ymin - (yrange * 0.15) if ymin < 0 else 0, ymax + (yrange * 0.20))

    def annotate_axis(ax, df_anno):
        anno_vals = df_anno.values.flatten('F')
        for i, p in enumerate(ax.patches):
            val = anno_vals[i]
            height = p.get_height()
            if pd.isna(val) or pd.isna(height) or height == 0:
                continue
            xytext = (0, 5) if height >= 0 else (0, -15)
            ax.annotate(f"{val:.1%}", 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom' if height >= 0 else 'top',
                        xytext=xytext, textcoords='offset points', 
                        fontsize=9, fontweight='bold', color='#333333')

    annotate_axis(axes[0], df_fgsm_anno)
    annotate_axis(axes[1], df_pgd_anno)

    for ax in axes:
        highlight_missing_models(ax, all_models, missing_models)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    axes[0].get_legend().remove()
    axes[1].legend(title='Modification Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved '{filename}'")
    plt.close()

def main():
    df, all_models = load_and_compute_deltas()
    if df is None:
        return

    grouped = df.groupby(['dataset', 'pre_epochs', 'post_epochs'])

    for (dataset, pre, post), group_df in grouped:
        print(f"\n--- Generating plots for {dataset} | Pre: {pre} | Post: {post} ---")
        
        file_suffix = f"{dataset}_pre{pre}_post{post}"
        title_suffix = f"\n({dataset} | Pre: {pre}, Post: {post})"

        # Plot 1: Change in Task Accuracy
        plot_bar_chart(
            df=group_df,
            all_models=all_models,
            columns=['delta_acc_imp', 'delta_acc_collapse'],
            labels=['IMP (Unstructured)', 'Collapsed'],
            title=f'Change in Task Accuracy (Compared to Original){title_suffix}',
            ylabel='Accuracy Delta\n(Higher is Better)',
            filename=f'plot_1_delta_accuracy_{file_suffix}.png',
            zero_line=True,
            df_annotations=['acc_unstructured', 'acc_collapsed'],
            anno_fmt="{val:.1%}"
        )

        # Plot 2: Energy Savings
        plot_bar_chart(
            df=group_df,
            all_models=all_models,
            columns=['energy_saved_imp', 'energy_saved_collapse'],
            labels=['IMP (Unstructured)', 'Collapsed'],
            title=f'Energy Savings (Joules Reduced from Baseline){title_suffix}',
            ylabel='Joules Saved\n(Higher is Better)',
            filename=f'plot_2_energy_savings_{file_suffix}.png',
            zero_line=True,
            df_annotations=['energy_joules_unstructured', 'energy_joules_collapsed'],
            anno_fmt="{val:.1f} J"
        )

        # Plot 3: CKA Scores
        plot_bar_chart(
            df=group_df,
            all_models=all_models,
            columns=['cka_unstructured', 'cka_collapsed'],
            labels=['IMP (Unstructured)', 'Collapsed'],
            title=f'Representational Similarity (CKA) to Original{title_suffix}',
            ylabel='CKA Score\n(Higher = More Similar to Original)',
            filename=f'plot_3_cka_scores_{file_suffix}.png',
            zero_line=False,
            df_annotations=['cka_unstructured', 'cka_collapsed'],
            anno_fmt="{val:.3f}"
        )

        # Plot 4: Adversarial Robustness Subplots
        plot_adversarial_subplots(
            df=group_df,
            all_models=all_models,
            title=f'Change in Adversarial Robustness{title_suffix}',
            ylabel='Accuracy Delta under Attack\n(Higher is Better)',
            filename=f'plot_4_delta_adversarial_{file_suffix}.png'
        )
        
        # Plot 5: Percentage Pruned (if present)
        if 'percent_pruned' in group_df.columns:
            plot_bar_chart(
                df=group_df,
                all_models=all_models,
                columns=['percent_pruned'],
                labels=['Target Sparsity (Same for IMP & Collapsed)'],
                title=f'Percentage of Weights Pruned from Original{title_suffix}',
                ylabel='Percentage Pruned\n(Higher = More Compressed)',
                filename=f'plot_5_percent_pruned_{file_suffix}.png',
                zero_line=False,
                df_annotations=['percent_pruned'],
                anno_fmt="{val:.1%}"
            )
    
    print("\nAll delta comparisons generated successfully!")

if __name__ == "__main__":
    main()