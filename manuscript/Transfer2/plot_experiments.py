import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def load_and_compute_deltas():
    """Reads CSVs, skips empty files, and computes deltas."""
    csv_files = glob.glob("*.csv")
    if not csv_files:
        print("No CSV files found.")
        return None
    
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
        return None
        
    df = pd.concat(df_list, ignore_index=True)
    df = df[df['model'] != 'model'].reset_index(drop=True)

    # Convert numeric columns
    num_cols = [col for col in df.columns if col not in ['model', 'dataset', 'model_before']]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure grouping columns are integers for clean filenames
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
    
    return df

def plot_bar_chart(df, x_col, columns, labels, title, ylabel, filename, zero_line=False, df_annotations=None, anno_fmt="{val:.2f}"):
    """Helper function to plot standard grouped bar charts with robust text annotations and NaN handling."""
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    df_plot = df.set_index(x_col)[columns]
    df_plot.columns = labels
    
    ax = df_plot.plot(kind='bar', figsize=(12, 6), width=0.7, colormap='Set2')
    
    if zero_line:
        plt.axhline(0, color='black', linewidth=1.2, linestyle='--')
        
    if df_annotations is not None:
        df_anno = df.set_index(x_col)[df_annotations]
        anno_vals = df_anno.values.flatten('F')
        
        # Expand Y-axis limits slightly so text annotations don't get cut off
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin
        # Give more space for missing data labels
        ax.set_ylim(ymin - (yrange * 0.15) if ymin < 0 else 0, ymax + (yrange * 0.20))

        for i, p in enumerate(ax.patches):
            val = anno_vals[i]
            height = p.get_height()
            
            # Check for missing data (NaN)
            if pd.isna(val) or pd.isna(height):
                ax.annotate("Missing", 
                            (p.get_x() + p.get_width() / 2., 0),
                            ha='center', va='bottom',
                            xytext=(0, 5), textcoords='offset points', 
                            fontsize=10, fontweight='bold', color='red', rotation=90)
                continue
            
            # Format and plot valid values
            text_str = anno_fmt.format(val=val)
            xytext = (0, 5) if height >= 0 else (0, -15)
            ax.annotate(text_str, 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom' if height >= 0 else 'top',
                        xytext=xytext, textcoords='offset points', 
                        fontsize=9, fontweight='bold', color='#333333')
                        
    plt.title(title, fontsize=14, pad=15)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    plt.legend(title='Modification Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved '{filename}'")
    plt.close()

def plot_adversarial_subplots(df, x_col, title, ylabel, filename):
    """Specific function to plot FGSM and PGD adversarial attacks side-by-side with NaN handling."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    sns.set_theme(style="whitegrid")

    # Prepare FGSM Data
    df_fgsm = df.set_index(x_col)[['delta_fgsm_imp', 'delta_fgsm_collapse']]
    df_fgsm.columns = ['IMP (Unstructured)', 'Collapsed']
    df_fgsm_anno = df.set_index(x_col)[['adv_acc_FGSM_unstructured', 'adv_acc_FGSM_collapsed']]

    # Prepare PGD Data
    df_pgd = df.set_index(x_col)[['delta_pgd_imp', 'delta_pgd_collapse']]
    df_pgd.columns = ['IMP (Unstructured)', 'Collapsed']
    df_pgd_anno = df.set_index(x_col)[['adv_acc_PGD_unstructured', 'adv_acc_PGD_collapsed']]

    # Plot FGSM on Subplot 0
    df_fgsm.plot(kind='bar', ax=axes[0], width=0.7, colormap='Set2')
    axes[0].set_title('FGSM Attack', fontsize=13, fontweight='bold')
    axes[0].axhline(0, color='black', linewidth=1.2, linestyle='--')
    axes[0].set_ylabel(ylabel, fontsize=12)
    axes[0].set_xlabel('Model', fontsize=12)

    # Plot PGD on Subplot 1
    df_pgd.plot(kind='bar', ax=axes[1], width=0.7, colormap='Set2')
    axes[1].set_title('PGD Attack', fontsize=13, fontweight='bold')
    axes[1].axhline(0, color='black', linewidth=1.2, linestyle='--')
    axes[1].set_xlabel('Model', fontsize=12)

    # Calculate global ylim to fit text annotations across both shared axes
    ymin, ymax = axes[0].get_ylim()
    yrange = ymax - ymin
    axes[0].set_ylim(ymin - (yrange * 0.15) if ymin < 0 else 0, ymax + (yrange * 0.20))

    # Helper function to annotate a specific axis
    def add_annotations(ax, df_anno, anno_fmt="{val:.1%}"):
        anno_vals = df_anno.values.flatten('F')
        for i, p in enumerate(ax.patches):
            val = anno_vals[i]
            height = p.get_height()
            
            # Handle Missing Data
            if pd.isna(val) or pd.isna(height):
                ax.annotate("Missing", 
                            (p.get_x() + p.get_width() / 2., 0),
                            ha='center', va='bottom',
                            xytext=(0, 5), textcoords='offset points', 
                            fontsize=10, fontweight='bold', color='red', rotation=90)
                continue
            
            text_str = anno_fmt.format(val=val)
            xytext = (0, 5) if height >= 0 else (0, -15)
            ax.annotate(text_str, 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom' if height >= 0 else 'top',
                        xytext=xytext, textcoords='offset points', 
                        fontsize=9, fontweight='bold', color='#333333')

    add_annotations(axes[0], df_fgsm_anno)
    add_annotations(axes[1], df_pgd_anno)

    # Clean up formatting
    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Set one shared legend on the right side
    axes[0].get_legend().remove()
    axes[1].legend(title='Modification Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved '{filename}'")
    plt.close()

def main():
    df = load_and_compute_deltas()
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
            x_col='model',
            columns=['delta_acc_imp', 'delta_acc_collapse'],
            labels=['IMP (Unstructured)', 'Collapsed'],
            title=f'Change in Task Accuracy (Compared to Original){title_suffix}',
            ylabel='Accuracy Delta\n(Higher is Better)',
            filename=f'plot_1_delta_accuracy_{file_suffix}.png',
            zero_line=True,
            df_annotations=['acc_unstructured', 'acc_collapsed'],
            anno_fmt="{val:.1%}" # Formats as percentage (e.g. 74.5%)
        )

        # Plot 2: Energy Savings (Annotations show Actual Energy Consumption)
        plot_bar_chart(
            df=group_df,
            x_col='model',
            columns=['energy_saved_imp', 'energy_saved_collapse'],
            labels=['IMP (Unstructured)', 'Collapsed'],
            title=f'Energy Savings (Joules Reduced from Baseline){title_suffix}',
            ylabel='Joules Saved\n(Higher is Better)',
            filename=f'plot_2_energy_savings_{file_suffix}.png',
            zero_line=True,
            df_annotations=['energy_joules_unstructured', 'energy_joules_collapsed'],
            anno_fmt="{val:.1f} J" # Formats as Joules float (e.g. 450.5 J)
        )

        # Plot 3: CKA Scores
        plot_bar_chart(
            df=group_df,
            x_col='model',
            columns=['cka_unstructured', 'cka_collapsed'],
            labels=['IMP (Unstructured)', 'Collapsed'],
            title=f'Representational Similarity (CKA) to Original{title_suffix}',
            ylabel='CKA Score\n(Higher = More Similar to Original)',
            filename=f'plot_3_cka_scores_{file_suffix}.png',
            zero_line=False,
            df_annotations=['cka_unstructured', 'cka_collapsed'],
            anno_fmt="{val:.3f}" # Formats to 3 decimal places (e.g. 0.852)
        )

        # Plot 4: Change in Adversarial Robustness (Subplots)
        plot_adversarial_subplots(
            df=group_df,
            x_col='model',
            title=f'Change in Adversarial Robustness{title_suffix}',
            ylabel='Accuracy Delta under Attack\n(Higher is Better)',
            filename=f'plot_4_delta_adversarial_{file_suffix}.png'
        )
        
        # Plot 5: Percentage Pruned (Only generates if 'percent_pruned' exists)
        if 'percent_pruned' in group_df.columns:
            plot_bar_chart(
                df=group_df,
                x_col='model',
                columns=['percent_pruned'],
                labels=['Target Sparsity (Same for IMP & Collapsed)'],
                title=f'Percentage of Weights Pruned from Original{title_suffix}',
                ylabel='Percentage Pruned\n(Higher = More Compressed)',
                filename=f'plot_5_percent_pruned_{file_suffix}.png',
                zero_line=False,
                df_annotations=['percent_pruned'],
                anno_fmt="{val:.1%}" # Formats as percentage
            )
    
    print("\nAll delta comparisons generated successfully!")

if __name__ == "__main__":
    main()