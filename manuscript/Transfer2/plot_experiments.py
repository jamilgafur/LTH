import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

    # Convert numeric columns (will automatically catch 'percent_pruned' if it exists)
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

def plot_bar_chart(df, x_col, columns, labels, title, ylabel, filename, zero_line=False, df_annotations=None, is_percent=False):
    """Helper function to plot grouped bar charts with optional text annotations."""
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    df_plot = df.set_index(x_col)[columns]
    df_plot.columns = labels
    
    ax = df_plot.plot(kind='bar', figsize=(12, 6), width=0.7, colormap='Set2')
    
    if zero_line:
        plt.axhline(0, color='black', linewidth=1.2, linestyle='--')
        
    if df_annotations is not None:
        # Extract the actual absolute values to overlay on the delta bars
        df_anno = df.set_index(x_col)[df_annotations]
        anno_vals = df_anno.values.flatten('F') # Flatten column-major to map perfectly to matplotlib patches
        
        # Expand Y-axis limits slightly so text annotations don't get cut off
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin
        ax.set_ylim(ymin - (yrange * 0.15) if ymin < 0 else ymin, ymax + (yrange * 0.15))

        for i, p in enumerate(ax.patches):
            val = anno_vals[i]
            if pd.isna(val):
                continue
            
            height = p.get_height()
            text_str = f"{val:.1%}" if is_percent else f"{val:.2f}"
            
            # Position text above positive bars, below negative bars
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
    
    # Move legend outside the plot so it doesn't cover bars/text
    plt.legend(title='Modification Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved '{filename}'")
    plt.close()

def main():
    df = load_and_compute_deltas()
    if df is None:
        return

    # Group by dataset, pre_epochs, and post_epochs
    grouped = df.groupby(['dataset', 'pre_epochs', 'post_epochs'])

    for (dataset, pre, post), group_df in grouped:
        print(f"\n--- Generating plots for {dataset} | Pre: {pre} | Post: {post} ---")
        
        file_suffix = f"{dataset}_pre{pre}_post{post}"
        title_suffix = f"\n({dataset} | Pre: {pre}, Post: {post})"

        # Plot 1: Change in Task Accuracy (Annotated with Actual % Accuracies)
        plot_bar_chart(
            df=group_df,
            x_col='model',
            columns=['delta_acc_imp', 'delta_acc_collapse'],
            labels=['IMP (Unstructured)', 'Collapsed'],
            title=f'Change in Task Accuracy (Compared to Original){title_suffix}',
            ylabel='Accuracy Delta',
            filename=f'plot_1_delta_accuracy_{file_suffix}.png',
            zero_line=True,
            df_annotations=['acc_unstructured', 'acc_collapsed'],
            is_percent=True
        )

        # Plot 2: Energy Savings (Joules are not percents, so no % annotation)
        plot_bar_chart(
            df=group_df,
            x_col='model',
            columns=['energy_saved_imp', 'energy_saved_collapse'],
            labels=['IMP (Unstructured)', 'Collapsed'],
            title=f'Energy Savings (Joules Reduced from Baseline){title_suffix}',
            ylabel='Joules Saved (Higher = More Efficient)',
            filename=f'plot_2_energy_savings_{file_suffix}.png',
            zero_line=True
        )

        # Plot 3: CKA Scores (Similarity score 0.0 - 1.0, technically not a percent)
        plot_bar_chart(
            df=group_df,
            x_col='model',
            columns=['cka_unstructured', 'cka_collapsed'],
            labels=['IMP (Unstructured)', 'Collapsed'],
            title=f'Representational Similarity (CKA) to Original{title_suffix}',
            ylabel='CKA Score (1.0 = Identical Features)',
            filename=f'plot_3_cka_scores_{file_suffix}.png',
            zero_line=False
        )

        # Plot 4: Change in Adversarial Robustness (Annotated with Actual % Accuracies)
        plot_bar_chart(
            df=group_df,
            x_col='model',
            columns=['delta_fgsm_imp', 'delta_fgsm_collapse', 'delta_pgd_imp', 'delta_pgd_collapse'],
            labels=['IMP - FGSM', 'Collapsed - FGSM', 'IMP - PGD', 'Collapsed - PGD'],
            title=f'Change in Adversarial Robustness{title_suffix}',
            ylabel='Accuracy Delta under Attack',
            filename=f'plot_4_delta_adversarial_{file_suffix}.png',
            zero_line=True,
            df_annotations=['adv_acc_FGSM_unstructured', 'adv_acc_FGSM_collapsed', 'adv_acc_PGD_unstructured', 'adv_acc_PGD_collapsed'],
            is_percent=True
        )
        
        # Plot 5: Percentage Pruned (Only generates if 'percent_pruned' exists in CSVs)
        if 'percent_pruned' in group_df.columns:
            plot_bar_chart(
                df=group_df,
                x_col='model',
                columns=['percent_pruned'],
                labels=['Target Sparsity (Same for IMP & Collapsed)'],
                title=f'Percentage of Weights Pruned from Original{title_suffix}',
                ylabel='Percentage Pruned',
                filename=f'plot_5_percent_pruned_{file_suffix}.png',
                zero_line=False,
                df_annotations=['percent_pruned'],
                is_percent=True
            )
    
    print("\nAll delta comparisons generated successfully!")

if __name__ == "__main__":
    main()
