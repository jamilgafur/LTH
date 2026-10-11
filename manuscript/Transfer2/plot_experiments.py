import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_compute_deltas():
    """Reads CSVs, skipping empty files, and computes the change relative to the original model."""
    csv_files = glob.glob("*.csv")
    if not csv_files:
        print("No CSV files found.")
        return None
    
    df_list = []
    for f in csv_files:
        # Check if the file is completely empty at the OS level
        if os.path.getsize(f) == 0:
            print(f"Skipping empty file (0 bytes): {f}")
            continue
        
        # Catch pandas errors for files that have a header but no data, or malformed data
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

    df['experiment_name'] = df['model'] + "\n(" + df['dataset'] + ")"
    
    # 1. Accuracy Deltas (Compressed - Original)
    df['delta_acc_imp'] = df['acc_unstructured'] - df['acc_original_after']
    df['delta_acc_collapse'] = df['acc_collapsed'] - df['acc_original_after']
    
    # 2. Energy Savings (Original - Compressed)
    df['energy_saved_imp'] = df['energy_joules_original'] - df['energy_joules_unstructured']
    df['energy_saved_collapse'] = df['energy_joules_original'] - df['energy_joules_collapsed']
    
    # 3. Adversarial Deltas (Compressed - Original)
    df['delta_fgsm_imp'] = df['adv_acc_FGSM_unstructured'] - df['adv_acc_FGSM_original']
    df['delta_fgsm_collapse'] = df['adv_acc_FGSM_collapsed'] - df['adv_acc_FGSM_original']
    df['delta_pgd_imp'] = df['adv_acc_PGD_unstructured'] - df['adv_acc_PGD_original']
    df['delta_pgd_collapse'] = df['adv_acc_PGD_collapsed'] - df['adv_acc_PGD_original']
    
    return df

def plot_bar_chart(df, columns, labels, title, ylabel, filename, zero_line=False):
    """Helper function to plot grouped bar charts."""
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")
    
    df_plot = df.set_index('experiment_name')[columns]
    df_plot.columns = labels
    
    ax = df_plot.plot(kind='bar', figsize=(14, 6), width=0.7, colormap='Set2')
    
    if zero_line:
        plt.axhline(0, color='black', linewidth=1.2, linestyle='--')
        
    plt.title(title, fontsize=14, pad=15)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel('Model & Dataset', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Modification Type')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved '{filename}'")
    plt.close()

def main():
    df = load_and_compute_deltas()
    if df is None:
        return

    # Plot 1: Change in Task Accuracy
    plot_bar_chart(
        df=df,
        columns=['delta_acc_imp', 'delta_acc_collapse'],
        labels=['IMP (Unstructured)', 'Collapsed'],
        title='Change in Task Accuracy (Compared to Original Baseline)',
        ylabel='Accuracy Delta (Positive = Improvement)',
        filename='plot_1_delta_accuracy.png',
        zero_line=True
    )

    # Plot 2: Energy Savings
    plot_bar_chart(
        df=df,
        columns=['energy_saved_imp', 'energy_saved_collapse'],
        labels=['IMP (Unstructured)', 'Collapsed'],
        title='Energy Savings (Joules Reduced from Baseline)',
        ylabel='Joules Saved (Higher = More Efficient)',
        filename='plot_2_energy_savings.png',
        zero_line=True
    )

    # Plot 3: CKA Scores (Similarity to Original)
    plot_bar_chart(
        df=df,
        columns=['cka_unstructured', 'cka_collapsed'],
        labels=['IMP (Unstructured)', 'Collapsed'],
        title='Representational Similarity (CKA) to Original Baseline',
        ylabel='CKA Score (1.0 = Identical Features)',
        filename='plot_3_cka_scores.png',
        zero_line=False
    )

    # Plot 4: Change in Adversarial Robustness (FGSM & PGD)
    plot_bar_chart(
        df=df,
        columns=['delta_fgsm_imp', 'delta_fgsm_collapse', 'delta_pgd_imp', 'delta_pgd_collapse'],
        labels=['IMP - FGSM', 'Collapsed - FGSM', 'IMP - PGD', 'Collapsed - PGD'],
        title='Change in Adversarial Robustness (FGSM vs PGD)',
        ylabel='Accuracy Delta under Attack (Positive = More Robust)',
        filename='plot_4_delta_adversarial.png',
        zero_line=True
    )
    
    print("All delta comparisons generated successfully!")

if __name__ == "__main__":
    main()
