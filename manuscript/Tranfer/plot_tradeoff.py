import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re
import os

# ==============================================================================
# DATA PARSING
# ==============================================================================

def parse_txt_files(file_paths):
    data_frames = []
    for file_path in file_paths:
        if not os.path.exists(file_path): continue
        with open(file_path, 'r') as f:
            content = f.read()
        sections = content.split('==> ')
        for section in sections:
            if not section.strip(): continue
            lines = section.strip().split('\n')
            header = lines[0]
            csv_data = '\n'.join(lines[1:])
            
            # Extract pre and post epochs
            match = re.search(r'figures_ep(\d+)_post(\d+)', header)
            
            if match:
                pre_train_ep, post_collapse_ep = int(match.group(1)), int(match.group(2))
            else: continue
            
            try:
                df = pd.read_csv(io.StringIO(csv_data))
                df['Pre_Train_Epochs'], df['Post_Collapse_Epochs'] = pre_train_ep, post_collapse_ep
                df['Type'] = df['Experiment'].apply(lambda x: 'Combined Region' if 'Combined' in str(x) else ('Individual Set' if 'Set' in str(x) else 'Control'))
                data_frames.append(df)
            except Exception: continue
            
    df_final = pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()
    
    # CORRECTED: Promote single sets to "Combined Region" if they are the only set available
    if not df_final.empty:
        groups = df_final.groupby(['Dataset', 'Architecture', 'Pre_Train_Epochs', 'Post_Collapse_Epochs', 'Is_Quantized'])
        new_rows = []
        
        for name, group in groups:
            if not any(group['Type'] == 'Combined Region'):
                individual_sets = group[group['Type'] == 'Individual Set']
                
                # If there's exactly one individual set, it IS the combined region
                if len(individual_sets) == 1:
                    combined_proxy = individual_sets.copy()
                    combined_proxy['Type'] = 'Combined Region'
                    combined_proxy['Experiment'] = 'Dynamic_Region_All_Combined'
                    new_rows.append(combined_proxy)
                    
        if new_rows:
            df_final = pd.concat([df_final] + new_rows, ignore_index=True)
            
    return df_final

# ==============================================================================
# VISUALIZATION LOGIC
# ==============================================================================

def generate_journal_figures(df):
    """Generates a strict, publication-ready 2-panel figure with dynamic epoch handling."""
    df_filtered = df[(df['Is_Quantized'] == False) & (df['Type'] != 'Control')].dropna(subset=['Delta_Acc', 'Params_Reduction_%']).copy()

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "figure.dpi": 300})
    sns.set_theme(style="ticks", context="paper")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

    # PANEL A: Plots the full distribution of available data.
    sns.barplot(data=df_filtered, x='Architecture', y='Params_Reduction_%', hue='Type', 
                palette={'Individual Set': '#E0E0E0', 'Combined Region': '#1F449C'}, 
                edgecolor='black', capsize=0.1, ax=ax1)
    
    sns.stripplot(data=df_filtered[df_filtered['Type'] == 'Individual Set'], 
                  x='Architecture', y='Params_Reduction_%', color='black', 
                  alpha=0.6, jitter=True, ax=ax1)
    
    ax1.set_title("(a) Structural Compression Yield (All Epochs)", weight='bold')
    ax1.set_ylabel("Parameter Reduction (%)", weight='bold')

    # PANEL B: Heatmap with explicit Color Bar Label
    df_heat = df_filtered[(df_filtered['Type'] == 'Combined Region')]
    heat_data = df_heat.pivot_table(index='Architecture', columns='Post_Collapse_Epochs', 
                                    values='Delta_Acc', aggfunc='mean')
    
    sns.heatmap(heat_data, annot=True, fmt=".2f", cmap="RdBu", center=0, ax=ax2,
                cbar_kws={'label': '$\\Delta$ Accuracy (%)'})
    
    ax2.set_title("(b) Post-Collapse Predictive Recovery", weight='bold')
    ax2.set_xlabel("Finetuning Budget (Epochs)", weight='bold')

    plt.tight_layout()
    plt.savefig("journal_tradeoff_analysis.png")
    print("✅ Saved: journal_tradeoff_analysis.png with updated labeling and data filtering.")

# ==============================================================================
# LATEX GENERATION LOGIC
# ==============================================================================

def get_metrics(df, dataset, arch, pre, fine, quantized=False):
    subset = df[(df['Dataset'] == dataset) & (df['Architecture'] == arch) & (df['Pre_Train_Epochs'] == pre) & 
                (df['Post_Collapse_Epochs'] == fine) & (df['Experiment'] == 'Dynamic_Region_All_Combined') & 
                (df['Is_Quantized'] == quantized)].dropna(subset=['Delta_Acc', 'Params_Reduction_%'])
    return subset.iloc[0] if not subset.empty else None

def generate_tables(df):
    """Generates and saves the LaTeX files for both tables."""
    
    # 1. Hardware Efficiency Table
    datasets = ['CIFAR-10', 'CIFAR-100', 'TinyImageNet']
    architectures = ['VGG16', 'RegNetX_400MF', 'ConvNeXt', 'InceptionNet', 'XceptionNet']
    epochs = [(100, 300), (200, 200), (300, 100)]
    
    latex_hw = ["\\begin{table*}[htbp]\n\\centering", "\\begin{tabular}{@{}llccccc@{}}", "\\toprule", 
                "\\textbf{Dataset} & \\textbf{Architecture} & \\textbf{Pre/Fine} & \\textbf{Params Red (\\%)} & \\textbf{FLOPs Red (\\%)} & \\textbf{Mem Red (\\%)} & \\textbf{$\\Delta$ Acc (\\%)} \\\\", "\\midrule"]
    
    for dataset in datasets:
        for a_idx, arch in enumerate(architectures):
            for e_idx, (pre, fine) in enumerate(epochs):
                m = get_metrics(df, dataset, arch, pre, fine, quantized=False)
                row = f"{dataset if e_idx == 0 and a_idx == 0 else ''} & {arch.replace('_', '\\_') if e_idx == 0 else ''} & {pre}/{fine} & "
                row += f"{m['Params_Reduction_%']:.2f} & {m['FLOPs_Reduction_%']:.2f} & {m['Memory_Reduction_%']:.2f} & {m['Delta_Acc']:.2f} \\\\" if m is not None else "N/A & N/A & N/A & N/A \\\\"
                latex_hw.append(row)
    latex_hw.append("\\bottomrule\n\\end{tabular}\n\\end{table*}")
    with open("table_hardware_efficiency.tex", "w") as f: f.write("\n".join(latex_hw))
    
    # 2. Quantization Table
    target_runs = [("CIFAR-10", "VGG16", 100, 300), ("CIFAR-10", "ConvNeXt", 200, 200),
                   ("CIFAR-100", "RegNetX_400MF", 100, 300), ("CIFAR-100", "XceptionNet", 200, 200),
                   ("TinyImageNet", "InceptionNet", 200, 200)]
    
    latex_quant = ["\\begin{table}[htbp]\n\\centering", "\\begin{tabular}{@{}llccc@{}}", "\\toprule", 
                   "Dataset & Arch & Pre/Fine & FP32 $\\Delta$ & INT8 $\\Delta$ \\\\", "\\midrule"]
    
    for ds, arch, pre, fine in target_runs:
        m_fp32 = get_metrics(df, ds, arch, pre, fine, quantized=False)
        m_int8 = get_metrics(df, ds, arch, pre, fine, quantized=True)
        val_fp32 = f"{m_fp32['Delta_Acc']:.2f}" if m_fp32 is not None else "N/A"
        val_int8 = f"{m_int8['Delta_Acc']:.2f}" if m_int8 is not None else "N/A"
        latex_quant.append(f"{ds} & {arch.replace('_', '\\_')} & {pre}/{fine} & {val_fp32} & {val_int8} \\\\")
        
    latex_quant.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
    with open("table_quantization.tex", "w") as f: f.write("\n".join(latex_quant))
    
    print("✅ LaTeX Tables Generated (table_hardware_efficiency.tex, table_quantization.tex)")

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    df = parse_txt_files(["info.txt"])
    if not df.empty:
        generate_journal_figures(df)
        generate_tables(df)
    else:
        print("❌ Error: No valid data found in info.txt.")