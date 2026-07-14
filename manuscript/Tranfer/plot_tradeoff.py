import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import io
import re
import os
import json
import glob

# ==============================================================================
# DATA PARSING (TEXT & JSON)
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
            
            csv_start_idx = -1
            for i, line in enumerate(lines):
                if line.startswith('Architecture,Dataset') or line.startswith('Dataset,Architecture'):
                    csv_start_idx = i
                    break
            
            if csv_start_idx == -1: continue
            
            csv_data = '\n'.join(lines[csv_start_idx:])
            
            try:
                df = pd.read_csv(io.StringIO(csv_data))
                
                if 'Pre_Train_Epochs' not in df.columns:
                    match = re.search(r'ep(\d+)_(?:pre|post)(\d+)|epochs(\d+)_pretrain(\d+)', header)
                    if match:
                        if match.group(1): 
                            df['Pre_Train_Epochs'] = int(match.group(1))
                            df['Post_Collapse_Epochs'] = int(match.group(2))
                        else: 
                            df['Pre_Train_Epochs'] = int(match.group(3))
                            df['Post_Collapse_Epochs'] = int(match.group(4))
                    else: continue
                
                df['Dataset'] = df['Dataset'].replace({'Tiny ImageNet': 'TinyImageNet', 'Tiny-ImageNet': 'TinyImageNet'})
                if 'Is_Quantized' in df.columns:
                    df['Is_Quantized'] = df['Is_Quantized'].astype(str).str.strip().str.lower().map({'true': True, 'false': False})
                    
                df['Type'] = df['Experiment'].apply(lambda x: 'Combined Region' if 'Combined' in str(x) else ('Individual Set' if 'Set' in str(x) else 'Control'))
                data_frames.append(df)
            except Exception: continue
            
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

def build_global_controls(base_path="."):
    """Scans all JSONs to build a global dictionary of maximum Control metrics."""
    global_controls = {}
    search_pattern = os.path.join(base_path, "*", "metrics", "merged_metrics.json")
    
    for filepath in glob.glob(search_pattern):
        folder_name = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
        
        if "_Cifar100_" in folder_name: dataset = "CIFAR-100"
        elif "_Cifar10_" in folder_name: dataset = "CIFAR-10"
        elif "_tinyimagenet_" in folder_name: dataset = "TinyImageNet"
        else: continue
        
        arch = "RegNetX_400MF" if folder_name.startswith("RegNetX_400MF") else folder_name.split("_")[0]
        
        try:
            with open(filepath, 'r') as f: data = json.load(f)
        except Exception: continue
        
        ctrl_key = next((k for k in data.keys() if "Control" in k and "quant" not in k and "Continuted" not in k), None)
        if ctrl_key:
            acc = data[ctrl_key].get("best_acc", data[ctrl_key].get("final_accuracy", 0))
            params = data[ctrl_key].get("param_count", 0)
            flops = data[ctrl_key].get("flops", 0)
            mem = data[ctrl_key].get("total_size_mb", 0)
            
            if arch not in global_controls: global_controls[arch] = {}
            if dataset not in global_controls[arch]:
                global_controls[arch][dataset] = {'acc': acc, 'params': params, 'flops': flops, 'mem': mem}
            else:
                global_controls[arch][dataset]['acc'] = max(global_controls[arch][dataset]['acc'], acc)
                global_controls[arch][dataset]['params'] = max(global_controls[arch][dataset]['params'], params)
                global_controls[arch][dataset]['flops'] = max(global_controls[arch][dataset]['flops'], flops)
                global_controls[arch][dataset]['mem'] = max(global_controls[arch][dataset]['mem'], mem)
                
    return global_controls

def augment_df_with_json(df, base_path="."):
    """Reads all merged_metrics.json, extracts missing runs, applies Set Promotion, and calculates Deltas vs Control."""
    global_controls = build_global_controls(base_path)
    new_rows = []
    
    for filepath in glob.glob(os.path.join(base_path, "*", "metrics", "merged_metrics.json")):
        folder_name = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
        if "_Cifar100_" in folder_name: dataset = "CIFAR-100"
        elif "_Cifar10_" in folder_name: dataset = "CIFAR-10"
        elif "_tinyimagenet_" in folder_name: dataset = "TinyImageNet"
        else: continue
        
        arch = "RegNetX_400MF" if folder_name.startswith("RegNetX_400MF") else folder_name.split("_")[0]
        
        ep_match = re.search(r'epochs(\d+)_pretrain(\d+)', folder_name)
        if not ep_match: continue
        
        pre_train_ep, post_collapse_ep = int(ep_match.group(1)), int(ep_match.group(2))
        
        try:
            with open(filepath, 'r') as f: data = json.load(f)
        except: continue
        
        control_metrics = global_controls.get(arch, {}).get(dataset)
        if not control_metrics: continue
        
        for k, v in data.items():
            if "Control" in k: continue 
            
            is_q = "quant" in k
            row_type = "Combined Region" if "Combined" in k else "Individual Set"
            exp_name = k.replace("_quant_JF", "").replace("_JF", "")
            
            acc = v.get("best_acc", v.get("final_accuracy", 0))
            params = v.get("param_count", control_metrics['params'])
            flops = v.get("flops", control_metrics['flops'])
            mem = v.get("total_size_mb", control_metrics['mem'])
            
            new_rows.append({
                'Dataset': dataset, 'Architecture': arch, 'Pre_Train_Epochs': pre_train_ep,
                'Post_Collapse_Epochs': post_collapse_ep, 'Experiment': exp_name,
                'Type': row_type, 'Is_Quantized': is_q, 'Delta_Acc': acc - control_metrics['acc'],
                'Params_Reduction_%': (1 - params / control_metrics['params']) * 100,
                'FLOPs_Reduction_%': (1 - flops / control_metrics['flops']) * 100,
                'Memory_Reduction_%': (1 - mem / control_metrics['mem']) * 100
            })
            
    df_json = pd.DataFrame(new_rows)
    df_combined = pd.concat([df, df_json], ignore_index=True) if not df_json.empty else df
    
    if not df_combined.empty:
        df_combined = df_combined.drop_duplicates(
            subset=['Dataset', 'Architecture', 'Pre_Train_Epochs', 'Post_Collapse_Epochs', 'Type', 'Experiment', 'Is_Quantized'],
            keep='last' 
        )
    
        groups = df_combined.groupby(['Dataset', 'Architecture', 'Pre_Train_Epochs', 'Post_Collapse_Epochs', 'Is_Quantized'])
        promoted_rows = []
        for name, group in groups:
            if not any(group['Type'] == 'Combined Region'):
                ind_sets = group[group['Type'] == 'Individual Set']
                if len(ind_sets) == 1:
                    row = ind_sets.iloc[0].copy()
                    row['Type'] = 'Combined Region'
                    row['Experiment'] = 'Dynamic_Region_All_Combined'
                    promoted_rows.append(row.to_frame().T)
                    
        if promoted_rows:
            df_combined = pd.concat([df_combined] + promoted_rows, ignore_index=True)
        
    return df_combined

# ==============================================================================
# VISUALIZATION LOGIC
# ==============================================================================

def generate_journal_figures(df):
    """Generates a highly polished, publication-ready composite figure with shared Y-axes."""
    if df.empty: return
    
    df['Delta_Acc'] = pd.to_numeric(df['Delta_Acc'], errors='coerce')
    df['Params_Reduction_%'] = pd.to_numeric(df['Params_Reduction_%'], errors='coerce')
    
    df_filtered = df[(df['Is_Quantized'] == False) & (df['Type'] != 'Control')].dropna(subset=['Delta_Acc', 'Params_Reduction_%']).copy()

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "figure.dpi": 300})
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    
    fig = plt.figure(figsize=(18, 6))
    gs_main = gridspec.GridSpec(1, 2, width_ratios=[1, 1.8], wspace=0.25)

    # Panel (a)
    ax1 = fig.add_subplot(gs_main[0])
    sns.barplot(data=df_filtered, x='Architecture', y='Params_Reduction_%', hue='Type', 
                palette={'Individual Set': '#E0E0E0', 'Combined Region': '#1F449C'}, 
                edgecolor='black', linewidth=1.2, capsize=0.1, err_kws={'linewidth': 1.5}, ax=ax1)
    
    sns.stripplot(data=df_filtered[df_filtered['Type'] == 'Individual Set'], 
                  x='Architecture', y='Params_Reduction_%', color='#333333', 
                  alpha=0.7, size=5, jitter=0.2, ax=ax1)
    
    ax1.set_title("(a) Structural Compression Yield", weight='bold', pad=15)
    ax1.set_ylabel("Parameter Reduction (%) vs Control", weight='bold')
    ax1.set_xlabel("")
    ax1.tick_params(axis='x', rotation=30)
    
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles[:2], labels=labels[:2], loc='upper left', framealpha=0.9)
    sns.despine(ax=ax1)

    # Panel (b)
    gs_right = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[1], wspace=0.1)
    
    df_lines = df_filtered[df_filtered['Type'] == 'Combined Region'].sort_values('Post_Collapse_Epochs')
    datasets = ['CIFAR-10', 'CIFAR-100', 'TinyImageNet']
    
    archs = df_lines['Architecture'].unique()
    line_colors = sns.color_palette("tab10", n_colors=len(archs))
    
    axes = []
    for i, ds in enumerate(datasets):
        ax = fig.add_subplot(gs_right[i], sharey=axes[0] if i > 0 else None)
        axes.append(ax)

        ds_data = df_lines[df_lines['Dataset'] == ds]

        sns.lineplot(data=ds_data, x='Post_Collapse_Epochs', y='Delta_Acc', 
                     hue='Architecture', palette=line_colors, marker='o', 
                     markersize=8, markeredgecolor='white', markeredgewidth=1.5,
                     linewidth=2.5, ax=ax, legend=False)

        ax.set_title(f"{ds}", weight='bold')
        ax.set_xlabel("Finetuning Budget", weight='bold')
        ax.set_xticks([100, 200, 300])
        
        ax.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
        ax.yaxis.grid(True, linestyle='-', alpha=0.2)
        ax.xaxis.grid(False)

        if i == 0:
            ax.set_ylabel("$\\Delta$ Accuracy (%) vs Control", weight='bold')
        else:
            ax.set_ylabel("")
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='y', left=False) 

        sns.despine(ax=ax)

    fig.text(0.66, 0.95, "(b) Post-Collapse Predictive Recovery Trajectories", 
             ha='center', weight='bold', fontsize=13)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.66, -0.05), 
               ncol=3, frameon=False, title="Architecture", title_fontproperties={'weight':'bold'})

    plt.savefig("journal_tradeoff_analysis.png", bbox_inches='tight')
    print("✅ Saved: journal_tradeoff_analysis.png")

# ==============================================================================
# LATEX GENERATION LOGIC
# ==============================================================================

def get_metrics(df, dataset, arch, pre, fine, quantized=False):
    subset = df[(df['Dataset'].str.lower() == dataset.lower()) & 
                (df['Architecture'].str.lower() == arch.lower()) & 
                (df['Pre_Train_Epochs'] == pre) & 
                (df['Post_Collapse_Epochs'] == fine) & 
                (df['Type'] == 'Combined Region') & 
                (df['Is_Quantized'] == quantized)].dropna(subset=['Delta_Acc'])
    return subset.iloc[0] if not subset.empty else None

def get_best_delta_acc(df, dataset, arch, quantized=False):
    subset = df[(df['Dataset'].str.lower() == dataset.lower()) & 
                (df['Architecture'].str.lower() == arch.lower()) & 
                (df['Type'] == 'Combined Region') & 
                (df['Is_Quantized'] == quantized)].dropna(subset=['Delta_Acc'])
    if subset.empty: return -999.0
    return subset['Delta_Acc'].max()

def generate_tables(df):
    """Generates and saves the Condensed/Transposed LaTeX files with typographic enhancements."""
    if df.empty: return
    
    architectures = ['ConvNeXt', 'InceptionNet', 'MobileNet', 'RegNetX_400MF', 'VGG16', 'XceptionNet']
    epochs_list = [(100, 300), (200, 200), (300, 100)] 
    datasets = ['CIFAR-10', 'CIFAR-100', 'TinyImageNet']
    
    # 1. Transposed Hardware Efficiency Table
    latex_hw = [
        "\\begin{table*}[htbp]", "\\centering", 
        "\\caption{\\textbf{Hardware Efficiency and Predictive Accuracy Trade-offs.} This table presents a condensed, transposed view of the performance metrics for the holistic structural collapse (\\textit{Dynamic\\_Region\\_All\\_Combined}). Results are tracked across varying allocations of pretraining and post-collapse finetuning epochs (Pre/Fine). Reductions in parameters (PR), FLOPs (FR), and memory (MR) are reported as percentages relative to the fully trained, uncompressed Control models. The accuracy change ($\\Delta$A) represents the absolute percentage point difference in top-1 accuracy between the recovered collapsed model and the Control. Bold values indicate the optimal predictive recovery configuration per architecture. Missing values (-) denote incomplete configurations.}",
        "\\label{tab:consolidated_hardware_efficiency}", "\\resizebox{\\textwidth}{!}{", "\\begin{tabular}{@{}ll | cccc | cccc | cccc@{}}",
        "\\toprule", "\\multirow{2}{*}{\\textbf{Architecture}} & \\multirow{2}{*}{\\textbf{Pre/Fine}} & \\multicolumn{4}{c|}{\\textbf{CIFAR-10}} & \\multicolumn{4}{c|}{\\textbf{CIFAR-100}} & \\multicolumn{4}{c}{\\textbf{Tiny ImageNet}} \\\\",
        "\\cmidrule(l){3-6} \\cmidrule(l){7-10} \\cmidrule(l){11-14}", " & & \\textbf{PR(\\%)} & \\textbf{FR(\\%)} & \\textbf{MR(\\%)} & \\textbf{$\\Delta$A(\\%)} & \\textbf{PR(\\%)} & \\textbf{FR(\\%)} & \\textbf{MR(\\%)} & \\textbf{$\\Delta$A(\\%)} & \\textbf{PR(\\%)} & \\textbf{FR(\\%)} & \\textbf{MR(\\%)} & \\textbf{$\\Delta$A(\\%)} \\\\", "\\midrule"
    ]

    for arch in architectures:
        best_das = {ds: get_best_delta_acc(df, ds, arch, False) for ds in datasets}
        
        for i, (pre, fine) in enumerate(epochs_list):
            row = [f"\\multirow{{3}}{{*}}{{\\textbf{{{arch.replace('_', '\\_')}}}}}" if i == 0 else "", f"{pre}/{fine}"]
            for ds in datasets:
                m = get_metrics(df, ds, arch, pre=pre, fine=fine, quantized=False)
                if m is not None:
                    pr = f"{m['Params_Reduction_%']:.2f}"
                    fr = f"{m['FLOPs_Reduction_%']:.2f}"
                    mr = f"{m['Memory_Reduction_%']:.2f}"
                    
                    da_val = m['Delta_Acc']
                    da_str = f"{da_val:+.2f}"
                    if da_val == best_das[ds] and da_val != -999.0:
                        da_str = f"\\textbf{{{da_str}}}"
                        
                    row.append(f"{pr} & {fr} & {mr} & {da_str}")
                else:
                    row.append("- & - & - & -")
            latex_hw.append(" & ".join(row) + " \\\\")
        latex_hw.append("\\midrule")
    latex_hw[-1] = "\\bottomrule"
    latex_hw.extend(["\\end{tabular}", "}", "\\end{table*}"])
    with open("table_hardware_efficiency.tex", "w") as f: f.write("\n".join(latex_hw))
    
    # 2. Transposed Quantization Table (ISOLATING INT8 PENALTY)
    latex_quant = [
        "\\begin{table*}[htbp]", "\\centering",
        "\\caption{\\textbf{Robustness to Post-Training Quantization (INT8).} Comparison isolating the specific degradation introduced by applying INT8 Post-Training Quantization to the collapsed structures. The \\textbf{FP32 $\\Delta$} tracks the accuracy change of the collapsed model relative to the fully trained Control model. The \\textbf{INT8 $\\Delta$} tracks the additional penalty incurred solely by quantizing that FP32 collapsed model. Missing values (-) denote incomplete runs or empty architectural sets.}",
        "\\label{tab:quantization_results}", "\\begin{tabular}{@{}ll | cc | cc | cc@{}}",
        "\\toprule", "\\multirow{2}{*}{\\textbf{Architecture}} & \\multirow{2}{*}{\\textbf{Pre/Fine}} & \\multicolumn{2}{c|}{\\textbf{CIFAR-10}} & \\multicolumn{2}{c|}{\\textbf{CIFAR-100}} & \\multicolumn{2}{c}{\\textbf{Tiny ImageNet}} \\\\",
        "\\cmidrule(l){3-4} \\cmidrule(l){5-6} \\cmidrule(l){7-8}", " & & \\textbf{\\makecell{FP32 $\\Delta$ \\\\ (vs Control)}} & \\textbf{\\makecell{INT8 $\\Delta$ \\\\ (vs FP32)}} & \\textbf{\\makecell{FP32 $\\Delta$ \\\\ (vs Control)}} & \\textbf{\\makecell{INT8 $\\Delta$ \\\\ (vs FP32)}} & \\textbf{\\makecell{FP32 $\\Delta$ \\\\ (vs Control)}} & \\textbf{\\makecell{INT8 $\\Delta$ \\\\ (vs FP32)}} \\\\", "\\midrule"
    ]
    
    for arch in architectures:
        best_das = {ds: get_best_delta_acc(df, ds, arch, False) for ds in datasets}
        
        for i, (pre, fine) in enumerate(epochs_list):
            row = [f"\\multirow{{3}}{{*}}{{\\textbf{{{arch.replace('_', '\\_')}}}}}" if i == 0 else "", f"{pre}/{fine}"]
            for ds in datasets:
                m_fp32 = get_metrics(df, ds, arch, pre=pre, fine=fine, quantized=False)
                m_int8 = get_metrics(df, ds, arch, pre=pre, fine=fine, quantized=True)
                
                if m_fp32 is not None:
                    fp_val = m_fp32['Delta_Acc']
                    fp_str = f"{fp_val:+.2f}"
                    if fp_val == best_das[ds] and fp_val != -999.0:
                        fp_str = f"\\textbf{{{fp_str}}}"
                        
                    if m_int8 is not None:
                        ptq_penalty = m_int8['Delta_Acc'] - m_fp32['Delta_Acc']
                        int_str = f"{ptq_penalty:+.2f}" 
                    else:
                        int_str = "-"
                else:
                    fp_str, int_str = "-", "-"
                    
                row.extend([fp_str, int_str])
            latex_quant.append(" & ".join(row) + " \\\\")
        latex_quant.append("\\midrule")
    latex_quant[-1] = "\\bottomrule"
    latex_quant.extend(["\\end{tabular}", "\\end{table*}"])
    with open("table_quantization.tex", "w") as f: f.write("\n".join(latex_quant))
    
    print("✅ LaTeX Tables Generated (table_hardware_efficiency.tex, table_quantization.tex)")

def generate_control_table(base_path="."):
    """Parses JSON logs to extract fully trained Control models and format them into a strict LaTeX table."""
    metrics_dict = {}
    search_pattern = os.path.join(base_path, "*", "metrics", "merged_metrics.json")
    
    for filepath in glob.glob(search_pattern):
        folder_name = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
        
        if "_Cifar100_" in folder_name: dataset = "CIFAR-100"
        elif "_Cifar10_" in folder_name: dataset = "CIFAR-10"
        elif "_tinyimagenet_" in folder_name: dataset = "Tiny ImageNet"
        else: continue
        
        arch = "RegNetX_400MF" if folder_name.startswith("RegNetX_400MF") else folder_name.split("_")[0]
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                ctrl_key = next((k for k in data.keys() if "Control" in k and "quant" not in k and "Continuted" not in k), None)
                if not ctrl_key: continue
                ctrl = data[ctrl_key]
                
                acc = ctrl.get("best_acc", ctrl.get("final_accuracy", 0))
                params = ctrl.get("param_count", 0) / 1e6
                flops = ctrl.get("flops", 0) / 1e9
                
                if arch not in metrics_dict: metrics_dict[arch] = {}
                if dataset not in metrics_dict[arch]:
                    metrics_dict[arch][dataset] = {'acc': acc, 'params': params, 'flops': flops}
                else:
                    metrics_dict[arch][dataset]['acc'] = max(metrics_dict[arch][dataset]['acc'], acc)
                    metrics_dict[arch][dataset]['params'] = max(metrics_dict[arch][dataset]['params'], params)
                    metrics_dict[arch][dataset]['flops'] = max(metrics_dict[arch][dataset]['flops'], flops)
        except Exception: pass

    latex_lines = [
        "\\begin{table*}[htbp]",
        "    \\centering",
        "    \\caption{\\textbf{Control Performance and Hardware Metrics for Evaluated Architectures.} Performance of the uncompressed control models following the complete pre-training and finetuning budget. Accuracies denote the maximum top-1 validation recovery. Parameters are reported in millions (M) and FLOPs in billions (G).}",
        "    \\label{tab:control_metrics}",
        "    \\begin{tabular}{l | ccc | ccc | ccc}\\toprule",
        "        \\textbf{Architecture} & \\multicolumn{3}{c|}{\\textbf{CIFAR-10}} & \\multicolumn{3}{c|}{\\textbf{CIFAR-100}} & \\multicolumn{3}{c}{\\textbf{Tiny ImageNet}} \\\\",
        "        \\cmidrule{2-10}",
        "        & \\textbf{Acc (\\%)} & \\textbf{Params (M)} & \\textbf{FLOPs (G)} & \\textbf{Acc (\\%)} & \\textbf{Params (M)} & \\textbf{FLOPs (G)} & \\textbf{Acc (\\%)} & \\textbf{Params (M)} & \\textbf{FLOPs (G)} \\\\",
        "        \\midrule"
    ]
    
    architectures = sorted(list(metrics_dict.keys()))
    for arch in architectures:
        row = [f"        \\textbf{{{arch.replace('_', '\\_')}}}"]
        for ds in ["CIFAR-10", "CIFAR-100", "Tiny ImageNet"]:
            if ds in metrics_dict[arch]:
                d = metrics_dict[arch][ds]
                row.append(f"{d['acc']:.2f} & {d['params']:.2f} & {d['flops']:.2f}")
            else:
                row.append("- & - & -")
                
        latex_lines.append(" & ".join(row) + " \\\\")
        
    latex_lines.extend([
        "        \\bottomrule",
        "    \\end{tabular}",
        "\\end{table*}"
    ])
    
    with open("table_control_metrics.tex", "w") as f: f.write("\n".join(latex_lines))
    print("✅ LaTeX Table Generated (table_control_metrics.tex)")

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("Parsing text logs and augmenting with JSON data...")
    df = parse_txt_files(["info.txt"])
    df = augment_df_with_json(df, base_path=".")
    
    if not df.empty:
        generate_journal_figures(df)
        generate_tables(df)
    else:
        print("❌ Error: No valid data found in info.txt or JSON files.")
        
    print("Generating Control Metrics Table...")
    generate_control_table(base_path=".")