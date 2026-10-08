import pandas as pd
import io
import re
import os

def parse_txt_files(file_paths):
    """Parses output.txt into a Pandas DataFrame."""
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
            
            match = re.search(r'figures_ep(\d+)_pre(\d+)', header)
            if match:
                pre_train_ep = int(match.group(1))
                post_collapse_ep = int(match.group(2))
            else: continue
                
            if not csv_data.strip(): continue
                
            try:
                df = pd.read_csv(io.StringIO(csv_data))
                df['Pre_Train_Epochs'] = pre_train_ep
                df['Post_Collapse_Epochs'] = post_collapse_ep
                data_frames.append(df)
            except Exception:
                continue
                
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

def get_metrics(df, dataset, arch, pre, fine, quantized=False):
    """Safely retrieves metrics for a specific run, returning None if incomplete/missing."""
    subset = df[
        (df['Dataset'] == dataset) &
        (df['Architecture'] == arch) &
        (df['Pre_Train_Epochs'] == pre) &
        (df['Post_Collapse_Epochs'] == fine) &
        (df['Experiment'] == 'Dynamic_Region_All_Combined') &
        (df['Is_Quantized'] == quantized)
    ].dropna(subset=['Delta_Acc', 'Params_Reduction_%'])
    
    if subset.empty:
        return None
    return subset.iloc[0]

def format_delta_acc(val):
    """Formats accuracy with a forced positive sign for gains."""
    if pd.isna(val): return "N/A"
    return f"+{val:.2f}" if val > 0 else f"{val:.2f}"

def format_delta_acc_percent(val):
    """Formats accuracy with a forced positive sign and percent symbol for Table 2."""
    if pd.isna(val): return "N/A"
    return f"+{val:.3f}\\%" if val > 0 else f"{val:.3f}\\%"

def generate_hardware_efficiency_table(df):
    """Generates the massive IEEE Consolidated Hardware Efficiency Table."""
    
    datasets = ['CIFAR-10', 'CIFAR-100', 'TinyImageNet']
    architectures = ['VGG-16', 'RegNetX_400MF', 'ConvNeXt', 'InceptionNet', 'XceptionNet']
    epochs = [(100, 300), (200, 200), (300, 100)]
    
    latex_out = []
    latex_out.append("% ==========================================")
    latex_out.append("% AUTO-GENERATED TABLE: Consolidated Hardware Efficiency")
    latex_out.append("% ==========================================")
    latex_out.append("\\begin{table*}[htbp]")
    latex_out.append("\\centering")
    latex_out.append("\\caption{Consolidated Hardware Efficiency Profiles utilizing the Combined Dynamic Region. Metrics reflect the percentage reduction relative to the uncompressed baseline control across varying pretraining and finetuning epoch budgets. Pending optimization runs are denoted by N/A.}")
    latex_out.append("\\label{tab:consolidated_hardware_efficiency}")
    latex_out.append("\\begin{tabular}{@{}llccccc@{}}")
    latex_out.append("\\toprule")
    latex_out.append("\\textbf{Dataset} & \\textbf{Architecture} & \\textbf{Pre / Fine Ep.} & \\textbf{Params Red. (\\%)} & \\textbf{FLOPs Red. (\\%)} & \\textbf{Memory Red. (\\%)} & \\textbf{$\\Delta$ Accuracy (\\%)} \\\\")
    latex_out.append("\\midrule")
    
    for d_idx, dataset in enumerate(datasets):
        # Format Dataset name for LaTeX display
        display_dataset = "Tiny ImageNet" if dataset == "TinyImageNet" else dataset
        
        latex_out.append(f"% ------------------------------------------")
        latex_out.append(f"% {display_dataset}")
        latex_out.append(f"% ------------------------------------------")
        
        # Calculate total rows for this dataset (5 architectures * 3 epoch configs = 15)
        latex_out.append(f"\\multirow{{15}}{{*}}{{\\textbf{{{display_dataset}}}}}")
        
        for a_idx, arch in enumerate(architectures):
            # Escape underscores for LaTeX
            display_arch = arch.replace('_', '\\_')
            
            for e_idx, (pre, fine) in enumerate(epochs):
                row_str = ""
                
                # Column 1: Dataset (only on first row of dataset)
                if a_idx == 0 and e_idx == 0:
                    row_str += "& "
                elif e_idx == 0:
                    row_str += "& "
                else:
                    row_str += "& "
                    
                # Column 2: Architecture (only on first row of architecture)
                if e_idx == 0:
                    row_str += f"\\multirow{{3}}{{*}}{{{display_arch}}} "
                row_str += f"& {pre} / {fine} & "
                
                # Fetch Data
                metrics = get_metrics(df, dataset, arch, pre, fine, quantized=False)
                
                if metrics is not None:
                    pr = f"{metrics['Params_Reduction_%']:.2f}"
                    fr = f"{metrics['FLOPs_Reduction_%']:.2f}"
                    mr = f"{metrics['Memory_Reduction_%']:.2f}"
                    acc = format_delta_acc(metrics['Delta_Acc'])
                    row_str += f"{pr} & {fr} & {mr} & {acc} \\\\"
                else:
                    row_str += "N/A & N/A & N/A & N/A \\\\"
                    
                latex_out.append(row_str)
            
            # Add midrule between architectures, but not after the last one in the dataset
            if a_idx < len(architectures) - 1:
                latex_out.append("\\cmidrule{2-7}")
                
        # Add midrule between datasets, but not after the last one
        if d_idx < len(datasets) - 1:
            latex_out.append("\\midrule")
            
    latex_out.append("\\bottomrule")
    latex_out.append("\\end{tabular}")
    latex_out.append("\\end{table*}")
    
    return "\n".join(latex_out)

def generate_quantization_table(df):
    """Generates the IEEE Quantization Robustness Table."""
    
    # Define the specific targeted runs you highlighted in your text
    target_runs = [
        ("CIFAR-10", "VGG-16", 100, 300),
        ("CIFAR-10", "ConvNeXt", 200, 200),
        ("CIFAR-100", "RegNetX_400MF", 100, 300),
        ("CIFAR-100", "XceptionNet", 200, 200),
        ("TinyImageNet", "InceptionNet", 200, 200),
    ]
    
    latex_out = []
    latex_out.append("% ==========================================")
    latex_out.append("% AUTO-GENERATED TABLE: Quantization Robustness")
    latex_out.append("% ==========================================")
    latex_out.append("\\begin{table}[htbp]")
    latex_out.append("\\centering")
    latex_out.append("\\caption{Impact of INT8 Post-Training Quantization (PTQ) on Structurally Collapsed Models. Metrics reflect $\\Delta$ Accuracy (\\%) relative to the uncompressed FP32 baseline control.}")
    latex_out.append("\\label{tab:quantization_results}")
    latex_out.append("\\begin{tabular}{@{}llccc@{}}")
    latex_out.append("\\toprule")
    latex_out.append("\\textbf{Dataset} & \\textbf{Architecture} & \\textbf{Pre / Fine Ep.} & \\textbf{FP32 $\\Delta$ Acc.} & \\textbf{INT8 $\\Delta$ Acc.} \\\\")
    latex_out.append("\\midrule")
    
    current_dataset = ""
    
    for dataset, arch, pre, fine in target_runs:
        display_dataset = "Tiny ImageNet" if dataset == "TinyImageNet" else dataset
        display_arch = arch.replace('_', '\\_')
        
        # Determine if we need a new dataset multirow block
        if display_dataset != current_dataset:
            if current_dataset != "":
                latex_out.append("\\midrule")
            current_dataset = display_dataset
            
            # Count how many rows belong to this dataset
            dataset_count = sum(1 for r in target_runs if r[0] == dataset)
            if dataset_count > 1:
                latex_out.append(f"\\multirow{{{dataset_count}}}{{*}}{{\\textbf{{{display_dataset}}}}} ")
            else:
                latex_out.append(f"\\textbf{{{display_dataset}}} ")
        else:
            latex_out.append("& ") # Empty space for dataset column
            
        latex_out[-1] += f"& {display_arch} & {pre} / {fine} & "
        
        # Fetch FP32 and INT8 metrics
        fp32_metrics = get_metrics(df, dataset, arch, pre, fine, quantized=False)
        int8_metrics = get_metrics(df, dataset, arch, pre, fine, quantized=True)
        
        fp32_val = fp32_metrics['Delta_Acc'] if fp32_metrics is not None else float('nan')
        int8_val = int8_metrics['Delta_Acc'] if int8_metrics is not None else float('nan')
        
        latex_out[-1] += f"{format_delta_acc_percent(fp32_val)} & {format_delta_acc_percent(int8_val)} \\\\"

    latex_out.append("\\bottomrule")
    latex_out.append("\\end{tabular}")
    latex_out.append("\\end{table}")
    
    return "\n".join(latex_out)

if __name__ == "__main__":
    print("Parsing output.txt...")
    df = parse_txt_files(["info.txt"])
    
    if not df.empty:
        # 1. Generate Hardware Table
        hardware_tex = generate_hardware_efficiency_table(df)
        with open("table_hardware_efficiency.tex", "w") as f:
            f.write(hardware_tex)
        print("\n✅ Saved: table_hardware_efficiency.tex")
        
        # 2. Generate Quantization Table
        quant_tex = generate_quantization_table(df)
        with open("table_quantization.tex", "w") as f:
            f.write(quant_tex)
        print("✅ Saved: table_quantization.tex")
        
        # Also print to console for quick copy-pasting
        print("\n\n--- PREVIEW: HARDWARE EFFICIENCY TABLE ---")
        print(hardware_tex)
        
    else:
        print("❌ Error: No valid data found. Make sure output.txt is in the directory.")