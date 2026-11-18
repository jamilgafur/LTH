import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Canonical architectures expected in the benchmark suite
CANONICAL_MODELS = ['ConvNeXt', 'InceptionNet', 'MobileNet', 'RegNetX_400MF', 'VGG16', 'XceptionNet']

def load_and_compute_deltas():
    """Reads CSVs, skips empty files, and computes hardware-normalized deltas."""
    csv_files = glob.glob("cka_results_*.csv")
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
                # Properly detect fp16 from 'precision' column (new main.py) or filename (legacy)
                if 'precision' in temp_df.columns:
                    temp_df['quant'] = (
                        temp_df['precision']
                        .astype(str).str.strip().str.lower()
                        .isin(['fp16', 'quant', 'true', '1'])
                    )
                elif 'quant' in temp_df.columns:
                    temp_df['quant'] = (
                        temp_df['quant']
                        .astype(str).str.strip().str.lower()
                        .isin(['fp16', 'quant', 'true', '1'])
                    )
                else:
                    temp_df['quant'] = '_quant' in os.path.basename(f)
                df_list.append(temp_df)
        except pd.errors.EmptyDataError:
            print(f"Skipping unparsable/empty CSV data: {f}")
            
    if not df_list:
        print("No valid data found in any of the CSV files.")
        return None, []
        
    df = pd.concat(df_list, ignore_index=True)
    df = df[df['model'] != 'model'].reset_index(drop=True)

    # Drop status-only placeholder rows ('no_collapsed_regions')
    if 'status' in df.columns:
        df = df[df['status'].isna() | (df['status'] != 'no_collapsed_regions')].reset_index(drop=True)

    # Convert numeric columns (excluding string metadata columns so 'precision' is preserved)
    str_cols = {'model', 'dataset', 'model_before', 'gpu_name', 'quant', 'precision', 'status', 'ood_source'}
    num_cols = [col for col in df.columns if col not in str_cols]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Median over rerun reps: per-GPU median drops single polluted power samples, then median across GPUs
    g6 = ['dataset', 'quant', 'pre_epochs', 'post_epochs', 'model', 'gpu_name']
    df = (df.groupby(g6, as_index=False).median(numeric_only=True)
            .groupby(g6[:-1], as_index=False).median(numeric_only=True))

    df['pre_epochs'] = df['pre_epochs'].astype(int)
    df['post_epochs'] = df['post_epochs'].astype(int)
    
    # 1. Accuracy Deltas (percentage points; using TEST accuracy everywhere)
    df['delta_acc_imp'] = (df['acc_test_unstructured'] - df['acc_test_original_after']) * 100.0
    df['delta_acc_collapse'] = (df['acc_test_collapsed'] - df['acc_test_original_after']) * 100.0
    
    # 2. Energy Savings: Prefer compute-saturated J/img (zero DataLoader overhead) when available,
    #    falling back to eval-window energy_joules_*
    has_sat = (
        'j_per_img_saturated_original' in df.columns
        and df['j_per_img_saturated_original'].notna().any()
    )
    if has_sat:
        orig_e = df['j_per_img_saturated_original'].replace(0, np.nan)
        imp_e = df['j_per_img_saturated_unstructured']
        col_e = df['j_per_img_saturated_collapsed']
    else:
        orig_e = df['energy_joules_original'].replace(0, np.nan)
        imp_e = df['energy_joules_unstructured']
        col_e = df['energy_joules_collapsed']

    df['energy_saved_imp'] = orig_e - imp_e
    df['energy_saved_collapse'] = orig_e - col_e
    df['energy_saved_imp_pct'] = ((orig_e - imp_e) / orig_e) * 100.0
    df['energy_saved_collapse_pct'] = ((orig_e - col_e) / orig_e) * 100.0
    
    # 3. Adversarial Deltas (percentage points)
    df['delta_fgsm_imp'] = (df['adv_acc_FGSM_unstructured'] - df['adv_acc_FGSM_original']) * 100.0
    df['delta_fgsm_collapse'] = (df['adv_acc_FGSM_collapsed'] - df['adv_acc_FGSM_original']) * 100.0
    df['delta_pgd_imp'] = (df['adv_acc_PGD_unstructured'] - df['adv_acc_PGD_original']) * 100.0
    df['delta_pgd_collapse'] = (df['adv_acc_PGD_collapsed'] - df['adv_acc_PGD_original']) * 100.0

    # 4. FLOPs Reduction: Separate Actual Measured Hardware FLOPs vs Theoretical Parameter Sparsity
    if 'flops_original' in df.columns:
        safe_original_flops = df['flops_original'].replace(0, np.nan)
        # Theoretical non-zero parameter sparsity (what main.py stored in flops_unstructured)
        df['flops_reduction_imp_theo_pct'] = (df['flops_original'] - df['flops_unstructured']) / safe_original_flops
        # Actual measured dense-tensor hardware FLOPs for IMP (0% reduction on standard GPUs)
        if 'flops_unstructured_measured' in df.columns:
            df['flops_reduction_imp_hw_pct'] = (df['flops_original'] - df['flops_unstructured_measured']) / safe_original_flops
        else:
            df['flops_reduction_imp_hw_pct'] = 0.0
        # Keep flops_reduction_imp_pct as theoretical for scatter plot back-compat
        df['flops_reduction_imp_pct'] = df['flops_reduction_imp_theo_pct']
        # Actual measured hardware FLOPs for Collapsed (true structural reduction)
        df['flops_reduction_collapse_pct'] = (df['flops_original'] - df['flops_collapsed']) / safe_original_flops
        df['pruned_imp'] = df['flops_reduction_imp_theo_pct']

    # 5. Storage footprint reduction
    if 'storage_mb_original' in df.columns:
        safe_orig_storage = df['storage_mb_original'].replace(0, np.nan)
        df['pruned_collapse'] = (df['storage_mb_original'] - df['storage_mb_collapsed']) / safe_orig_storage
        for m in ['unstructured', 'collapsed']:
            df[f'storage_reduction_{m}_pct'] = (df['storage_mb_original'] - df[f'storage_mb_{m}']) / safe_orig_storage

    # 6. Efficiency ratios vs original (for speed/memory plots)
    for m in ['unstructured', 'collapsed']:
        df[f'latency_ratio_{m}'] = df[f'latency_ms_{m}'] / df['latency_ms_original'].replace(0, np.nan)
        df[f'throughput_speedup_{m}'] = df[f'throughput_imgs_{m}'] / df['throughput_imgs_original'].replace(0, np.nan)
        df[f'peakmem_ratio_{m}'] = df[f'peak_mem_mb_{m}'] / df['peak_mem_mb_original'].replace(0, np.nan)
        df[f'featmap_ratio_{m}'] = df[f'feature_maps_mb_{m}'] / df['feature_maps_mb_original'].replace(0, np.nan)

    all_models = sorted(list(set(df['model'].dropna().unique()).union(CANONICAL_MODELS)))
    return df, all_models

def highlight_missing_models(ax, all_models, missing_models):
    """Draws red indicators and colors x-tick labels red for models completely missing data."""
    for idx, model in enumerate(all_models):
        if model in missing_models:
            ax.axvspan(idx - 0.38, idx + 0.38, color='red', alpha=0.12, zorder=0)
            ax.text(idx, 0, "MISSING", color='red', fontsize=10, fontweight='bold',
                    ha='center', va='center', rotation=90, zorder=3)

    plt.draw()
    for tick, model in zip(ax.get_xticklabels(), all_models):
        if model in missing_models:
            tick.set_color('red')
            tick.set_fontweight('bold')
        else:
            tick.set_color('black')

def draw_uni_bars(ax, df_fp32, df_quant, all_models, columns, labels, panel_title,
                  zero_line=False, legend=False, df_annotations=None, anno_fmt="{val:.2f}"):
    """Draws both universes' grouped bars on ONE axes: fp32 = solid, fp16 = hatched/alpha."""
    from matplotlib.patches import Patch
    rf = df_fp32.set_index('model').reindex(all_models)
    rq = df_quant.set_index('model').reindex(all_models)

    m, n = len(columns), len(all_models)
    x = np.arange(n)
    span = 0.84
    bw = span / (2 * m)
    colors = sns.color_palette('Set2', m)

    for i, col in enumerate(columns):
        for u, d in enumerate([rf, rq]):
            pos = x + (2 * i + u - (2 * m - 1) / 2) * bw
            if u == 0:
                ax.bar(pos, d[col].values, width=bw, color=colors[i], edgecolor='black', linewidth=0.6)
            else:
                ax.bar(pos, d[col].values, width=bw, color=colors[i], alpha=0.55,
                       hatch='///', edgecolor='#444444', linewidth=0.4)

    # Only draw a red MISSING span when BOTH fp32 and fp16 are missing for a model (e.g., MobileNet).
    # If a model has fp32 data but lacks fp16 (e.g., VGG16 before --quant rerun), do not put a red bar through it.
    half = span / 2
    miss_both = set()
    non_control_cols = [c for c in columns if c != 'delta_acc_control']
    
    for idx, model in enumerate(all_models):
        missing_fp32 = rf.loc[model, non_control_cols].isna().all() if model in rf.index else True
        missing_quant = rq.loc[model, non_control_cols].isna().all() if model in rq.index else True
        
        if missing_fp32 and missing_quant:
            miss_both.add(model)
            ax.axvspan(idx - half, idx + half, color='red', alpha=0.12, zorder=0)
            ax.text(idx, 0, 'MISSING', color='red', fontsize=9, fontweight='bold',
                    ha='center', va='center', rotation=90, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(all_models, rotation=45, ha='right')
    plt.draw()
    for tick, model in zip(ax.get_xticklabels(), all_models):
        if model in miss_both:
            tick.set_color('red')
            tick.set_fontweight('bold')

    if zero_line:
        ax.axhline(0, color='black', linewidth=1.2, linestyle='--')
    pad_shared_ylim(ax)

    if df_annotations is not None:
        # Annotate fp32 bars only: fp16 values are near-duplicates and adjacent labels collide
        fp32_rects = [rect for i, container in enumerate(ax.containers) if i % 2 == 0
                      for rect in container]
        anno_vals = pd.concat([rf[a] for a in df_annotations], axis=1).values.flatten('F')
        for p, val in zip(fp32_rects, anno_vals):
            height = p.get_height()
            if pd.isna(val) or pd.isna(height):
                continue
            xytext = (0, 5) if height >= 0 else (0, -15)
            ax.annotate(anno_fmt.format(val=val),
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom' if height >= 0 else 'top',
                        xytext=xytext, textcoords='offset points',
                        fontsize=8, fontweight='bold', color='#333333')

    if legend:
        handles = [Patch(facecolor=colors[i], edgecolor='black', label=labels[i]) for i in range(m)]
        handles += [Patch(facecolor='white', edgecolor='black', label='fp32 (control, solid)'),
                    Patch(facecolor='white', edgecolor='#444444', hatch='///', alpha=0.55,
                          label='fp16 (quant, hatched)')]
        ax.legend(handles=handles, title='Modification / Universe',
                  bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)

    ax.set_title(panel_title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=11)
    return rf, rq

def draw_bar_panel(ax, df, all_models, columns, labels, panel_title, zero_line=False, legend=False):
    """Draws one universe's grouped bars into ax; returns the reindexed df for later annotation."""
    df_reindexed = df.set_index('model').reindex(all_models)
    missing_models = set(df_reindexed[df_reindexed[columns].isna().all(axis=1)].index)

    df_plot = df_reindexed[columns]
    df_plot.columns = labels
    df_plot.plot(kind='bar', ax=ax, width=0.7, colormap='Set2', legend=legend)

    if zero_line:
        ax.axhline(0, color='black', linewidth=1.2, linestyle='--')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title(panel_title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=11)
    highlight_missing_models(ax, all_models, missing_models)
    return df_reindexed

def annotate_bars(ax, df_anno, anno_fmt, fontsize=9):
    """Writes raw values on bars; container order must match df column order."""
    rects = [rect for container in ax.containers for rect in container]
    anno_vals = df_anno.values.flatten('F')
    for p, val in zip(rects, anno_vals):
        height = p.get_height()
        if pd.isna(val) or pd.isna(height) or height == 0:
            continue
        xytext = (0, 5) if height >= 0 else (0, -15)
        ax.annotate(anno_fmt.format(val=val),
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom' if height >= 0 else 'top',
                    xytext=xytext, textcoords='offset points',
                    fontsize=fontsize, fontweight='bold', color='#333333')

def pad_shared_ylim(ax):
    """Pads the shared y-axis so bar annotations fit."""
    ymin, ymax = ax.get_ylim()
    yrange = (ymax - ymin) if (ymax - ymin) > 0 else 1.0
    ax.set_ylim(ymin - (yrange * 0.15) if ymin < 0 else 0, ymax + (yrange * 0.20))

def plot_universe_comparison(df_fp32, df_quant, all_models, columns, labels, title, ylabel, filename,
                             zero_line=False, df_annotations=None, anno_fmt="{val:.2f}"):
    """Single-axes grouped bars: fp32 solid + fp16 hatched per method."""
    fig, ax = plt.subplots(figsize=(16, 7))
    sns.set_theme(style="whitegrid")
    draw_uni_bars(ax, df_fp32, df_quant, all_models, columns, labels, title,
                  zero_line=zero_line, legend=True, df_annotations=df_annotations, anno_fmt=anno_fmt)
    ax.set_ylabel(ylabel, fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved '{filename}'")
    plt.close()

def plot_grid(df_fp32, df_quant, all_models, rows, title, ylabels, filename):
    """N-row single-column grid; each axes merges fp32+fp16 (solid vs hatched)."""
    fig, axes = plt.subplots(len(rows), 1, figsize=(16, 6 * len(rows) + 1), squeeze=False)
    sns.set_theme(style="whitegrid")

    for r, (row_title, cols, labels, anno_cols, anno_fmt, zero_line) in enumerate(rows):
        ax = axes[r][0]
        draw_uni_bars(ax, df_fp32, df_quant, all_models, cols, labels, row_title,
                      zero_line=zero_line, legend=(r == 0), df_annotations=anno_cols, anno_fmt=anno_fmt)
        ax.set_ylabel(ylabels[r], fontsize=11)

    fig.suptitle(title, fontsize=15, y=1.0)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved '{filename}'")
    plt.close()

# FIXED: Use TEST accuracy columns ('acc_test_*'), not training accuracy ('acc_*')!
SCATTER_METHODS = [
    ('IMP (Unstructured)', 'acc_test_unstructured', 'flops_reduction_imp_pct'),
    ('Collapsed', 'acc_test_collapsed', 'flops_reduction_collapse_pct'),
]

def plot_scatter_universes(df_fp32, df_quant, all_models, title, xlabel, ylabel, filename,
                           pareto=False, diagonal=False):
    """Single-axes scatter: color = method; filled = fp32, open = fp16."""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette('Set2', 3)

    for d, filled in [(df_fp32, True), (df_quant, False)]:
        for (label, acc_col, flops_col), color in zip(SCATTER_METHODS, colors):
            sub = d.dropna(subset=[acc_col])
            x = sub[flops_col] * 100 if pareto else sub['acc_test_original_after']
            pts = pd.DataFrame({'x': x, 'y': sub[acc_col]}).dropna()
            if pts.empty:
                continue
            if filled:
                ax.scatter(pts['x'], pts['y'], alpha=0.85, s=70, color=color)
            else:
                ax.scatter(pts['x'], pts['y'], s=70, facecolors='none', edgecolors=color, linewidths=1.5)
    for (label, acc_col, flops_col), color in zip(SCATTER_METHODS, colors):
        sub = df_fp32.dropna(subset=[acc_col])
        x = sub[flops_col] * 100 if pareto else sub['acc_test_original_after']
        pts = pd.DataFrame({'x': x, 'y': sub[acc_col]}).dropna()
        for xi, yi, mdl in zip(pts['x'], pts['y'], sub.loc[pts.index, 'model']):
            ax.annotate(mdl, (xi, yi), textcoords='offset points', xytext=(4, 4), fontsize=8)

    if diagonal:
        ax.plot([0, 1.05], [0, 1.05], ls='--', color='black', lw=1)
    handles = [Line2D([], [], marker='o', color='w', markerfacecolor=c, markersize=9, label=l)
               for (l, _, _), c in zip(SCATTER_METHODS, colors)]
    handles += [Line2D([], [], marker='o', color='w', markerfacecolor='#888888', markersize=9, label='fp32 (control, filled)'),
                Line2D([], [], marker='o', color='w', markerfacecolor='white', markeredgecolor='#888888',
                       markeredgewidth=1.5, markersize=9, label='fp16 (quant, open)')]
    ax.legend(handles=handles, title='Modification / Universe', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved '{filename}'")
    plt.close()

# FIXED: Use TEST accuracy columns for PTQ penalty
ACC_METHODS = [
    ('IMP (Unstructured)', 'acc_test_unstructured'),
    ('Collapsed', 'acc_test_collapsed'),
]

def plot_ptq_penalty(df_fp32, df_quant, all_models, title, filename):
    """Grouped bars: fp16 minus fp32 test accuracy (percentage points) per model and method."""
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.set_theme(style="whitegrid")

    labels = [m[0] for m in ACC_METHODS]
    pen = pd.DataFrame({label: (df_quant.set_index('model')[col] - df_fp32.set_index('model')[col]) * 100
                        for label, col in ACC_METHODS})
    pen.index.name = 'model'
    dr = draw_bar_panel(ax, pen.reset_index(), all_models, labels, labels,
                        title, zero_line=True, legend=True)
    pad_shared_ylim(ax)
    annotate_bars(ax, dr[labels], "{val:+.2f}")
    ax.set_ylabel('PTQ Test Accuracy Penalty (pp)\n(Negative = fp16 conversion hurt accuracy)\n$\\Delta = \\mathrm{Acc}_{\\mathrm{fp16}} - \\mathrm{Acc}_{\\mathrm{fp32}}$', fontsize=11)
    ax.legend(title='Modification Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved '{filename}'")
    plt.close()

def plot_budget_transfer(df, all_models, dataset, filename):
    """Accuracy delta of the collapsed arm across (pre, post) budgets; solid = fp32, dashed = fp16."""
    from matplotlib.lines import Line2D
    d = df[df['dataset'] == dataset]
    configs = sorted(set(zip(d['pre_epochs'], d['post_epochs'])))
    xlabels = [f"pre{p}/post{q}" for p, q in configs]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette('tab10', len(all_models))
    handles = []
    for model, color in zip(all_models, colors):
        plotted = False
        for dd, ls in [(d[~d['quant']], '-'), (d[d['quant']], '--')]:
            sub = dd[dd['model'] == model].set_index(['pre_epochs', 'post_epochs']).reindex(configs)
            if sub['delta_acc_collapse'].notna().sum() == 0:
                continue
            ax.plot(range(len(configs)), sub['delta_acc_collapse'], marker='o', ls=ls,
                    color=color, alpha=0.9 if ls == '-' else 0.7)
            plotted = True
        if plotted:
            handles.append(Line2D([], [], ls='-', color=color, marker='o', label=model))
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(xlabels)
    ax.axhline(0, color='black', linewidth=1.2, linestyle='--')
    handles += [Line2D([], [], ls='-', color='black', label='fp32 (control, solid)'),
                Line2D([], [], ls='--', color='black', alpha=0.7, label='fp16 (quant, dashed)')]
    ax.legend(handles=handles, title='Model / Universe', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_ylabel('Test Accuracy Delta (pp, Collapsed vs Control)', fontsize=12)
    ax.set_xlabel('Budget (pre / post epochs)', fontsize=11)
    ax.set_title(f'Budget Transfer — How pre/post split affects collapse accuracy\n({dataset})', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved '{filename}'")
    plt.close()

def plot_cka_heatmap(df, filename):
    """Global CKA heatmaps: rows = IMP/Collapsed, cols = fp32/fp16; cells = dataset x config."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), squeeze=False)
    sns.set_theme(style="whitegrid")

    pivots, vals = {}, []
    for r, (mname, col) in enumerate([('IMP (Unstructured)', 'cka_unstructured'), ('Collapsed', 'cka_collapsed')]):
        for c, (univ, dd) in enumerate([('fp32 (control)', df[~df['quant']]), ('fp16 (quant)', df[df['quant']])]):
            t = dd.copy()
            t['config'] = 'pre' + t['pre_epochs'].astype(str) + '/post' + t['post_epochs'].astype(str)
            piv = t.pivot_table(index='dataset', columns='config', values=col, aggfunc='mean')
            pivots[(r, c, mname, univ)] = piv
            vals.append(piv.values.ravel())

    valid_vals = [v[~np.isnan(v)] for v in vals if v.size > 0 and not np.all(np.isnan(v))]
    if not valid_vals:
        plt.close()
        return
    allv = np.concatenate(valid_vals)
    vmin, vmax = allv.min(), allv.max()
    for (r, c, mname, univ), piv in pivots.items():
        if piv.empty:
            continue
        sns.heatmap(piv, annot=True, fmt='.3f', cmap='viridis', vmin=vmin, vmax=vmax,
                    ax=axes[r][c], cbar=(c == 1))
        axes[r][c].set_title(f"{mname} — {univ}", fontsize=12, fontweight='bold')
        axes[r][c].set_ylabel('Dataset' if c == 0 else '')
        axes[r][c].set_xlabel('Config' if r == 1 else '')
    fig.suptitle('CKA to Original (mean over models)', fontsize=15)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved '{filename}'")
    plt.close()

def plot_pruning_amount(df_fp32, all_models, title, filename):
    """Fraction of the original model removed per arm (removed dead Structural column)."""
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.set_theme(style="whitegrid")
    cols = ['pruned_imp', 'pruned_collapse']
    labels = ['IMP (Target Param Sparsity)', 'Collapsed (Storage Reduction)']
    dr = draw_bar_panel(ax, df_fp32, all_models, cols, labels, title, legend=True)
    pad_shared_ylim(ax)
    annotate_bars(ax, dr[cols], "{val:.1%}")
    ax.set_ylabel('Fraction of Original Model Pruned\n(Higher = More Removed)', fontsize=12)
    ax.legend(title='Modification Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved '{filename}'")
    plt.close()

def plot_xy_scatter(df_fp32, df_quant, methods, title, xlabel, ylabel, filename,
                    hzero=False, vzero=False):
    """Single-axes scatter: color = method; filled = fp32, open = fp16."""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette('Set2', max(len(methods), 3))

    for d, filled in [(df_fp32, True), (df_quant, False)]:
        for (label, xcol, ycol), color in zip(methods, colors):
            pts = d[['model', xcol, ycol]].dropna()
            if pts.empty:
                continue
            if filled:
                ax.scatter(pts[xcol], pts[ycol], alpha=0.85, s=70, color=color)
            else:
                ax.scatter(pts[xcol], pts[ycol], s=70, facecolors='none',
                           edgecolors=color, linewidths=1.5)
    for (label, xcol, ycol), color in zip(methods, colors):
        pts = df_fp32[['model', xcol, ycol]].dropna()
        for xi, yi, mdl in zip(pts[xcol], pts[ycol], pts['model']):
            ax.annotate(mdl, (xi, yi), textcoords='offset points', xytext=(4, 4), fontsize=8)

    if hzero:
        ax.axhline(0, color='black', lw=1.2, ls='--')
    if vzero:
        ax.axvline(0, color='black', lw=1.2, ls='--')
    handles = [Line2D([], [], marker='o', color='w', markerfacecolor=c, markersize=9, label=l)
               for (l, _, _), c in zip(methods, colors)]
    handles += [Line2D([], [], marker='o', color='w', markerfacecolor='#888888', markersize=9, label='fp32 (control, filled)'),
                Line2D([], [], marker='o', color='w', markerfacecolor='white', markeredgecolor='#888888',
                       markeredgewidth=1.5, markersize=9, label='fp16 (quant, open)')]
    ax.legend(handles=handles, title='Modification / Universe', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved '{filename}'")
    plt.close()

def main():
    CONTROL_TEST_ACCURACY_400 = {
        'CIFAR-10': {'ConvNeXt': 74.44, 'InceptionNet': 84.52, 'MobileNet': 81.89, 
                     'RegNetX_400MF': 75.77, 'VGG16': 93.07, 'XceptionNet': 88.05},
        'CIFAR-100': {'ConvNeXt': 42.25, 'InceptionNet': 40.06, 'MobileNet': 44.58,
                      'RegNetX_400MF': 37.58, 'VGG16': 63.07, 'XceptionNet': 46.24},
        'tinyimagenet': {'ConvNeXt': 42.96, 'InceptionNet': 46.39, 'MobileNet': 43.05,
                        'RegNetX_400MF': 38.11, 'VGG16': 46.50, 'XceptionNet': 48.79}
    }
    
    df, all_models = load_and_compute_deltas()
    if df is None:
        return

    grouped = df.groupby(['dataset', 'pre_epochs', 'post_epochs'])

    for (dataset, pre, post), group_df in grouped:
        outdir = os.path.join('plots', str(dataset), f"pre{pre}_post{post}")
        os.makedirs(outdir, exist_ok=True)
        print(f"\n--- Generating plots for {dataset} | Pre: {pre} | Post: {post} -> {outdir}/ ---")

        df_fp32 = group_df[~group_df['quant']].copy()
        df_quant = group_df[group_df['quant']].copy()

        title_suffix = f"\n({dataset} | Pre: {pre}, Post: {post})"
        
        # Override Control test accuracy with table values if using 400 total epochs,
        # AND recompute delta_acc_imp / delta_acc_collapse so bar heights match annotations!
        total_epochs = pre + post
        use_table_control = (total_epochs == 400) and (dataset in CONTROL_TEST_ACCURACY_400)
        if use_table_control:
            for model in all_models:
                if model in CONTROL_TEST_ACCURACY_400[dataset]:
                    table_val = CONTROL_TEST_ACCURACY_400[dataset][model] / 100.0
                    df_fp32.loc[df_fp32['model'] == model, 'acc_test_original_after'] = table_val
                    df_quant.loc[df_quant['model'] == model, 'acc_test_original_after'] = table_val
            for d_sub in (df_fp32, df_quant):
                d_sub['delta_acc_imp'] = (d_sub['acc_test_unstructured'] - d_sub['acc_test_original_after']) * 100.0
                d_sub['delta_acc_collapse'] = (d_sub['acc_test_collapsed'] - d_sub['acc_test_original_after']) * 100.0

        def fname(name):
            return os.path.join(outdir, f"{name}.png")

        # Plot 1: Change in Task Accuracy (TEST accuracy)
        plot_universe_comparison(
            df_fp32=df_fp32, df_quant=df_quant, all_models=all_models,
            columns=['delta_acc_imp', 'delta_acc_collapse'],
            labels=['IMP (Unstructured)', 'Collapsed'],
            title=f'Test Accuracy Delta (Compared to Control){title_suffix}',
            ylabel='Test Accuracy Delta (pp)\n(Higher is Better)\n' + r'$\Delta = \mathrm{Acc}_{\mathrm{Mod}} - \mathrm{Acc}_{\mathrm{Control}}$',
            filename=fname('plot_1_delta_accuracy'),
            zero_line=True,
            df_annotations=['acc_test_unstructured', 'acc_test_collapsed'],
            anno_fmt="{val:.1%}"
        )

        # Plot 2: Normalized Energy Savings (%)
        plot_universe_comparison(
            df_fp32=df_fp32, df_quant=df_quant, all_models=all_models,
            columns=['energy_saved_imp_pct', 'energy_saved_collapse_pct'],
            labels=['IMP (Unstructured)', 'Collapsed'],
            title=f'Relative Energy Savings (% Reduced from Baseline){title_suffix}',
            ylabel='Energy Saved (%)\n(Higher is Better)\n' + r'$\Delta = \frac{E_{\mathrm{Orig}} - E_{\mathrm{Mod}}}{E_{\mathrm{Orig}}} \times 100$',
            filename=fname('plot_2_energy_savings'),
            zero_line=True,
            df_annotations=['energy_saved_imp_pct', 'energy_saved_collapse_pct'],
            anno_fmt="{val:+.1f}%"
        )

        # Plot 3: CKA Scores
        plot_universe_comparison(
            df_fp32=df_fp32, df_quant=df_quant, all_models=all_models,
            columns=['cka_unstructured', 'cka_collapsed'],
            labels=['IMP (Unstructured)', 'Collapsed'],
            title=f'Representational Similarity (CKA) to Original{title_suffix}',
            ylabel='CKA Score\n(Higher = More Similar to Original)\n' + r'$\mathrm{CKA}(X, Y) \in [0, 1]$',
            filename=fname('plot_3_cka_scores'),
            zero_line=False,
            df_annotations=['cka_unstructured', 'cka_collapsed'],
            anno_fmt="{val:.3f}"
        )

        # Plot 4: Adversarial Robustness
        adv_ylabel = 'Accuracy Delta under Attack (pp)\n(Higher is Better)\n' + r'$\Delta = \mathrm{AdvAcc}_{\mathrm{Mod}} - \mathrm{AdvAcc}_{\mathrm{Orig}}$'
        plot_grid(
            df_fp32=df_fp32, df_quant=df_quant, all_models=all_models,
            rows=[
                ('FGSM', ['delta_fgsm_imp', 'delta_fgsm_collapse'], ['IMP (Unstructured)', 'Collapsed'],
                 ['adv_acc_FGSM_unstructured', 'adv_acc_FGSM_collapsed'], "{val:.1%}", True),
                ('PGD', ['delta_pgd_imp', 'delta_pgd_collapse'], ['IMP (Unstructured)', 'Collapsed'],
                 ['adv_acc_PGD_unstructured', 'adv_acc_PGD_collapsed'], "{val:.1%}", True),
            ],
            title=f'Change in Adversarial Robustness{title_suffix}',
            ylabels=[adv_ylabel, adv_ylabel],
            filename=fname('plot_4_delta_adversarial')
        )
        
        # Plot 5: FLOPs Reduction — Shows Actual Hardware Executed FLOPs vs Theoretical Sparsity
        if 'flops_original' in group_df.columns:
            plot_universe_comparison(
                df_fp32=df_fp32, df_quant=df_quant, all_models=all_models,
                columns=['flops_reduction_imp_hw_pct', 'flops_reduction_imp_theo_pct', 'flops_reduction_collapse_pct'],
                labels=['IMP (Measured Hardware FLOPs)', 'IMP (Theoretical Param Sparsity)', 'Collapsed (Measured Hardware FLOPs)'],
                title=f'FLOPs Reduction: Hardware Executed vs. Theoretical{title_suffix}',
                ylabel='FLOPs Reduction (fraction)\n(Higher = More Compressed)\n' + r'$\Delta = \frac{\mathrm{FLOPs}_{\mathrm{Orig}} - \mathrm{FLOPs}_{\mathrm{Mod}}}{\mathrm{FLOPs}_{\mathrm{Orig}}}$',
                filename=fname('plot_5_flops_reduction'),
                zero_line=True,
                df_annotations=['flops_reduction_imp_hw_pct', 'flops_reduction_imp_theo_pct', 'flops_reduction_collapse_pct'],
                anno_fmt="{val:.1%}"
            )

        # Plot 6: Pareto frontier — TEST accuracy vs FLOPs reduction
        if 'flops_original' in group_df.columns:
            plot_scatter_universes(
                df_fp32=df_fp32, df_quant=df_quant, all_models=all_models,
                title=f'Test Accuracy vs Compression (Pareto){title_suffix}',
                xlabel='FLOPs Reduction (%)', ylabel='Test Accuracy (Higher is Better)',
                filename=fname('plot_6_pareto_accuracy_flops'),
                pareto=True
            )

        # Plot 7: Absolute TEST accuracy vs original (y = x reference)
        plot_scatter_universes(
            df_fp32=df_fp32, df_quant=df_quant, all_models=all_models,
            title=f'Absolute Test Accuracy vs Original{title_suffix}',
            xlabel='Original Test Accuracy (after fine-tune)', ylabel='Pruned-Model Test Accuracy',
            filename=fname('plot_7_absolute_accuracy'),
            diagonal=True
        )

        # Plot 8: Latency & Throughput speedups
        plot_grid(
            df_fp32=df_fp32, df_quant=df_quant, all_models=all_models,
            rows=[
                ('Latency (ratio to original)', ['latency_ratio_unstructured', 'latency_ratio_collapsed'],
                 ['IMP (Unstructured)', 'Collapsed'],
                 ['latency_ratio_unstructured', 'latency_ratio_collapsed'], "{val:.2f}x", False),
                ('Throughput (speedup vs original)', ['throughput_speedup_unstructured', 'throughput_speedup_collapsed'],
                 ['IMP (Unstructured)', 'Collapsed'],
                 ['throughput_speedup_unstructured', 'throughput_speedup_collapsed'], "{val:.2f}x", False),
            ],
            title=f'Measured Speed: Latency & Throughput{title_suffix}',
            ylabels=[
                'Latency Ratio (Lower = Faster)\n' + r'$\mathrm{Ratio} = \frac{\mathrm{Latency}_{\mathrm{Mod}}}{\mathrm{Latency}_{\mathrm{Orig}}}$',
                'Throughput Speedup (Higher = Faster)\n' + r'$\mathrm{Speedup} = \frac{\mathrm{IPS}_{\mathrm{Mod}}}{\mathrm{IPS}_{\mathrm{Orig}}}$'
            ],
            filename=fname('plot_8_latency_throughput')
        )

        # Plot 9: Memory (peak & activation) ratios
        plot_grid(
            df_fp32=df_fp32, df_quant=df_quant, all_models=all_models,
            rows=[
                ('Peak GPU memory (ratio to original)', ['peakmem_ratio_unstructured', 'peakmem_ratio_collapsed'],
                 ['IMP (Unstructured)', 'Collapsed'],
                 ['peakmem_ratio_unstructured', 'peakmem_ratio_collapsed'], "{val:.2f}x", False),
                ('Activation memory (ratio to original)', ['featmap_ratio_unstructured', 'featmap_ratio_collapsed'],
                 ['IMP (Unstructured)', 'Collapsed'],
                 ['featmap_ratio_unstructured', 'featmap_ratio_collapsed'], "{val:.2f}x", False),
            ],
            title=f'Memory Footprint{title_suffix}',
            ylabels=[
                'Peak Memory Ratio (Lower = Smaller)\n' + r'$\mathrm{Ratio} = \frac{\mathrm{Mem}_{\mathrm{Mod}}}{\mathrm{Mem}_{\mathrm{Orig}}}$',
                'Activation Memory Ratio (Lower = Smaller)\n' + r'$\mathrm{Ratio} = \frac{\mathrm{ActMem}_{\mathrm{Mod}}}{\mathrm{ActMem}_{\mathrm{Orig}}}$'
            ],
            filename=fname('plot_9_memory')
        )

        # Plot 10: PTQ penalty (fp16 - fp32 TEST accuracy)
        plot_ptq_penalty(
            df_fp32=df_fp32, df_quant=df_quant, all_models=all_models,
            title=f'fp16 PTQ Penalty on Test Accuracy{title_suffix}',
            filename=fname('plot_10_ptq_penalty')
        )

        # Plot 13: How much of the original model was pruned
        if 'flops_original' in group_df.columns:
            plot_pruning_amount(
                df_fp32=df_fp32, all_models=all_models,
                title=f'Model Compression by Pruning (vs Original){title_suffix}',
                filename=fname('plot_13_pruning_amount')
            )

        # Plot 14: On-disk storage reduction
        if 'storage_mb_original' in group_df.columns:
            plot_universe_comparison(
                df_fp32=df_fp32, df_quant=df_quant, all_models=all_models,
                columns=['storage_reduction_unstructured_pct', 'storage_reduction_collapsed_pct'],
                labels=['IMP (Unstructured)', 'Collapsed'],
                title=f'Storage Footprint Reduction (vs Original){title_suffix}',
                ylabel='Storage Reduction (fraction)\n(Higher = Smaller Checkpoint)\n' + r'$\Delta = \frac{\mathrm{Size}_{\mathrm{Orig}} - \mathrm{Size}_{\mathrm{Mod}}}{\mathrm{Size}_{\mathrm{Orig}}}$',
                filename=fname('plot_14_storage_reduction'),
                zero_line=False,
                df_annotations=['storage_reduction_unstructured_pct', 'storage_reduction_collapsed_pct'],
                anno_fmt="{val:.1%}"
            )

        # Plot 15: Robustness-vs-accuracy trade-off
        plot_xy_scatter(
            df_fp32=df_fp32, df_quant=df_quant,
            methods=[('IMP (Unstructured)', 'delta_acc_imp', 'delta_pgd_imp'),
                     ('Collapsed', 'delta_acc_collapse', 'delta_pgd_collapse')],
            title=f'Robustness vs Test Accuracy Trade-off (PGD){title_suffix}',
            xlabel='Clean Test Accuracy Delta (pp)', ylabel='PGD Accuracy Delta (pp)',
            filename=fname('plot_15_robustness_vs_accuracy'),
            hzero=True, vzero=True
        )

        # Plot 16: CKA vs accuracy retention
        plot_xy_scatter(
            df_fp32=df_fp32, df_quant=df_quant,
            methods=[('IMP (Unstructured)', 'cka_unstructured', 'delta_acc_imp'),
                     ('Collapsed', 'cka_collapsed', 'delta_acc_collapse')],
            title=f'Representational Similarity vs Test Accuracy Retention{title_suffix}',
            xlabel='CKA to Original', ylabel='Test Accuracy Delta (pp)',
            filename=fname('plot_16_cka_vs_accuracy'),
            hzero=True
        )

        # Plot 17: Measured energy savings vs TEST accuracy (real-hardware Pareto)
        plot_xy_scatter(
            df_fp32=df_fp32, df_quant=df_quant,
            methods=[('IMP (Unstructured)', 'energy_saved_imp_pct', 'acc_test_unstructured'),
                     ('Collapsed', 'energy_saved_collapse_pct', 'acc_test_collapsed')],
            title=f'Energy Savings vs Test Accuracy (Pareto){title_suffix}',
            xlabel='Energy Saved vs Original (%)', ylabel='Test Accuracy (Higher is Better)',
            filename=fname('plot_17_energy_vs_accuracy')
        )

    # Plot 11: Budget-transfer curves
    for dataset in df['dataset'].unique():
        outdir = os.path.join('plots', str(dataset))
        os.makedirs(outdir, exist_ok=True)
        plot_budget_transfer(
            df=df, all_models=all_models, dataset=dataset,
            filename=os.path.join(outdir, 'plot_11_budget_transfer_accuracy.png')
        )

    # Plot 12: Global CKA heatmap
    os.makedirs('plots', exist_ok=True)
    plot_cka_heatmap(df=df, filename=os.path.join('plots', 'plot_12_cka_heatmap.png'))
    
    print("\nAll delta comparisons generated successfully!")
    
if __name__ == "__main__":
    main()