import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Canonical architectures expected in the benchmark suite
CANONICAL_MODELS = ['ConvNeXt', 'InceptionNet', 'MobileNet', 'RegNetX_400MF', 'VGG16', 'XceptionNet']

ADV_ATTACKS = [
    ('FGSM', 'delta_fgsm_imp', 'delta_fgsm_collapse', 'adv_acc_FGSM_unstructured', 'adv_acc_FGSM_collapsed'),
    ('PGD', 'delta_pgd_imp', 'delta_pgd_collapse', 'adv_acc_PGD_unstructured', 'adv_acc_PGD_collapsed'),
]

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
                temp_df['quant'] = '_quant' in f
                df_list.append(temp_df)
        except pd.errors.EmptyDataError:
            print(f"Skipping unparsable/empty CSV data: {f}")
            
    if not df_list:
        print("No valid data found in any of the CSV files.")
        return None, []
        
    df = pd.concat(df_list, ignore_index=True)
    df = df[df['model'] != 'model'].reset_index(drop=True)
    # shortcut: rerun files can hold multiple rows per config; keep the last row, upgrade to a median if that starts mattering
    df = df.drop_duplicates(subset=['dataset', 'quant', 'pre_epochs', 'post_epochs', 'model'], keep='last').reset_index(drop=True)

    # Convert numeric columns (excluding string metadata columns)
    num_cols = [col for col in df.columns if col not in ['model', 'dataset', 'model_before', 'gpu_name', 'quant']]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['pre_epochs'] = df['pre_epochs'].astype(int)
    df['post_epochs'] = df['post_epochs'].astype(int)
    
    # 1. Accuracy Deltas
    df['delta_acc_imp'] = df['acc_unstructured'] - df['acc_original_after']
    df['delta_acc_collapse'] = df['acc_collapsed'] - df['acc_original_after']
    
    # 2. Energy Savings (Raw Joules & Hardware-Normalized Relative %)
    df['energy_saved_imp'] = df['energy_joules_original'] - df['energy_joules_unstructured']
    df['energy_saved_collapse'] = df['energy_joules_original'] - df['energy_joules_collapsed']
    
    safe_orig_energy = df['energy_joules_original'].replace(0, np.nan)
    df['energy_saved_imp_pct'] = ((df['energy_joules_original'] - df['energy_joules_unstructured']) / safe_orig_energy) * 100.0
    df['energy_saved_collapse_pct'] = ((df['energy_joules_original'] - df['energy_joules_collapsed']) / safe_orig_energy) * 100.0
    
    # 3. Adversarial Deltas
    df['delta_fgsm_imp'] = df['adv_acc_FGSM_unstructured'] - df['adv_acc_FGSM_original']
    df['delta_fgsm_collapse'] = df['adv_acc_FGSM_collapsed'] - df['adv_acc_FGSM_original']
    df['delta_pgd_imp'] = df['adv_acc_PGD_unstructured'] - df['adv_acc_PGD_original']
    df['delta_pgd_collapse'] = df['adv_acc_PGD_collapsed'] - df['adv_acc_PGD_original']

    if 'flops_original' in df.columns:
        safe_original_flops = df['flops_original'].replace(0, np.nan)
        df['flops_reduction_imp_pct'] = (df['flops_original'] - df['flops_unstructured']) / safe_original_flops
        df['flops_reduction_collapse_pct'] = (df['flops_original'] - df['flops_collapsed']) / safe_original_flops
        df['flops_reduction_structural_pct'] = (df['flops_original'] - df['flops_structural']) / safe_original_flops

    # 4. Structural arm deltas (columns are empty when torch-pruning was unavailable)
    df['delta_acc_structural'] = df['acc_structural'] - df['acc_original_after']
    df['energy_saved_structural_pct'] = ((df['energy_joules_original'] - df['energy_joules_structural']) / safe_orig_energy) * 100.0

    # 5. Efficiency ratios vs original (for speed/memory plots)
    for m in ['unstructured', 'collapsed', 'structural']:
        df[f'latency_ratio_{m}'] = df[f'latency_ms_{m}'] / df['latency_ms_original'].replace(0, np.nan)
        df[f'throughput_speedup_{m}'] = df[f'throughput_imgs_{m}'] / df['throughput_imgs_original'].replace(0, np.nan)
        df[f'peakmem_ratio_{m}'] = df[f'peak_mem_mb_{m}'] / df['peak_mem_mb_original'].replace(0, np.nan)
        df[f'featmap_ratio_{m}'] = df[f'feature_maps_mb_{m}'] / df['feature_maps_mb_original'].replace(0, np.nan)

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
                ax.bar(pos, d[col].values, width=bw, color=colors[i], edgecolor='black', linewidth=0.4)
            else:
                ax.bar(pos, d[col].values, width=bw, color=colors[i], alpha=0.55,
                       hatch='///', edgecolor='#444444', linewidth=0.0)

    # Per-universe MISSING half-spans; tick red only if missing in both universes
    half = span / 2
    miss_both = set()
    for idx, model in enumerate(all_models):
        missing = [rf.loc[model, columns].isna().all(), rq.loc[model, columns].isna().all()]
        if all(missing):
            miss_both.add(model)
        for u, is_missing in enumerate(missing):
            if is_missing:
                lo = idx - half if u == 0 else idx
                ax.axvspan(lo, lo + half, color='red', alpha=0.12, zorder=0)
                ax.text(lo + half / 2, 0, 'MISSING', color='red', fontsize=7, fontweight='bold',
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
        interleaved = pd.concat([f for acol in df_annotations for f in (rf[acol], rq[acol])], axis=1)
        annotate_bars(ax, interleaved, anno_fmt, fontsize=8)

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
    # ax.containers = bar columns only; ax.patches also includes MISSING axvspans
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
    ax.set_ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved '{filename}'")
    plt.close()

def plot_grid(df_fp32, df_quant, all_models, rows, title, ylabels, filename):
    """N-row single-column grid; each axes merges fp32+fp16 (solid vs hatched).
    rows = list of (row_title, bar_cols, labels, anno_cols_or_None, anno_fmt, zero_line)."""
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

SCATTER_METHODS = [
    ('IMP (Unstructured)', 'acc_unstructured', 'flops_reduction_imp_pct'),
    ('Collapsed', 'acc_collapsed', 'flops_reduction_collapse_pct'),
    ('Structural', 'acc_structural', 'flops_reduction_structural_pct'),
]

def plot_scatter_universes(df_fp32, df_quant, all_models, title, xlabel, ylabel, filename,
                           pareto=False, diagonal=False):
    """1x2 universe scatter: accuracy on y; x = FLOPs reduction % (pareto) or original accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette('Set2', 3)

    for ax, (univ, d) in zip(axes, [('fp32 (control)', df_fp32), ('fp16 (quant)', df_quant)]):
        for (label, acc_col, flops_col), color in zip(SCATTER_METHODS, colors):
            sub = d.dropna(subset=[acc_col])
            x = sub[flops_col] * 100 if pareto else sub['acc_original_after']
            pts = pd.DataFrame({'x': x, 'y': sub[acc_col]}).dropna()
            if pts.empty:
                continue
            ax.scatter(pts['x'], pts['y'], label=label, alpha=0.85, s=70, color=color)
            for xi, yi, mdl in zip(pts['x'], pts['y'], sub.loc[pts.index, 'model']):
                ax.annotate(mdl, (xi, yi), textcoords='offset points', xytext=(4, 4), fontsize=8)
        if diagonal:
            ax.plot([0, 1.05], [0, 1.05], ls='--', color='black', lw=1)
        ax.set_title(univ, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
    axes[0].set_ylabel(ylabel, fontsize=12)
    axes[1].legend(title='Modification Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved '{filename}'")
    plt.close()


ACC_METHODS = [
    ('IMP (Unstructured)', 'acc_unstructured'),
    ('Collapsed', 'acc_collapsed'),
    ('Structural', 'acc_structural'),
]

def plot_ptq_penalty(df_fp32, df_quant, all_models, title, filename):
    """Grouped bars: fp16 minus fp32 accuracy (percentage points) per model and method."""
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
    ax.set_ylabel('PTQ penalty (percentage points)\n(Negative = fp16 conversion hurt accuracy)', fontsize=12)
    ax.legend(title='Modification Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved '{filename}'")
    plt.close()

def plot_budget_transfer(df, all_models, dataset, filename):
    """Accuracy delta of the collapsed arm across (pre, post) budgets, lines per model."""
    d = df[df['dataset'] == dataset]
    configs = sorted(set(zip(d['pre_epochs'], d['post_epochs'])))
    xlabels = [f"pre{p}/post{q}" for p, q in configs]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    sns.set_theme(style="whitegrid")
    for ax, (univ, dd) in zip(axes, [('fp32 (control)', d[~d['quant']]), ('fp16 (quant)', d[d['quant']])]):
        for model in all_models:
            sub = dd[dd['model'] == model].set_index(['pre_epochs', 'post_epochs']).reindex(configs)
            if sub['delta_acc_collapse'].notna().sum() == 0:
                continue
            ax.plot(range(len(configs)), sub['delta_acc_collapse'], marker='o', label=model)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(xlabels)
        ax.axhline(0, color='black', linewidth=1.2, linestyle='--')
        ax.set_title(univ, fontsize=12, fontweight='bold')
        ax.set_xlabel('Budget (pre / post epochs)', fontsize=11)
    axes[0].set_ylabel('Accuracy Delta (Collapsed vs Original)', fontsize=12)
    axes[1].legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.suptitle(f'Budget Transfer — How pre/post split affects collapse accuracy\n({dataset})', fontsize=14, y=1.02)
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

    allv = np.concatenate([v[~np.isnan(v)] for v in vals if not np.all(np.isnan(v))])
    vmin, vmax = allv.min(), allv.max()
    for (r, c, mname, univ), piv in pivots.items():
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

def main():
    df, all_models = load_and_compute_deltas()
    if df is None:
        return

    grouped = df.groupby(['dataset', 'pre_epochs', 'post_epochs'])

    for (dataset, pre, post), group_df in grouped:
        outdir = os.path.join('plots', str(dataset))
        os.makedirs(outdir, exist_ok=True)
        print(f"\n--- Generating plots for {dataset} | Pre: {pre} | Post: {post} -> {outdir}/ ---")

        df_fp32 = group_df[~group_df['quant']]
        df_quant = group_df[group_df['quant']]

        title_suffix = f"\n({dataset} | Pre: {pre}, Post: {post})"

        def fname(name):
            return os.path.join(outdir, f"{name}_pre{pre}_post{post}.png")

        # Plot 1: Change in Task Accuracy
        plot_universe_comparison(
            df_fp32=df_fp32, df_quant=df_quant, all_models=all_models,
            columns=['delta_acc_imp', 'delta_acc_collapse', 'delta_acc_structural'],
            labels=['IMP (Unstructured)', 'Collapsed', 'Structural'],
            title=f'Change in Task Accuracy (Compared to Original){title_suffix}',
            ylabel='Accuracy Delta\n(Higher is Better)',
            filename=fname('plot_1_delta_accuracy'),
            zero_line=True,
            df_annotations=['acc_unstructured', 'acc_collapsed', 'acc_structural'],
            anno_fmt="{val:.1%}"
        )

        # Plot 2: Normalized Energy Savings (%) across heterogeneous GPUs
        plot_universe_comparison(
            df_fp32=df_fp32, df_quant=df_quant, all_models=all_models,
            columns=['energy_saved_imp_pct', 'energy_saved_collapse_pct', 'energy_saved_structural_pct'],
            labels=['IMP (Unstructured)', 'Collapsed', 'Structural'],
            title=f'Relative Energy Savings (% Reduced from Baseline){title_suffix}',
            ylabel='Energy Saved (%)\n(Higher is Better)',
            filename=fname('plot_2_energy_savings'),
            zero_line=True,
            df_annotations=['energy_saved_imp_pct', 'energy_saved_collapse_pct', 'energy_saved_structural_pct'],
            anno_fmt="{val:+.1f}%"
        )

        # Plot 3: CKA Scores
        plot_universe_comparison(
            df_fp32=df_fp32, df_quant=df_quant, all_models=all_models,
            columns=['cka_unstructured', 'cka_collapsed', 'cka_structural'],
            labels=['IMP (Unstructured)', 'Collapsed', 'Structural'],
            title=f'Representational Similarity (CKA) to Original{title_suffix}',
            ylabel='CKA Score\n(Higher = More Similar to Original)',
            filename=fname('plot_3_cka_scores'),
            zero_line=False,
            df_annotations=['cka_unstructured', 'cka_collapsed', 'cka_structural'],
            anno_fmt="{val:.3f}"
        )

        # Plot 4: Adversarial Robustness 2x2 grid (rows = attack, cols = universe)
        plot_adversarial_universes(
            df_fp32=df_fp32, df_quant=df_quant, all_models=all_models,
            title=f'Change in Adversarial Robustness{title_suffix}',
            ylabel='Accuracy Delta under Attack\n(Higher is Better)',
            filename=fname('plot_4_delta_adversarial')
        )
        
        # Plot 5: FLOPs Reduction (% below original)
        if 'flops_original' in group_df.columns:
            plot_universe_comparison(
                df_fp32=df_fp32, df_quant=df_quant, all_models=all_models,
                columns=['flops_reduction_imp_pct', 'flops_reduction_collapse_pct', 'flops_reduction_structural_pct'],
                labels=['IMP (Unstructured)', 'Collapsed', 'Structural'],
                title=f'FLOPs Reduction (Compared to Original){title_suffix}',
                ylabel='FLOPs Reduction (fraction)\n(Higher = More Compressed)',
                filename=fname('plot_5_flops_reduction'),
                zero_line=False,
                df_annotations=['flops_reduction_imp_pct', 'flops_reduction_collapse_pct', 'flops_reduction_structural_pct'],
                anno_fmt="{val:.1%}"
            )

        # Plot 6: Pareto frontier — accuracy vs FLOPs reduction
        if 'flops_original' in group_df.columns:
            plot_scatter_universes(
                df_fp32=df_fp32, df_quant=df_quant, all_models=all_models,
                title=f'Accuracy vs Compression (Pareto){title_suffix}',
                xlabel='FLOPs Reduction (%)', ylabel='Accuracy (Higher is Better)',
                filename=fname('plot_6_pareto_accuracy_flops'),
                pareto=True
            )

        # Plot 7: Absolute accuracy vs original (y = x reference)
        plot_scatter_universes(
            df_fp32=df_fp32, df_quant=df_quant, all_models=all_models,
            title=f'Absolute Accuracy vs Original{title_suffix}',
            xlabel='Original Accuracy (after fine-tune)', ylabel='Pruned-Model Accuracy',
            filename=fname('plot_7_absolute_accuracy'),
            diagonal=True
        )

        # Plot 8: Latency & Throughput speedups
        plot_grid(
            df_fp32=df_fp32, df_quant=df_quant, all_models=all_models,
            rows=[
                ('Latency (ratio to original)', ['latency_ratio_unstructured', 'latency_ratio_collapsed', 'latency_ratio_structural'],
                 ['IMP (Unstructured)', 'Collapsed', 'Structural'],
                 ['latency_ratio_unstructured', 'latency_ratio_collapsed', 'latency_ratio_structural'], "{val:.2f}x", True),
                ('Throughput (speedup vs original)', ['throughput_speedup_unstructured', 'throughput_speedup_collapsed', 'throughput_speedup_structural'],
                 ['IMP (Unstructured)', 'Collapsed', 'Structural'],
                 ['throughput_speedup_unstructured', 'throughput_speedup_collapsed', 'throughput_speedup_structural'], "{val:.2f}x", True),
            ],
            title=f'Measured Speed: Latency & Throughput{title_suffix}',
            ylabels=['Latency Ratio\n(Lower = Faster)', 'Throughput Speedup\n(Higher = Faster)'],
            filename=fname('plot_8_latency_throughput')
        )

        # Plot 9: Memory (peak & activation) ratios
        plot_grid(
            df_fp32=df_fp32, df_quant=df_quant, all_models=all_models,
            rows=[
                ('Peak GPU memory (ratio to original)', ['peakmem_ratio_unstructured', 'peakmem_ratio_collapsed', 'peakmem_ratio_structural'],
                 ['IMP (Unstructured)', 'Collapsed', 'Structural'],
                 ['peakmem_ratio_unstructured', 'peakmem_ratio_collapsed', 'peakmem_ratio_structural'], "{val:.2f}x", False),
                ('Activation memory (ratio to original)', ['featmap_ratio_unstructured', 'featmap_ratio_collapsed', 'featmap_ratio_structural'],
                 ['IMP (Unstructured)', 'Collapsed', 'Structural'],
                 ['featmap_ratio_unstructured', 'featmap_ratio_collapsed', 'featmap_ratio_structural'], "{val:.2f}x", False),
            ],
            title=f'Memory Footprint{title_suffix}',
            ylabels=['Peak Memory Ratio\n(Lower = Smaller)', 'Activation Memory Ratio\n(Lower = Smaller)'],
            filename=fname('plot_9_memory')
        )

        # Plot 10: PTQ penalty (fp16 - fp32 accuracy)
        plot_ptq_penalty(
            df_fp32=df_fp32, df_quant=df_quant, all_models=all_models,
            title=f'fp16 PTQ Penalty on Accuracy{title_suffix}',
            filename=fname('plot_10_ptq_penalty')
        )

    # Plot 11: Budget-transfer curves (one per dataset, spans all pre/post configs)
    for dataset in df['dataset'].unique():
        outdir = os.path.join('plots', str(dataset))
        os.makedirs(outdir, exist_ok=True)
        plot_budget_transfer(
            df=df, all_models=all_models, dataset=dataset,
            filename=os.path.join(outdir, 'plot_11_budget_transfer_accuracy.png')
        )

    # Plot 12: Global CKA heatmap (dataset x config, IMP/Collapsed x universe)
    os.makedirs('plots', exist_ok=True)
    plot_cka_heatmap(df=df, filename=os.path.join('plots', 'plot_12_cka_heatmap.png'))
    
    print("\nAll delta comparisons generated successfully!")
    
if __name__ == "__main__":
    main()
