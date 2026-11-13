"""Dissertation figure layer: IEEE vector-PDF figures from the cka_results CSVs.

One figure per proposed-work question:
  Q1  fig_cka_{ds}.pdf                CKA vs control per variant (representation)
  Q2  fig_robustness_curves_{ds}.pdf  FGSM/PGD accuracy vs epsilon
      fig_reliability_{ds}.pdf        ECE/MCE calibration
      fig_ood_{ds}.pdf                OOD AUROC
      fig_corruption_{ds}.pdf         mean corruption accuracy vs severity
  Q3  fig_ptq_{ds}.pdf                fp32 vs fp16 (PTQ): accuracy + saturated J/img
  Q4  fig_energy_transfer.pdf         removed FLOPs vs removed joules per config
      fig_efficiency_{ds}.pdf         hardware ratios vs control

Variant palette is Control / IMP / Collapsed (the L1-structural comparator was
retired — its artifacts measured at chance on both splits — and is not plotted,
even where older CSVs still carry its columns).

All summaries are medians with IQR (pooled figures take each config's own median
first, then the spread across configs). Missing data means a skipped figure with
a [WARN] — no panel is ever drawn from thin air.

Row policy matches aggregate_results.py: verdict rows (no_collapsed_regions) are
not measurements and CPU rows shadowed by a GPU twin are dropped. `_None_`-token
control dirs carry pairing_flag=none_token_dir: flagged and counted in the run
summary, never silently dropped (their measured rows are valid; the flag names a
suspect control pairing, and the near-zero CKA tail is in the named-scorer family).

Usage: python plot_dissertation.py [csv_dir] [out_dir]
       (defaults: . and <csv_dir>/dissertation_figs)
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = [("original", "Control"), ("unstructured", "IMP"),
            ("collapsed", "Collapsed")]
COLORS = {"Control": "#4d4d4d", "IMP": "#1f77b4", "Collapsed": "#d62728"}
MODEL_ORDER = ["VGG16", "RegNetX_400MF", "XceptionNet", "InceptionNet",
               "MobileNet", "ConvNeXt"]
DS_LABELS = {"Cifar10": "CIFAR-10", "Cifar100": "CIFAR-100",
             "tinyimagenet": "Tiny ImageNet"}
CFG = ["arch", "dataset", "pre", "post", "quant"]
STEM_RE = re.compile(
    r"^(?P<arch>.+?)_(?P<dataset>Cifar100|Cifar10|tinyimagenet)"
    r"(?P<quant>_quant)?_pre(?P<pre>\d+)_post(?P<post>\d+)$")

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "figure.figsize": (3.5, 2.4),
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def warn(fig, reason):
    print(f"[WARN] {fig}: skipped ({reason})")


def parse_stem(stem):
    m = STEM_RE.match(stem)
    return m.groupdict() if m else None


def load_rows(d):
    """Pool every cka_results CSV into one frame; returns None when empty."""
    frames = []
    for f in sorted(glob.glob(os.path.join(d, "cka_results_*.csv"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        tag = stem[len("cka_results_"):]
        meta = parse_stem(tag)
        if meta is None:
            print(f"[WARN] unparseable filename, skipped: {f}")
            continue
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        for k, v in meta.items():
            df[k] = (v == "_quant") if k == "quant" else (int(v) if k in ("pre", "post") else v)
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df[df["model"].astype(str) != "model"]  # header echo rows
    status = df.get("status", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    df = df[status.isin(["", "ok"])]  # verdict rows are not measurements
    before = df.get("model_before", pd.Series("", index=df.index)).astype(str)
    n_flag = int(before.str.contains("_None_", regex=False, na=False).sum())
    if n_flag:
        print(f"[INFO] {n_flag} rows carry pairing_flag=none_token_dir "
              "(suspect control pairing — included, see aggregate_results.py)")
    # CPU rows shadowed by a GPU twin of the same config are dropped
    is_cpu = df.get("gpu_name", pd.Series("", index=df.index)).astype(str).str.upper().eq("CPU")
    if is_cpu.any():
        gpu_cfgs = set(map(tuple, df.loc[~is_cpu, CFG].drop_duplicates().values))
        df = df[~(is_cpu & df[CFG].apply(tuple, axis=1).isin(gpu_cfgs))]
    for c in df.columns:
        if c not in ("model", "dataset", "gpu_name", "model_before", "status",
                     "arch", "ood_source", "regime", "source"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.reset_index(drop=True)


def pooled(df, col):
    """Median + IQR: each config's own median over reps first, spread across configs."""
    if col not in df or df[col].dropna().empty:
        return np.nan, np.nan, np.nan
    per_cfg = df.groupby(CFG, observed=True)[col].median().dropna()
    if per_cfg.empty:
        return np.nan, np.nan, np.nan
    med, q1, q3 = per_cfg.median(), per_cfg.quantile(0.25), per_cfg.quantile(0.75)
    return float(med), float(med - q1), float(q3 - med)


def acc_col(key, df):
    """Prefer the test-split read; fall back to the train read (old schema)."""
    test, train = f"acc_test_{key}", ("acc_original_after" if key == "original" else f"acc_{key}")
    if test in df and df[test].notna().any():
        return test
    return train if train in df else None


def grouped_bars(ax, groups, series, df, title, ylabel):
    """series: list of (column, label). One bar group per entry in groups."""
    xs = np.arange(len(groups))
    width = 0.8 / max(len(series), 1)
    for i, (col, label) in enumerate(series):
        ys, lo, hi = [], [], []
        for g in groups:
            y, l, h = pooled(df[df["group"] == g], col)
            ys.append(y), lo.append(l), hi.append(h)
        ax.bar(xs + i * width - 0.4 + width / 2, ys, width,
               yerr=[lo, hi], label=label, color=COLORS[label],
               capsize=1.5, linewidth=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.margins(y=0.1)


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


def long_frames(d, prefix):
    """Pool long-format side CSVs (robustness_curves_/corruption_); adds CFG cols."""
    frames = []
    for f in sorted(glob.glob(os.path.join(d, f"{prefix}_*.csv"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        meta = parse_stem(stem[len(prefix) + 1:])
        if meta is None:
            continue
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        for k, v in meta.items():
            df[k] = (v == "_quant") if k == "quant" else (int(v) if k in ("pre", "post") else v)
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else None


# =========================================================
# Q1 — representation
# =========================================================

def fig_cka(df, ds, out):
    name = f"fig_cka_{ds}.pdf"
    sub = df[df["dataset"] == ds]
    series = [(c, l) for c, l in [("cka_unstructured", "IMP"), ("cka_collapsed", "Collapsed")]
              if c in sub and sub[c].notna().any()]
    if sub.empty or not series:
        warn(name, "no CKA columns with data")
        return
    models = [m for m in MODEL_ORDER if (sub["arch"] == m).any()]
    sub = sub.assign(group=sub["arch"].map({m: m for m in models}))
    fig, ax = plt.subplots()
    grouped_bars(ax, models, series, sub, f"{DS_LABELS[ds]}: CKA vs control", "linear CKA")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    save(fig, out, name)


# =========================================================
# Q2 — robustness
# =========================================================

def fig_robustness_curves(d, ds, out):
    name = f"fig_robustness_curves_{ds}.pdf"
    df = long_frames(d, "robustness_curves")
    if df is None:
        warn(name, "no robustness_curves_*.csv (needs --analysis full)")
        return
    df = df[df["dataset"] == ds]
    attacks = [a for a in ["FGSM", "PGD"] if (df["attack"] == a).any()]
    if df.empty or not attacks:
        warn(name, "no curve rows for this dataset")
        return
    fig, axes = plt.subplots(1, len(attacks), figsize=(3.5 * len(attacks), 2.4), squeeze=False)
    for ax, atk in zip(axes[0], attacks):
        sub = df[df["attack"] == atk]
        for key, label in VARIANTS:
            s = sub[sub["model"] == key]
            if s.empty:
                continue
            g = s.groupby("epsilon")["adv_acc"]
            med = g.median()
            ax.plot(med.index, med.values, marker=".", label=label, color=COLORS[label])
            ax.fill_between(med.index, g.quantile(0.25).values, g.quantile(0.75).values,
                            color=COLORS[label], alpha=0.15, linewidth=0)
        ax.set_title(f"{DS_LABELS[ds]}: {atk}")
        ax.set_xlabel("epsilon")
        ax.set_ylabel("adversarial accuracy")
        ax.margins(y=0.1)
    axes[0][0].legend(frameon=False)
    save(fig, out, name)


def fig_reliability(df, ds, out):
    name = f"fig_reliability_{ds}.pdf"
    sub = df[df["dataset"] == ds]
    panels = []
    for metric, title in [("ece", "Expected calibration error"), ("mce", "Maximum calibration error")]:
        series = [(f"{metric}_{k}", l) for k, l in VARIANTS
                  if f"{metric}_{k}" in sub and sub[f"{metric}_{k}"].notna().any()]
        panels.append((title, series))
    panels = [(t, s) for t, s in panels if s]
    if sub.empty or not panels:
        warn(name, "no ece_/mce_ columns (needs --analysis full)")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 2.4), squeeze=False)
    for ax, (title, series) in zip(axes[0], panels):
        labels = [l for _, l in series]
        sub2 = sub.assign(group=ds)
        xs = np.arange(1)
        width = 0.8 / len(series)
        for i, (col, label) in enumerate(series):
            y, lo, hi = pooled(sub2, col)
            ax.bar([i * width - 0.4 + width / 2], [y], width, yerr=[[lo], [hi]],
                   color=COLORS[label], label=label, capsize=1.5, linewidth=0)
        ax.set_xticks([])
        ax.set_title(f"{DS_LABELS[ds]}: {title}")
        ax.set_ylabel("error")
        ax.margins(y=0.15)
    axes[0][0].legend(frameon=False)
    save(fig, out, name)


def fig_ood(df, ds, out):
    name = f"fig_ood_{ds}.pdf"
    sub = df[df["dataset"] == ds]
    series = [(f"ood_auroc_{k}", l) for k, l in VARIANTS
              if f"ood_auroc_{k}" in sub and sub[f"ood_auroc_{k}"].notna().any()]
    if sub.empty or not series:
        warn(name, "no ood_auroc_ columns (needs --analysis full; CIFAR only)")
        return
    fig, ax = plt.subplots()
    sub = sub.assign(group=ds)
    xs = np.arange(1)
    width = 0.8 / len(series)
    for i, (col, label) in enumerate(series):
        y, lo, hi = pooled(sub, col)
        ax.bar([i * width - 0.4 + width / 2], [y], width, yerr=[[lo], [hi]],
               color=COLORS[label], label=label, capsize=1.5, linewidth=0)
    src = sub.get("ood_source", pd.Series(dtype=str)).dropna().unique()
    note = f" (vs {', '.join(map(str, src))})" if len(src) else ""
    ax.set_xticks([])
    ax.set_title(f"{DS_LABELS[ds]}: OOD AUROC{note}")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False)
    save(fig, out, name)


def fig_corruption(d, ds, out):
    name = f"fig_corruption_{ds}.pdf"
    df = long_frames(d, "corruption")
    if df is None:
        warn(name, "no corruption_*.csv (needs --analysis full + --corruptions-dir)")
        return
    df = df[df["dataset"] == ds]
    if df.empty:
        warn(name, "no corruption rows for this dataset")
        return
    # mean over corruptions per (config, variant, severity), then median + IQR
    per = df.groupby(CFG + ["model", "severity"], observed=True)["acc"].mean().reset_index()
    fig, ax = plt.subplots()
    for key, label in VARIANTS:
        s = per[per["model"] == key]
        if s.empty:
            continue
        g = s.groupby("severity")["acc"]
        med = g.median()
        ax.plot(med.index, med.values, marker=".", label=label, color=COLORS[label])
        ax.fill_between(med.index, g.quantile(0.25).values, g.quantile(0.75).values,
                        color=COLORS[label], alpha=0.15, linewidth=0)
    ax.set_title(f"{DS_LABELS[ds]}: mean corruption accuracy")
    ax.set_xlabel("severity")
    ax.set_ylabel("accuracy")
    ax.set_xticks(sorted(per["severity"].unique()))
    ax.margins(y=0.1)
    ax.legend(frameon=False)
    save(fig, out, name)


# =========================================================
# Q3 — quantized efficiency
# =========================================================

def fig_ptq(df, ds, out):
    name = f"fig_ptq_{ds}.pdf"
    sub = df[df["dataset"] == ds]
    fp32, fp16 = sub[~sub["quant"]], sub[sub["quant"]]
    if fp32.empty or fp16.empty:
        warn(name, "needs both fp32 and _quant CSVs for this dataset")
        return
    accs = [(k, l, acc_col(k, sub)) for k, l in VARIANTS]
    accs = [(k, l, c) for k, l, c in accs if c is not None]
    ener = [(k, l, f"j_per_img_saturated_{k}") for k, l in VARIANTS
            if f"j_per_img_saturated_{k}" in sub and sub[f"j_per_img_saturated_{k}"].notna().any()]
    panels = [("accuracy", accs), ("saturated J/img", ener)]
    panels = [(t, s) for t, s in panels if s]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 2.4), squeeze=False)
    for ax, (title, series) in zip(axes[0], panels):
        xs = np.arange(len(series))
        width = 0.38
        for j, (uni, tag, hatch) in enumerate([(fp32, "fp32", ""), (fp16, "fp16 (PTQ)", "//")]):
            ys, lo, hi = [], [], []
            for k, l, col in series:
                y, lq, hq = pooled(uni, col)
                ys.append(y), lo.append(lq), hi.append(hq)
            ax.bar(xs + j * width - width / 2, ys, width, yerr=[lo, hi],
                   label=tag, color="#9ecae1" if j else "#08306b", hatch=hatch,
                   capsize=1.5, linewidth=0)
        ax.set_xticks(xs)
        ax.set_xticklabels([l for _, l, _ in series], rotation=15, ha="right")
        ax.set_title(f"{DS_LABELS[ds]}: {title}")
        ax.margins(y=0.15)
    axes[0][0].legend(frameon=False)
    save(fig, out, name)


# =========================================================
# Q4 — resource savings
# =========================================================

def fig_energy_transfer(df, out):
    name = "fig_energy_transfer.pdf"
    fig, ax = plt.subplots()
    drew = False
    for key, label in [("collapsed", "Collapsed"), ("unstructured", "IMP")]:
        f, e = f"flops_{key}", f"energy_joules_{key}"
        if not all(c in df for c in (f, e, "flops_original", "energy_joules_original")):
            continue
        per = df.groupby(CFG, observed=True)[[f, e, "flops_original", "energy_joules_original"]].median()
        fr = per[f] / per["flops_original"].replace(0, np.nan)
        er = per[e] / per["energy_joules_original"].replace(0, np.nan)
        ok = fr.notna() & er.notna()
        if not ok.any():
            continue
        ax.scatter(1 - fr[ok], 1 - er[ok], s=12, alpha=0.7,
                   color=COLORS[label], label=label, linewidths=0)
        drew = True
    if not drew:
        warn(name, "no flops/energy pairs")
        plt.close(fig)
        return
    ax.plot([0, 1], [0, 1], color="#999999", linewidth=0.8, linestyle="--")
    ax.set_xlabel("FLOPs removed (fraction of control)")
    ax.set_ylabel("joules removed (fraction of control)")
    ax.set_title("Energy transferability")
    ax.legend(frameon=False)
    save(fig, out, name)


def fig_regime(df, out):
    """Regime map: does removed compute become removed joules/latency?"""
    name = "fig_regime.pdf"
    need = ["flops_collapsed", "flops_original", "j_per_img_saturated_collapsed",
            "j_per_img_saturated_original", "latency_ms_collapsed", "latency_ms_original"]
    if not all(c in df for c in need):
        warn(name, "missing flops/energy/latency columns")
        return
    per = df.groupby(CFG, observed=True)[need].median()
    fr = 1 - per["flops_collapsed"] / per["flops_original"].replace(0, np.nan)
    jr = per["j_per_img_saturated_collapsed"] / per["j_per_img_saturated_original"].replace(0, np.nan)
    lr = per["latency_ms_collapsed"] / per["latency_ms_original"].replace(0, np.nan)
    ok = fr.notna() & (jr.notna() | lr.notna())
    if not ok.any():
        warn(name, "no complete flops/energy/latency triples")
        return
    # overhead-dominated: FLOPs clearly reduced but the measured ratio did not follow
    over = ok & (fr >= 0.3) & ((jr >= 1.0) | (lr >= 1.0))
    fig, ax = plt.subplots()
    for series, label, marker in [(jr, "saturated J/img", "o"), (lr, "latency", "s")]:
        good = ok & series.notna() & ~over
        bad = ok & series.notna() & over
        ax.scatter(fr[good], series[good], s=16, marker=marker, alpha=0.75,
                   color="#1f77b4", linewidths=0)
        ax.scatter(fr[bad], series[bad], s=22, marker=marker, alpha=0.9,
                   facecolors="none", edgecolors="#d62728", linewidths=1.1)
    for arch, row in per[ok].iterrows():
        ax.annotate(str(arch[0]).replace("_400MF", ""), (fr.loc[arch], jr.loc[arch]),
                    fontsize=5, alpha=0.6, xytext=(2, 2), textcoords="offset points")
    ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle="--")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color="#1f77b4", label="saturated J/img"),
               plt.Line2D([], [], marker="s", linestyle="", color="#1f77b4", label="latency"),
               plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                          markeredgecolor="#d62728", label="overhead-dominated")]
    ax.legend(handles=handles, frameon=False)
    ax.set_xlabel("FLOPs removed (fraction of control)")
    ax.set_ylabel("measured ratio vs control")
    ax.set_title("Regime map: counted vs measured savings")
    ax.margins(x=0.08, y=0.1)
    save(fig, out, name)


def fig_efficiency(df, ds, out):
    name = f"fig_efficiency_{ds}.pdf"
    sub = df[df["dataset"] == ds]
    families = [("latency_ms", "Latency"), ("throughput_imgs", "Throughput"),
                ("flops", "FLOPs"), ("j_per_img_saturated", "J/img (sat.)"),
                ("storage_mb", "Storage"), ("feature_maps_mb", "Feat. maps"),
                ("peak_mem_mb", "Peak mem")]
    key2label = dict(VARIANTS)
    present = []  # (family_label, [(ratio_col, variant_key)]) — ratios assigned in one shot
    new_cols = {}
    for fam, flabel in families:
        cols = []
        for key, _ in VARIANTS[1:]:
            a, b = f"{fam}_{key}", f"{fam}_original"
            if a not in sub or b not in sub:
                continue
            rc = f"ratio_{fam}_{key}"
            new_cols[rc] = (pd.to_numeric(sub[a], errors="coerce")
                            / pd.to_numeric(sub[b], errors="coerce").replace(0, np.nan))
            cols.append((rc, key))
        if cols:
            present.append((flabel, cols))
    if new_cols:
        sub = pd.concat([sub, pd.DataFrame(new_cols, index=sub.index)], axis=1)
    present = [(flabel, [(rc, k) for rc, k in cols if sub[rc].notna().any()])
               for flabel, cols in present]
    present = [(flabel, cols) for flabel, cols in present if cols]
    if sub.empty or not present:
        warn(name, "no hardware metric columns with data")
        return
    fig, ax = plt.subplots()
    xs = np.arange(len(present))
    width = 0.8 / 3
    for gi, (flabel, cols) in enumerate(present):
        for bi, (rc, key) in enumerate(cols):
            y, lo, hi = pooled(sub, rc)
            ax.bar(gi + bi * width - 0.4 + width / 2, y, width,
                   yerr=[[lo], [hi]], color=COLORS[key2label[key]],
                   label=key2label[key] if gi == 0 else None,
                   capsize=1.5, linewidth=0)
    ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([f for f, _ in present], rotation=20, ha="right")
    ax.set_ylabel("ratio vs control")
    ax.set_title(f"{DS_LABELS[ds]}: hardware efficiency")
    ax.legend(frameon=False)
    ax.margins(y=0.12)
    save(fig, out, name)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(d, "dissertation_figs")
    os.makedirs(out, exist_ok=True)
    df = load_rows(d)
    if df is None:
        print(f"[WARN] no cka_results_*.csv rows in {d} — nothing to plot")
        return
    datasets = [ds for ds in DS_LABELS if (df["dataset"] == ds).any()]
    for ds in datasets:
        fig_cka(df, ds, out)
        fig_robustness_curves(d, ds, out)
        fig_reliability(df, ds, out)
        fig_ood(df, ds, out)
        fig_corruption(d, ds, out)
        fig_ptq(df, ds, out)
        fig_efficiency(df, ds, out)
    fig_energy_transfer(df, out)
    fig_regime(df, out)


if __name__ == "__main__":
    main()
