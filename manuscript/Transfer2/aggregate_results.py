"""Aggregate all cka_results_*.csv into comparison-ready tables.

Cross-GPU apples-to-apples policy: raw latency/energy mean nothing across GPU
types, so the key numbers are *within-row ratios* vs the `original` variant —
every variant in a row was measured in the same session on the same GPU, so
session noise (clocks, I/O) cancels. Absolute numbers are kept per (config,
gpu) row for single-GPU reporting.

Outputs:
  aggregated_rows.csv   all rows from all CSVs (column-unioned), CPU rows
                        dropped where a GPU twin exists, ratio columns added
  aggregated_median.csv per-(config, quant) median over GPUs, plus per-GPU count

Usage: python aggregate_results.py [dir]   (dir defaults to .)
"""
import glob
import os
import sys

import pandas as pd

RATIOS = [  # (numerator_suffix, denominator_suffix, new_col)
    ("collapsed", "original"),
    ("unstructured", "original"),
]

def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(d, "cka_results_*.csv")))
    frames = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue  # 1-byte placeholder files from not-yet-run configs
        if df.empty:
            continue
        df["source"] = os.path.basename(f)
        df["quant"] = "_quant" in stem
        frames.append(df)
    if not frames:
        print("no data rows found")
        return
    df = pd.concat(frames, ignore_index=True)

    keys = ["model", "dataset", "pre_epochs", "post_epochs", "quant"]

    # drop CPU rows when a GPU row exists for the same config
    is_cpu = df["gpu_name"].astype(str).str.upper().eq("CPU")
    keys_with_gpu = set(map(tuple, df.loc[~is_cpu, keys].drop_duplicates().values))
    drop = is_cpu & df[keys].apply(tuple, axis=1).isin(keys_with_gpu)
    n_cpu = int(drop.sum())
    df = df[~drop]
    print(f"{len(frames)} csvs, {len(df)} rows kept; dropped {n_cpu} CPU shadow rows")

    # within-row ratios vs the original variant
    for num, den in RATIOS:
        for fam, col in [("energy", "energy_joules"),
                         ("latency", "latency_ms"), ("throughput", "throughput_imgs"),
                         ("storage", "storage_mb"), ("flops", "flops")]:
            a, b = f"{col}_{num}", f"{col}_{den}"
            if a in df and b in df:
                df[f"{fam}_ratio_{num}_vs_original"] = (
                    pd.to_numeric(df[a], errors="coerce") / pd.to_numeric(df[b], errors="coerce"))
        if f"acc_{num}" in df and "acc_original_after" in df:
            df[f"acc_ratio_{num}_vs_original"] = (
                pd.to_numeric(df[f"acc_{num}"], errors="coerce")
                / pd.to_numeric(df["acc_original_after"], errors="coerce"))

    # energy per image (CIFAR-10/100 and TinyImageNet test sets are all 10k)
    for v in ["original", "unstructured", "collapsed"]:
        c = f"energy_joules_{v}"
        if c in df:
            df[f"j_per_img_{v}"] = pd.to_numeric(df[c], errors="coerce") / 10000.0

    df.to_csv(os.path.join(d, "aggregated_rows.csv"), index=False)

    grp = df.groupby(keys, observed=True)
    med = grp.median(numeric_only=True).reset_index()
    med["n_rows"] = grp.size().values
    med["gpus"] = grp["gpu_name"].agg(lambda s: ",".join(sorted(set(map(str, s))))).values
    med.to_csv(os.path.join(d, "aggregated_median.csv"), index=False)

    # quick terminal summary of the headline ratios
    cols = [c for c in ("acc_ratio_collapsed_vs_original", "latency_ratio_collapsed_vs_original",
                        "energy_ratio_collapsed_vs_original", "storage_ratio_collapsed_vs_original") if c in med]
    show = med[keys + cols + ["n_rows"]].sort_values(["model", "dataset", "pre_epochs"], na_position="last")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {os.path.join(d, 'aggregated_rows.csv')} and {os.path.join(d, 'aggregated_median.csv')}")

if __name__ == "__main__":
    main()
