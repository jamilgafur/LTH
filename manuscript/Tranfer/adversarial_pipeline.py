"""Orchestration entrypoint for modular adversarial analysis pipeline."""

from __future__ import annotations

import argparse
import os

import pandas as pd

from adversarial_compute_tradeoff import ComputeTradeoffSuite
from adversarial_checkpointing import CheckpointManager
from adversarial_core import AdversarialCore
from adversarial_correlations import CorrelationSuite
from adversarial_experiments import AdvancedExperimentSuite
from adversarial_plotting import AdversarialPlotSuite
from adversarial_reporting import ReportingSuite


def _save_summary_records(args, output_dir: str, records: list[dict]) -> None:
    if not records:
        return
    df_records = pd.DataFrame(records)
    if args.model and args.attack:
        job_label = f"_{args.model}_{args.dataset}_{args.attack}_{args.kind or 'ALL'}"
        csv_path = os.path.join(output_dir, f"summary{job_label}.csv")
    else:
        csv_path = os.path.join(output_dir, "summary.csv")
    df_records.to_csv(csv_path, index=False)
    print(f"[INFO] Summary written to {csv_path}")


def _load_records_from_csv(output_dir: str) -> list[dict]:
    summary_path = os.path.join(output_dir, "summary.csv")
    if not os.path.exists(summary_path):
        return []
    return pd.read_csv(summary_path).to_dict(orient="records")


def _find_precomputed_adversarial_files(args, output_dir: str, checkpoints: list[tuple]) -> dict:
    adv_datasets = {}
    available_attacks = AdversarialCore.get_available_attacks()
    for model_name, dataset_name, kind, _ckpt_path in checkpoints:
        for attack in available_attacks:
            if args.model and model_name != args.model:
                continue
            if args.dataset:
                base_check = CheckpointManager.base_dataset_name(dataset_name)
                if base_check != args.dataset:
                    continue
            if args.kind and kind != args.kind:
                continue
            if args.attack and attack != args.attack:
                continue
            adv_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_{kind}_{attack}_adv.pt")
            if os.path.exists(adv_path):
                adv_datasets[(model_name, dataset_name, kind, attack)] = adv_path
    return adv_datasets


def _build_experiment_suite() -> AdvancedExperimentSuite:
    return AdvancedExperimentSuite(
        instantiate_attack=AdversarialCore.instantiate_attack,
        model_kind_label=ReportingSuite.model_kind_label,
        classify_transfer_pair=ReportingSuite.classify_transfer_pair,
    )


def run_pipeline(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    records = []
    transfer_records = []
    model_cache = {}
    loader_cache = {}
    adv_datasets = {}
    experiment_suite = _build_experiment_suite()

    if args.mode in ["full", "generate"]:
        print(f"\n{'='*70}\n[PHASE 1] Generating adversarial examples\n{'='*70}")
        records, model_cache, loader_cache, adv_datasets = AdversarialCore.generate_attacks_phase(
            args.output_dir,
            args.model,
            args.dataset,
            args.attack,
            args.kind,
        )
        _save_summary_records(args, args.output_dir, records)

    if args.mode in ["full", "analyze"]:
        print(f"\n{'='*70}\n[PHASE 2] Analyzing transferability\n{'='*70}")
        ReportingSuite.merge_parallel_csvs(args.output_dir)

        if args.mode == "analyze":
            records = _load_records_from_csv(args.output_dir)
            if not records:
                print("[ERROR] summary.csv not found. Run with --mode full or generate first.")
                return

            model_cache, loader_cache = AdversarialCore.rebuild_model_and_loader_cache(args)
            checkpoints = CheckpointManager.discover_checkpoints()
            adv_datasets = _find_precomputed_adversarial_files(args, args.output_dir, checkpoints)

        transfer_records = AdversarialCore.analyze_transferability_phase(args.output_dir, model_cache, adv_datasets)
        if transfer_records:
            tf_csv = os.path.join(args.output_dir, "transferability.csv")
            pd.DataFrame(transfer_records).to_csv(tf_csv, index=False)
            print(f"[INFO] Transferability matrix saved to {tf_csv}")

    if args.mode in ["full", "plot"]:
        print(f"\n{'='*70}\n[PHASE 3] Generating plots\n{'='*70}")
        if args.mode == "plot":
            ReportingSuite.merge_parallel_csvs(args.output_dir)
            records = _load_records_from_csv(args.output_dir)
            transfer_path = os.path.join(args.output_dir, "transferability.csv")
            transfer_records = pd.read_csv(transfer_path).to_dict(orient="records") if os.path.exists(transfer_path) else []

        AdversarialPlotSuite.generate_plots(args.output_dir, records, transfer_records)
        print(f"[INFO] All plots saved to {args.output_dir}")

    if args.mode in ["full", "analyze", "plot", "compare"]:
        print(f"\n{'='*70}\n[PHASE 4] Comparing original vs collapsed explainability\n{'='*70}")
        if records:
            ReportingSuite.generate_comparison_tables(args.output_dir, records)
        else:
            ReportingSuite.generate_comparison_tables_from_csv(args.output_dir)

    if args.mode in ["full", "gradient_sim"]:
        print(f"\n{'='*70}\n[PHASE EXP4] Gradient Similarity Analysis\n{'='*70}")
        if args.mode == "gradient_sim" or not model_cache:
            model_cache, loader_cache = AdversarialCore.rebuild_model_and_loader_cache(args)
        experiment_suite.gradient_similarity_phase(args.output_dir, model_cache, loader_cache)

    if args.mode in ["full", "epsilon_sweep"]:
        print(f"\n{'='*70}\n[PHASE EXP7] Epsilon Sensitivity Analysis\n{'='*70}")
        if args.mode == "epsilon_sweep" or not model_cache:
            model_cache, loader_cache = AdversarialCore.rebuild_model_and_loader_cache(args)
        experiment_suite.epsilon_sensitivity_phase(args.output_dir, model_cache, loader_cache, attacks=args.epsilon_attacks)

    if args.mode in ["full", "statistics"]:
        print(f"\n{'='*70}\n[PHASE EXP9] Statistical Significance Testing\n{'='*70}")
        result_dirs = args.result_dirs or [args.output_dir]
        experiment_suite.statistical_significance_phase(args.output_dir, result_dirs=result_dirs)

    if args.mode in ["full", "cka"]:
        print(f"\n{'='*70}\n[PHASE EXP10] CKA Feature Similarity Analysis\n{'='*70}")
        if args.mode == "cka" or not model_cache:
            model_cache, loader_cache = AdversarialCore.rebuild_model_and_loader_cache(args)
        experiment_suite.cka_similarity_phase(
            args.output_dir,
            model_cache,
            loader_cache,
            max_samples=args.cka_max_samples,
            max_layers=args.cka_max_layers,
        )

    if args.mode in ["full", "compute_tradeoff"]:
        print(f"\n{'='*70}\n[PHASE COST] Compute-Cost Tradeoff\n{'='*70}")
        if not model_cache:
            model_cache, loader_cache = AdversarialCore.rebuild_model_and_loader_cache(args)
        ComputeTradeoffSuite.run(args.output_dir, model_cache, loader_cache)

    if args.mode in ["full", "correlations"]:
        print(f"\n{'='*70}\n[PHASE CORR] Correlation Analysis\n{'='*70}")
        CorrelationSuite.run(args.output_dir)

    if args.mode in ["full", "explainability"]:
        print(f"\n{'='*70}\n[PHASE EXP11] SHAP Explainability Similarity\n{'='*70}")
        if args.mode == "explainability" or not model_cache:
            model_cache, loader_cache = AdversarialCore.rebuild_model_and_loader_cache(args)
        experiment_suite.explainability_similarity_phase(
            args.output_dir,
            model_cache,
            loader_cache,
            max_samples=args.shap_max_samples,
            background_samples=args.shap_background_samples,
            topk_ratio=args.shap_topk_ratio,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adversarial robustness analysis of pruned models.")
    parser.add_argument(
        "--mode",
        choices=[
            "full",
            "generate",
            "analyze",
            "plot",
            "compare",
            "gradient_sim",
            "epsilon_sweep",
            "statistics",
            "cka",
            "compute_tradeoff",
            "correlations",
            "explainability",
        ],
        default="full",
        help=(
            "Execution mode: full, generate, analyze, plot, compare, gradient_sim, "
            "epsilon_sweep, statistics, cka, compute_tradeoff, correlations, explainability."
        ),
    )
    parser.add_argument("--model", type=str, default=None, help="Filter by model name (e.g., VGG16).")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset (e.g., Cifar10).")
    parser.add_argument("--attack", type=str, default=None, help="Filter by attack (e.g., PGD).")
    parser.add_argument("--kind", choices=["Original", "Finetuned"], default=None)
    parser.add_argument("--output-dir", type=str, default="adversarial_results", help="Output directory for results.")
    parser.add_argument("--epsilon-attacks", type=str, nargs="+", default=["PGD", "FGSM", "BIM"])
    parser.add_argument("--result-dirs", type=str, nargs="+", default=None)
    parser.add_argument("--cka-max-samples", type=int, default=512)
    parser.add_argument("--cka-max-layers", type=int, default=8)
    parser.add_argument("--shap-max-samples", type=int, default=64)
    parser.add_argument("--shap-background-samples", type=int, default=32)
    parser.add_argument("--shap-topk-ratio", type=float, default=0.05)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)
