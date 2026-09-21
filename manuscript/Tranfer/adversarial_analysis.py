"""Analysis entrypoints for adversarial robustness and transferability."""

from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd

from adversarial_core import AdversarialCore
from adversarial_experiments import AdvancedExperimentSuite
from adversarial_reporting import ReportingSuite

VALID_PHASES = [
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
]


def check_summary_ready(output_dir: str) -> tuple[bool, str]:
    """Return whether a summary.csv exists and is non-empty."""
    summary_path = os.path.join(output_dir, "summary.csv")
    if not os.path.exists(summary_path):
        return False, summary_path
    try:
        df = pd.read_csv(summary_path)
        return not df.empty, summary_path
    except Exception:
        return False, summary_path


def compute_transfer_metrics(output_dir: str, model_cache: dict, adv_datasets: dict) -> list[dict]:
    """Compute transferability metrics and persist them to transferability.csv."""
    records = AdversarialCore.analyze_transferability_phase(output_dir, model_cache, adv_datasets)
    if records:
        transfer_df = pd.DataFrame(records)
        transfer_df.to_csv(os.path.join(output_dir, "transferability.csv"), index=False)
    return records


def run_generate_phase(
    output_dir: str,
    model_filter: str | None = None,
    dataset_filter: str | None = None,
    attack_filter: str | None = None,
    kind_filter: str | None = None,
) -> list[dict]:
    """Run the attack-generation phase and persist summary.csv when records are available."""
    records, model_cache, loader_cache, adv_datasets = AdversarialCore.generate_attacks_phase(
        output_dir=output_dir,
        model_filter=model_filter,
        dataset_filter=dataset_filter,
        attack_filter=attack_filter,
        kind_filter=kind_filter,
    )

    if records:
        summary_df = pd.DataFrame(records)
        summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
        print(f"[INFO] Generated {len(records)} summary rows in {output_dir}/summary.csv")
    else:
        print(f"[WARN] No attack-generation records were produced for {output_dir}")
    return records


def compute_shap_explainability(
    output_dir: str,
    model_cache: dict | None = None,
    loader_cache: dict | None = None,
    max_samples: int = 64,
    background_samples: int = 32,
    topk_ratio: float = 0.05,
) -> list[dict]:
    """Compute SHAP reference vectors and pairwise similarity metrics."""
    if model_cache is None or loader_cache is None:
        model_cache, loader_cache = AdversarialCore.rebuild_model_and_loader_cache(
            argparse.Namespace(output_dir=output_dir, model=None, dataset=None, kind=None)
        )

    suite = AdvancedExperimentSuite(
        instantiate_attack=lambda *args, **kwargs: None,
        model_kind_label=ReportingSuite.model_kind_label,
        classify_transfer_pair=ReportingSuite.classify_transfer_pair,
    )
    return suite.explainability_similarity_phase(
        output_dir=output_dir,
        model_cache=model_cache,
        loader_cache=loader_cache,
        max_samples=max_samples,
        background_samples=background_samples,
        topk_ratio=topk_ratio,
    )


def compute_shap_for_results(
    output_dir: str,
    model_filter: str | None = None,
    dataset_filter: str | None = None,
    kind_filter: str | None = None,
    max_samples: int = 64,
    background_samples: int = 32,
    topk_ratio: float = 0.05,
) -> list[dict]:
    """Build the model and loader cache for the selected output dir and run SHAP analysis."""
    args = argparse.Namespace(
        output_dir=output_dir,
        model=model_filter,
        dataset=dataset_filter,
        kind=kind_filter,
    )
    model_cache, loader_cache = AdversarialCore.rebuild_model_and_loader_cache(args)
    return compute_shap_explainability(
        output_dir,
        model_cache=model_cache,
        loader_cache=loader_cache,
        max_samples=max_samples,
        background_samples=background_samples,
        topk_ratio=topk_ratio,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial transfer analysis")
    parser.add_argument("--mode", default="analyze", choices=VALID_PHASES + ["full"], help="Pipeline phase to run")
    parser.add_argument("--output-dir", default="adversarial_results", help="Directory containing summary CSVs")
    parser.add_argument("--model", default=None, help="Optional model filter")
    parser.add_argument("--dataset", default=None, help="Optional dataset filter")
    parser.add_argument("--kind", default=None, help="Optional kind filter")
    parser.add_argument("--attack", default=None, help="Optional attack filter")
    parser.add_argument("--compute-shap", action="store_true", help="Compute SHAP explainability metrics for the selected run")
    parser.add_argument("--max-samples", type=int, default=64, help="Max number of samples used to build SHAP reference vectors")
    parser.add_argument("--background-samples", type=int, default=32, help="Number of background samples used by SHAP explainer")
    parser.add_argument("--topk-ratio", type=float, default=0.05, help="Top-k ratio used in SHAP feature vector comparisons")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "generate":
        print(f"[INFO] Running generate phase for {args.output_dir}")
        run_generate_phase(
            output_dir=args.output_dir,
            model_filter=args.model,
            dataset_filter=args.dataset,
            attack_filter=args.attack,
            kind_filter=args.kind,
        )
        return

    ready, summary_path = check_summary_ready(args.output_dir)
    if not ready:
        print(f"[WARN] Missing or empty summary.csv in {args.output_dir}. Run the generate phase first.")
        return

    if args.mode == "analyze":
        print(f"[INFO] Running analyze phase for {args.output_dir}")
        model_cache, loader_cache = AdversarialCore.rebuild_model_and_loader_cache(
            argparse.Namespace(output_dir=args.output_dir, model=args.model, dataset=args.dataset, kind=args.kind)
        )
        adv_datasets = {}
        for model_name, dataset_name, kind, attack_name in AdversarialCore.get_available_attacks():
            pass
        compute_transfer_metrics(args.output_dir, model_cache, adv_datasets)
        return

    if args.mode == "plot":
        print(f"[INFO] Plot phase will proceed using {summary_path}")
        return

    if args.mode == "compare":
        print(f"[INFO] Compare phase will proceed using {summary_path}")
        return

    if args.compute_shap:
        print(f"[INFO] Computing SHAP explainability for {args.output_dir}")
        pairs = compute_shap_for_results(
            output_dir=args.output_dir,
            model_filter=args.model,
            dataset_filter=args.dataset,
            kind_filter=args.kind,
            max_samples=args.max_samples,
            background_samples=args.background_samples,
            topk_ratio=args.topk_ratio,
        )
        print(f"[INFO] SHAP analysis produced {len(pairs)} pairwise records.")
        return

    print(f"[INFO] {args.mode} phase is ready for {args.output_dir}. Summary is available at {summary_path}.")


if __name__ == "__main__":
    main()
