"""Analysis entrypoints for adversarial robustness and transferability."""

from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd

from adversarial_core import AdversarialCore
from adversarial_experiments import AdvancedExperimentSuite
from adversarial_reporting import ReportingSuite


def compute_transfer_metrics(output_dir: str, model_cache: dict, adv_datasets: dict) -> list[dict]:
    """Compute transferability metrics and persist them to transferability.csv."""
    records = AdversarialCore.analyze_transferability_phase(output_dir, model_cache, adv_datasets)
    if records:
        transfer_df = pd.DataFrame(records)
        transfer_df.to_csv(os.path.join(output_dir, "transferability.csv"), index=False)
    return records


def compute_shap_explainability(
    output_dir: str,
    model_cache: dict | None = None,
    loader_cache: dict | None = None,
    max_samples: int = 64,
    background_samples: int = 32,
    topk_ratio: float = 0.05,
) -> list[dict]:
    """Compute SHAP reference vectors and pairwise similarity metrics.

    If no model_cache or loader_cache is given, build them from the checkpoint set for the
    selected output directory and filters.
    """
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
    summary_path = os.path.join(args.output_dir, "summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing summary.csv in {args.output_dir}")

    # Reuse the model cache builder and the recorded adversarial datasets when available.
    model_cache = {}
    adv_datasets = {}
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
    else:
        print(f"[INFO] Transfer analysis ready for {args.output_dir}")
        print("[INFO] Nothing additional to compute; summary and generated adversarial artifacts are the data source.")


if __name__ == "__main__":
    main()
