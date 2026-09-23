"""Analysis entrypoints for adversarial robustness and transferability."""

from __future__ import annotations

import argparse
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

# Import sibling modules using relative imports to work correctly when the
# package is executed with ``python -m manuscript.Tranfer.adversarial_analysis``.
# Absolute imports (e.g., ``from adversarial_core import ...``) fail because the
# top‑level module name ``adversarial_core`` is not found on ``sys.path``.
from .adversarial_core import AdversarialCore
from .adversarial_correlations import CorrelationSuite
from .adversarial_compute_tradeoff import TradeoffComputer
from .adversarial_experiments import AdvancedExperimentSuite
from .adversarial_reporting import ReportingSuite

VALID_PHASES = [
    "generate",
    "analyze",
    "plot",
    "compare",
    "gradient_sim",
    "epsilon_sweep",
    "statistics",
    "cka",
    "compute_shap",
    "compute_tradeoff",
    "correlations",
]


def _log_phase_context(args: argparse.Namespace) -> None:
    """Emit a concise phase banner so batch-job logs show the actual dispatch context."""
    print(
        "[INFO] Dispatching phase="
        f"{args.mode} output_dir={args.output_dir} "
        f"model={args.model or 'ALL'} dataset={args.dataset or 'ALL'} "
        f"attack={args.attack or 'ALL'} kind={args.kind or 'ALL'}"
    )


def _atomic_write_csv(path: str, df: pd.DataFrame) -> None:
    """Write CSV atomically to avoid partial writes from concurrent jobs."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=str(path_obj.parent))
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path_obj)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _acquire_lock(lock_path: str, timeout_seconds: float = 60.0, stale_seconds: float = 300.0) -> bool:
    """Acquire an exclusive lock via a lock file for shared result directories.

    If a lock file already exists, its modification time is inspected. If the
    lock is older than ``stale_seconds`` it is considered stale and removed
    before retrying. This prevents dead‑locks caused by crashed jobs.
    """
    lock_dir = Path(lock_path).parent
    lock_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            print(f"[DEBUG] Acquired lock {lock_path}")
            return True
        except FileExistsError:
            # Check for stale lock
            try:
                mtime = os.path.getmtime(lock_path)
                if (time.time() - mtime) > stale_seconds:
                    os.unlink(lock_path)
                    print(f"[DEBUG] Removed stale lock {lock_path}")
                    continue
            except OSError:
                pass
            time.sleep(0.2)
    print(f"[WARN] Timeout acquiring lock {lock_path}")
    return False


def _release_lock(lock_path: str) -> None:
    try:
        if os.path.exists(lock_path):
            os.unlink(lock_path)
    except OSError:
        pass


def _merge_summary_if_ready(output_dir: str) -> str | None:
    """Merge summary shards into canonical ``summary.csv`` when all shards are present.

    The function acquires a lock, concatenates all ``summary_*.csv`` shards, writes
    the merged CSV atomically, and creates a ``.summary_merge_complete`` flag file.
    Debug statements are emitted to aid troubleshooting.
    """
    summary_path = os.path.join(output_dir, "summary.csv")
    shard_paths = (
        sorted(
            [os.path.join(output_dir, p) for p in os.listdir(output_dir) if p.startswith("summary_") and p.endswith(".csv")]
        )
        if os.path.isdir(output_dir)
        else []
    )
    if not shard_paths:
        # No shards – either a pre‑existing summary or nothing yet.
        return summary_path if os.path.exists(summary_path) else None

    lock_path = os.path.join(output_dir, ".summary_merge.lock")
    if not _acquire_lock(lock_path, timeout_seconds=120.0):
        print(f"[WARN] Could not acquire summary merge lock in {output_dir}; skipping merge.")
        return summary_path if os.path.exists(summary_path) else None
    print(f"[DEBUG] Acquired merge lock {lock_path}")
    try:
        # Read all existing shard CSVs, ignoring empty or missing files.
        summary_df = pd.concat(
            [pd.read_csv(p) for p in shard_paths if os.path.exists(p) and os.path.getsize(p) > 0],
            ignore_index=True,
        )
        if not summary_df.empty:
            summary_df = summary_df.drop_duplicates().reset_index(drop=True)
            _atomic_write_csv(summary_path, summary_df)
            print(f"[INFO] Merged {len(shard_paths)} summary shards into {summary_path}")
            # Write completion flag so downstream phases know merge succeeded.
            flag_path = os.path.join(output_dir, ".summary_merge_complete")
            try:
                Path(flag_path).touch()
                print(f"[DEBUG] Created merge‑completion flag {flag_path}")
            except OSError as e:
                print(f"[WARN] Failed to create merge‑completion flag {flag_path}: {e}")
    finally:
        _release_lock(lock_path)
        print(f"[DEBUG] Released merge lock {lock_path}")
    return summary_path if os.path.exists(summary_path) else None


def check_summary_ready(output_dir: str) -> tuple[bool, str]:
    """Return whether a summary.csv exists and is non-empty.

    This function first checks for the canonical summary.csv, then falls back to
    shard files and merges them atomically to make the downstream phases safe under
    parallel generate jobs.
    """
    summary_path = os.path.join(output_dir, "summary.csv")
    if not os.path.exists(summary_path):
        merged_path = _merge_summary_if_ready(output_dir)
        if merged_path is not None:
            summary_path = merged_path

    if not os.path.exists(summary_path):
        return False, summary_path
    try:
        df = pd.read_csv(summary_path)
        return not df.empty, summary_path
    except Exception:
        return False, summary_path


def compute_transfer_metrics(
    output_dir: str,
    model_cache: dict,
    adv_datasets: dict,
    transferability_output: str | None = None,
) -> int:
    """Compute transferability metrics and persist them to transferability.csv."""
    records = AdversarialCore.analyze_transferability_phase(
        output_dir,
        model_cache,
        adv_datasets,
        transferability_output=transferability_output,
    )
    return records


def _merge_transferability_if_ready(output_dir: str) -> str | None:
    """Merge transferability shards into canonical ``transferability.csv``.

    Parallel analyze jobs write dataset-specific shards. This helper collects the
    shards, deduplicates rows, and writes the canonical CSV atomically under a
    lock so downstream phases see an all-or-nothing result.
    """
    canonical_path = os.path.join(output_dir, "transferability.csv")
    shard_paths: list[str] = []
    if os.path.isdir(output_dir):
        for name in sorted(os.listdir(output_dir)):
            if not name.startswith("transferability_") or not name.endswith(".csv"):
                continue
            if name == "transferability.csv":
                continue
            shard_paths.append(os.path.join(output_dir, name))

    if not shard_paths:
        return canonical_path if os.path.exists(canonical_path) else None

    lock_path = os.path.join(output_dir, ".transferability_merge.lock")
    if not _acquire_lock(lock_path, timeout_seconds=120.0):
        print(f"[WARN] Could not acquire transferability merge lock in {output_dir}; skipping merge.")
        return canonical_path if os.path.exists(canonical_path) else None

    print(f"[DEBUG] Acquired transferability merge lock {lock_path}")
    try:
        frames = []
        for path in shard_paths:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                continue
            frames.append(pd.read_csv(path))

        if not frames:
            print(f"[WARN] No non-empty transferability shards found in {output_dir}")
            return canonical_path if os.path.exists(canonical_path) else None

        merged_df = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
        _atomic_write_csv(canonical_path, merged_df)
        print(f"[INFO] Merged {len(frames)} transferability shards into {canonical_path}")
        return canonical_path
    finally:
        _release_lock(lock_path)
        print(f"[DEBUG] Released transferability merge lock {lock_path}")


def _persist_rebuilt_summary_records(output_dir: str, records: list[dict]) -> str | None:
    """Persist rebuilt summary rows as a shard and merge them into summary.csv."""
    if not records:
        return None

    shard_path = os.path.join(output_dir, "summary_rebuilt_from_artifacts.csv")
    _atomic_write_csv(shard_path, pd.DataFrame(records))
    print(f"[INFO] Wrote rebuilt summary shard to {shard_path}")

    merge_flag = os.path.join(output_dir, ".summary_merge_complete")
    if os.path.exists(merge_flag):
        try:
            os.remove(merge_flag)
        except OSError:
            pass

    return _merge_summary_if_ready(output_dir)


def run_generate_phase(
    output_dir: str,
    model_filter: str | None = None,
    dataset_filter: str | None = None,
    attack_filter: str | None = None,
    kind_filter: str | None = None,
) -> list[dict]:
    """Run the attack-generation phase and persist a per-job summary shard.

    Multiple generate jobs can execute in parallel against the same result directory.
    To avoid races on summary.csv, each job writes a unique shard file instead of
    writing the top-level summary file directly. Downstream phases merge shards before
    reading the canonical summary.
    """
    records, model_cache, loader_cache, adv_datasets = AdversarialCore.generate_attacks_phase(
        output_dir=output_dir,
        model_filter=model_filter,
        dataset_filter=dataset_filter,
        attack_filter=attack_filter,
        kind_filter=kind_filter,
    )

    if records:
        parts = [
            model_filter or "ALL",
            dataset_filter or "ALL",
            attack_filter or "ALL",
            kind_filter or "ALL",
        ]
        shard_name = "_".join(part for part in parts if part and part != "ALL") or f"job_{os.getpid()}"
        shard_path = os.path.join(output_dir, f"summary_{shard_name}.csv")
        _atomic_write_csv(shard_path, pd.DataFrame(records))
        print(f"[INFO] Generated {len(records)} summary rows in {shard_path}")

        merge_flag = os.path.join(output_dir, ".summary_merge_complete")
        if os.path.exists(merge_flag):
            try:
                os.remove(merge_flag)
            except OSError:
                pass
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


def compute_tradeoff_for_results(output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the per-run compute profile and tradeoff summary from the canonical summary."""
    summary_path = os.path.join(output_dir, "summary.csv")
    summary_df = pd.read_csv(summary_path)
    summary_df = ReportingSuite.enrich_summary_dataframe(summary_df)

    computer = TradeoffComputer(base_dir=output_dir, output_dir=output_dir)
    compute_df = computer.compute_compute_profile(summary_df)
    tradeoff_df = computer.compute_tradeoff_summary(summary_df)
    return compute_df, tradeoff_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial transfer analysis")
    parser.add_argument("--mode", default="analyze", choices=VALID_PHASES + ["full"], help="Pipeline phase to run")
    parser.add_argument("--output-dir", default="adversarial_results", help="Directory containing summary CSVs")
    parser.add_argument("--model", default=None, help="Optional model filter")
    parser.add_argument("--dataset", default=None, help="Optional dataset filter")
    parser.add_argument("--kind", default=None, help="Optional kind filter")
    parser.add_argument("--attack", default=None, help="Optional attack filter")
    parser.add_argument(
        "--epsilon-attacks",
        nargs="+",
        default=None,
        help="Optional attack list used by the epsilon sweep phase",
    )
    parser.add_argument(
        "--result-dirs",
        nargs="+",
        default=None,
        help="Optional list of result directories used by multi-run phases",
    )
    parser.add_argument("--compute-shap", action="store_true", help="Compute SHAP explainability metrics for the selected run")
    parser.add_argument("--max-samples", type=int, default=64, help="Max number of samples used to build SHAP reference vectors")
    parser.add_argument("--background-samples", type=int, default=32, help="Number of background samples used by SHAP explainer")
    parser.add_argument("--topk-ratio", type=float, default=0.05, help="Top-k ratio used in SHAP feature vector comparisons")
    parser.add_argument("--cka-max-samples", type=int, default=64, help="Max number of samples used in the CKA phase")
    parser.add_argument("--cka-max-layers", type=int, default=8, help="Maximum number of layers sampled in the CKA phase")
    parser.add_argument(
        "--transferability-output",
        default=None,
        help="Optional CSV filename for transferability output (used by parallel analyze shards)",
    )
    parser.add_argument(
        "--merge-transferability",
        action="store_true",
        help="Merge transferability shards into canonical transferability.csv and exit",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    _log_phase_context(args)

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

    if args.mode == "analyze":
        if args.merge_transferability:
            merged_path = _merge_transferability_if_ready(args.output_dir)
            if merged_path:
                print(f"[INFO] Transferability merge ready at {merged_path}")
            else:
                print(f"[WARN] No transferability shards were merged for {args.output_dir}")
            return
        try:
            analyze_start = time.perf_counter()
            merge_flag = os.path.join(args.output_dir, ".summary_merge_complete")
            if os.path.exists(merge_flag):
                print(f"[INFO] Summary merge completed; proceeding with analyze for {args.output_dir}")
            else:
                print(f"[INFO] Summary shards present; ensuring canonical summary is merged before analyze in {args.output_dir}")
                _merge_summary_if_ready(args.output_dir)

            print(f"[INFO] Running analyze phase for {args.output_dir}")
            cache_args = argparse.Namespace(
                output_dir=args.output_dir,
                model=args.model,
                dataset=args.dataset,
                kind=args.kind,
            )
            cache_start = time.perf_counter()
            model_cache, loader_cache = AdversarialCore.rebuild_model_and_loader_cache(cache_args)
            print(
                f"[INFO] Rebuilt caches in {time.perf_counter() - cache_start:.2f}s: "
                f"models={len(model_cache)} datasets={len(loader_cache)}"
            )

            artifact_start = time.perf_counter()
            adv_datasets = AdversarialCore.discover_existing_adversarial_artifacts(
                output_dir=args.output_dir,
                model_filter=args.model,
                dataset_filter=args.dataset,
                attack_filter=args.attack,
                kind_filter=args.kind,
            )
            print(
                f"[INFO] Discovered {len(adv_datasets)} adversarial artifacts in "
                f"{time.perf_counter() - artifact_start:.2f}s"
            )

            rebuild_start = time.perf_counter()
            rebuilt_records = AdversarialCore.rebuild_summary_records_from_artifacts(
                output_dir=args.output_dir,
                model_cache=model_cache,
                loader_cache=loader_cache,
                adv_datasets=adv_datasets,
            )
            print(
                f"[INFO] Rebuilt {len(rebuilt_records)} summary records from artifacts in "
                f"{time.perf_counter() - rebuild_start:.2f}s"
            )
            if rebuilt_records:
                rebuilt_summary_path = _persist_rebuilt_summary_records(args.output_dir, rebuilt_records)
                if rebuilt_summary_path:
                    print(f"[INFO] Reconciled canonical summary at {rebuilt_summary_path}")

            transfer_start = time.perf_counter()
            compute_transfer_metrics(
                args.output_dir,
                model_cache,
                adv_datasets,
                transferability_output=args.transferability_output,
            )
            print(
                f"[INFO] Analyze phase completed in {time.perf_counter() - analyze_start:.2f}s "
                f"(transfer stage {time.perf_counter() - transfer_start:.2f}s)"
            )
        except Exception as exc:
            print(f"[ERROR] Analyze phase failed for {args.output_dir}: {exc}")
            print(traceback.format_exc().rstrip())
            raise
        return

    ready, summary_path = check_summary_ready(args.output_dir)
    if not ready:
        print(f"[WARN] Missing or empty summary.csv in {args.output_dir}. Run the generate phase first.")
        return

    if args.mode == "plot":
        print(f"[INFO] Running plot phase for {args.output_dir}")
        
        # Import and run the plotter
        from adversarial_plotting import AdversarialPlotter
        
        plotter = AdversarialPlotter(args.output_dir)

        # Generate figures from the summaries already present in the output directory.
        plotter.run()
        print(f"[INFO] ✅ Plotting complete. Figures saved to {args.output_dir}")
        return

    if args.mode == "compare":
        print(f"[INFO] Running compare phase for {args.output_dir} using {summary_path}")
        if args.result_dirs:
            generated = ReportingSuite.generate_multi_run_kind_comparison_table(
                output_dir=args.output_dir,
                result_dirs=args.result_dirs,
            )
            if generated:
                print(
                    f"[INFO] Multi-run comparison table saved to "
                    f"{os.path.join(args.output_dir, 'multi_run_kind_comparison_table.csv')}"
                )
            else:
                print("[WARN] Multi-run comparison table was not generated.")
        else:
            generated = ReportingSuite.generate_comparison_tables_from_csv(args.output_dir)
            if generated:
                print(f"[INFO] Comparison tables generated for {args.output_dir}")
            else:
                print(f"[WARN] Comparison tables were not generated for {args.output_dir}")
        return

    # ---------------------------------------------------------------------
    # Optional phases that were previously missing from the dispatcher.
    # These phases rely on the AdvancedExperimentSuite and need the model
    # and loader caches built from the existing checkpoints.
    # ---------------------------------------------------------------------
    if args.mode == "gradient_sim":
        print(f"[INFO] Running gradient similarity phase for {args.output_dir}")
        # Rebuild caches (model and data loaders) for the selected filters.
        cache_args = argparse.Namespace(
            output_dir=args.output_dir,
            model=args.model,
            dataset=args.dataset,
            kind=args.kind,
        )
        model_cache, loader_cache = AdversarialCore.rebuild_model_and_loader_cache(cache_args)
        suite = AdvancedExperimentSuite(
            instantiate_attack=lambda *a, **kw: None,
            model_kind_label=ReportingSuite.model_kind_label,
            classify_transfer_pair=ReportingSuite.classify_transfer_pair,
        )
        records = suite.gradient_similarity_phase(
            output_dir=args.output_dir,
            model_cache=model_cache,
            loader_cache=loader_cache,
        )
        print(f"[INFO] Gradient similarity phase produced {len(records)} records.")
        return

    if args.mode == "epsilon_sweep":
        selected_attacks = args.epsilon_attacks or ["PGD", "FGSM", "BIM"]
        print(
            f"[INFO] Running epsilon sensitivity phase for {args.output_dir} "
            f"with attacks={selected_attacks}"
        )
        cache_args = argparse.Namespace(
            output_dir=args.output_dir,
            model=args.model,
            dataset=args.dataset,
            kind=args.kind,
        )
        model_cache, loader_cache = AdversarialCore.rebuild_model_and_loader_cache(cache_args)
        print(
            f"[INFO] Rebuilt epsilon-sweep caches: "
            f"models={len(model_cache)} datasets={len(loader_cache)}"
        )
        suite = AdvancedExperimentSuite(
            instantiate_attack=lambda *a, **kw: None,
            model_kind_label=ReportingSuite.model_kind_label,
            classify_transfer_pair=ReportingSuite.classify_transfer_pair,
        )
        records = suite.epsilon_sensitivity_phase(
            output_dir=args.output_dir,
            model_cache=model_cache,
            loader_cache=loader_cache,
            attacks=selected_attacks,
        )
        print(
            f"[INFO] Epsilon sensitivity phase produced {len(records)} records. "
            f"Expected outputs: {os.path.join(args.output_dir, 'epsilon_sensitivity.csv')} and "
            f"{os.path.join(args.output_dir, 'epsilon_sensitivity_delta.csv')}"
        )
        return

    if args.mode == "statistics":
        print(f"[INFO] Running statistical significance phase for {args.output_dir}")
        suite = AdvancedExperimentSuite(
            instantiate_attack=lambda *a, **kw: None,
            model_kind_label=ReportingSuite.model_kind_label,
            classify_transfer_pair=ReportingSuite.classify_transfer_pair,
        )
        records = suite.statistical_significance_phase(
            output_dir=args.output_dir,
            result_dirs=args.result_dirs,
        )
        print(f"[INFO] Statistical significance phase produced {len(records)} records.")
        return

    if args.mode == "cka":
        print(f"[INFO] Running CKA similarity phase for {args.output_dir}")
        cache_args = argparse.Namespace(
            output_dir=args.output_dir,
            model=args.model,
            dataset=args.dataset,
            kind=args.kind,
        )
        model_cache, loader_cache = AdversarialCore.rebuild_model_and_loader_cache(cache_args)
        suite = AdvancedExperimentSuite(
            instantiate_attack=lambda *a, **kw: None,
            model_kind_label=ReportingSuite.model_kind_label,
            classify_transfer_pair=ReportingSuite.classify_transfer_pair,
        )
        records = suite.cka_similarity_phase(
            output_dir=args.output_dir,
            model_cache=model_cache,
            loader_cache=loader_cache,
            max_samples=args.cka_max_samples,
            max_layers=args.cka_max_layers,
        )
        print(f"[INFO] CKA similarity phase produced {len(records)} records.")
        return

    if args.mode == "compute_shap":
        print(f"[INFO] Running SHAP explainability phase for {args.output_dir}")
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

    if args.mode == "compute_tradeoff":
        print(f"[INFO] Running compute tradeoff phase for {args.output_dir}")
        compute_df, tradeoff_df = compute_tradeoff_for_results(args.output_dir)
        print(
            f"[INFO] Compute tradeoff phase produced {len(compute_df)} compute-profile rows and "
            f"{len(tradeoff_df)} tradeoff rows."
        )
        return

    if args.mode == "correlations":
        print(f"[INFO] Running correlations phase for {args.output_dir}")
        records = CorrelationSuite.run(args.output_dir)
        print(f"[INFO] Correlations phase produced {len(records)} records.")
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
