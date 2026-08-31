"""Integration module for adversarial analysis into transfer.py workflow.

This module provides utilities to seamlessly run adversarial robustness
analysis after transfer learning experiments.

Usage in transfer.py:

    from adversarial_integration import run_adversarial_analysis_pipeline
    
    # After your transfer learning experiments:
    run_adversarial_analysis_pipeline(
        output_dir="adversarial_results",
        use_hpc=False  # Set to True if on HPC cluster
    )
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Optional, List


def run_adversarial_analysis_pipeline(
    output_dir: str = "adversarial_results",
    use_hpc: bool = False,
    mode: str = "full",
    model_filter: Optional[str] = None,
    dataset_filter: Optional[str] = None,
    attack_filter: Optional[str] = None,
    wait_for_completion: bool = False,
) -> dict:
    """Run the adversarial analysis pipeline.
    
    Args:
        output_dir: Output directory for results.
        use_hpc: If True, submit jobs to HPC cluster using PBS; otherwise run locally.
          mode: Execution mode ('full', 'generate', 'analyze', 'plot', 'compare',
              'compute_tradeoff', 'correlations').
        model_filter: Filter by model name (e.g., 'VGG16'). None means all.
        dataset_filter: Filter by dataset name (e.g., 'Cifar10'). None means all.
        attack_filter: Filter by attack name (e.g., 'PGD'). None means all.
        wait_for_completion: If True, wait for all jobs to complete before returning.
    
    Returns:
        Dictionary with status and job information.
    """
    if use_hpc:
        return _submit_hpc_jobs(
            Path(__file__).parent, output_dir, mode,
            model_filter, dataset_filter, attack_filter
        )
    else:
        return _run_locally(
            output_dir, mode,
            model_filter, dataset_filter, attack_filter
        )


def _run_locally(
    output_dir: str,
    mode: str,
    model_filter: Optional[str],
    dataset_filter: Optional[str],
    attack_filter: Optional[str],
) -> dict:
    """Run adversarial analysis locally."""
    cmd = [
        sys.executable,
        "-m",
        "manuscript.Tranfer.adversarial_analysis",
        "--mode", mode,
        "--output-dir", output_dir,
    ]
    
    if model_filter:
        cmd.extend(["--model", model_filter])
    if dataset_filter:
        cmd.extend(["--dataset", dataset_filter])
    if attack_filter:
        cmd.extend(["--attack", attack_filter])
    
    print(f"[ADVERSARIAL ANALYSIS] Running locally: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return {
            "status": "success",
            "output_dir": output_dir,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr,
        }


def _submit_hpc_jobs(
    script_dir: Path,
    output_dir: str,
    mode: str,
    model_filter: Optional[str],
    dataset_filter: Optional[str],
    attack_filter: Optional[str],
) -> dict:
    """Submit adversarial analysis jobs to HPC cluster."""
    orchestrate_script = script_dir / "adversarial_hpc_orchestrate.sh"
    
    if not orchestrate_script.exists():
        return {
            "status": "error",
            "message": f"HPC orchestration script not found at {orchestrate_script}"
        }
    
    # Make scripts executable
    os.chmod(orchestrate_script, 0o755)
    
    # Determine which phases to submit
    phases = ["generate", "analyze", "plot"] if mode == "full" else [mode]
    
    job_ids = []
    for phase in phases:
        cmd = [str(orchestrate_script), phase, output_dir]
        
        print(f"[ADVERSARIAL ANALYSIS] Submitting HPC phase: {phase}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Extract job IDs from output if available
            if "Job ID" in result.stdout:
                for line in result.stdout.split("\n"):
                    if "Job ID" in line:
                        job_id = line.split()[-1]
                        job_ids.append(job_id)
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "message": f"Failed to submit {phase} jobs: {e}",
                "stderr": e.stderr,
            }
    
    return {
        "status": "submitted",
        "output_dir": output_dir,
        "phases": phases,
        "job_ids": job_ids,
        "message": f"Submitted {len(job_ids)} HPC jobs. Monitor with: qstat",
    }


def get_adversarial_results(output_dir: str = "adversarial_results") -> dict:
    """Load adversarial analysis results from output CSVs.
    
    Args:
        output_dir: Output directory containing results.
    
    Returns:
        Dictionary with loaded dataframes and metadata.
    """
    import pandas as pd
    
    results = {"status": "success", "data": {}}
    
    summary_path = Path(output_dir) / "summary.csv"
    if summary_path.exists():
        results["data"]["summary"] = pd.read_csv(summary_path)
        results["data"]["summary_stats"] = {
            "mean_accuracy_drop": results["data"]["summary"]["accuracy_drop"].mean(),
            "max_accuracy_drop": results["data"]["summary"]["accuracy_drop"].max(),
            "by_attack": results["data"]["summary"].groupby("attack")["accuracy_drop"].mean().to_dict(),
            "by_kind": results["data"]["summary"].groupby("kind")["accuracy_drop"].mean().to_dict(),
        }
    else:
        results["data"]["summary"] = None
        results["summary_available"] = False
    
    transfer_path = Path(output_dir) / "transferability.csv"
    if transfer_path.exists():
        results["data"]["transferability"] = pd.read_csv(transfer_path)
        results["data"]["transfer_stats"] = {
            "mean_transfer_acc": results["data"]["transferability"]["transfer_acc"].mean(),
            "min_transfer_acc": results["data"]["transferability"]["transfer_acc"].min(),
        }
    else:
        results["data"]["transferability"] = None
        results["transferability_available"] = False
    
    # List generated plots
    plot_dir = Path(output_dir)
    plot_files = list(plot_dir.glob("*.png"))
    results["plots"] = [p.name for p in plot_files]
    
    return results


def print_adversarial_summary(output_dir: str = "adversarial_results") -> None:
    """Print a formatted summary of adversarial analysis results."""
    results = get_adversarial_results(output_dir)
    
    if results["status"] == "error":
        print(f"Error loading results: {results.get('message', 'Unknown error')}")
        return
    
    print("\n" + "=" * 80)
    print("ADVERSARIAL ROBUSTNESS ANALYSIS SUMMARY")
    print("=" * 80)
    
    if results["data"]["summary"] is not None:
        stats = results["data"]["summary_stats"]
        print(f"\nDirect Attack Results:")
        print(f"  Mean accuracy drop: {stats['mean_accuracy_drop']:.2%}")
        print(f"  Max accuracy drop:  {stats['max_accuracy_drop']:.2%}")
        print(f"\n  By Attack Type:")
        for attack, drop in stats["by_attack"].items():
            print(f"    {attack:15s}: {drop:.2%}")
        print(f"\n  By Model Kind:")
        for kind, drop in stats["by_kind"].items():
            print(f"    {kind:15s}: {drop:.2%}")
    
    if results["data"]["transferability"] is not None:
        stats = results["data"]["transfer_stats"]
        print(f"\nAdversarial Transferability:")
        print(f\"  Mean transfer accuracy: {stats['mean_transfer_acc']:.2%}")
        print(f"  Min transfer accuracy:  {stats['min_transfer_acc']:.2%}")
    
    if results[\"plots\"]:
        print(f"\nGenerated Visualizations ({len(results['plots'])} files):")
        for plot in sorted(results[\"plots\"])[:10]:
            print(f\"  - {plot}\")
        if len(results[\"plots\"]) > 10:
            print(f\"  ... and {len(results['plots']) - 10} more\")
    
    print(\"\\n\" + \"=\" * 80)


if __name__ == \"__main__\":
    import argparse
    
    parser = argparse.ArgumentParser(description=\"Adversarial analysis integration utilities.\")
    parser.add_argument(\"--run\", action=\"store_true\", help=\"Run the full adversarial analysis pipeline.\")
    parser.add_argument(\"--summary\", action=\"store_true\", help=\"Print summary of results.\")
    parser.add_argument(\"--output-dir\", type=str, default=\"adversarial_results\", help=\"Output directory.\")
    parser.add_argument(\"--use-hpc\", action=\"store_true\", help=\"Submit to HPC cluster.\")
    args = parser.parse_args()
    
    if args.run:
        result = run_adversarial_analysis_pipeline(
            output_dir=args.output_dir,
            use_hpc=args.use_hpc,
        )
        print(json.dumps(result, indent=2))
    
    if args.summary:
        print_adversarial_summary(args.output_dir)
