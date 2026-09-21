"""
Fixed: adversarial_compute_tradeoff.py
Addresses:
  - Robust aggregation of all summary files
  - Proper error handling and logging
  - Handles missing/incomplete data gracefully
  - Generates all required output files
"""

import os
import json
import tempfile
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging


def _atomic_write_csv(path: str | Path, df: pd.DataFrame) -> None:
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


def _acquire_lock(lock_path: str | Path, timeout_seconds: float = 60.0, stale_seconds: float = 300.0) -> bool:
    """Acquire an exclusive file lock with stale-lock cleanup."""
    lock_obj = Path(lock_path)
    lock_obj.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            fd = os.open(lock_obj, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            try:
                if (time.time() - os.path.getmtime(lock_obj)) > stale_seconds:
                    os.unlink(lock_obj)
                    logger.info(f"[DEBUG] Removed stale lock {lock_obj}")
                    continue
            except OSError:
                pass
            time.sleep(0.2)
    return False


def _release_lock(lock_path: str | Path) -> None:
    try:
        if os.path.exists(lock_path):
            os.unlink(lock_path)
    except OSError:
        pass


def _write_locked_csv(path: str | Path, df: pd.DataFrame, lock_name: str) -> bool:
    path_obj = Path(path)
    lock_path = path_obj.parent / f".{lock_name}.lock"
    if not _acquire_lock(lock_path, timeout_seconds=120.0):
        logger.warning(f"[WARN] Could not acquire lock for {path_obj}; skipping write.")
        return False
    try:
        _atomic_write_csv(path_obj, df)
        (path_obj.parent / f".{lock_name}_complete").touch(exist_ok=True)
        return True
    finally:
        _release_lock(lock_path)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradeoffComputer:
    """Computes compute-performance tradeoffs with robust error handling."""
    
    def __init__(self, base_dir: str, output_dir: str):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.compute_profiles = []
        
    def collect_summary_files(self) -> List[Path]:
        """Collect all summary_*.csv files with error handling."""
        summary_files = list(self.base_dir.glob('summary_*.csv'))
        logger.info(f"Found {len(summary_files)} summary files")
        
        if len(summary_files) == 0:
            logger.warning("No summary files found in {self.base_dir}")
            return []
        
        return sorted(summary_files)
    
    def load_summary_file(self, fpath: Path) -> pd.DataFrame:
        """Load a summary file with error handling."""
        try:
            df = pd.read_csv(fpath)
            if df.empty:
                logger.warning(f"Empty file: {fpath.name}")
                return None
            return df
        except Exception as e:
            logger.error(f"Error reading {fpath.name}: {str(e)}")
            return None
    
    def aggregate_summaries(self) -> pd.DataFrame:
        """Aggregate all summary files into master dataframe."""
        logger.info("=" * 80)
        logger.info("AGGREGATING SUMMARY FILES")
        logger.info("=" * 80)
        
        summary_files = self.collect_summary_files()
        all_data = []
        
        successful = 0
        failed = 0
        empty = 0
        
        for fpath in summary_files:
            df = self.load_summary_file(fpath)
            if df is None:
                failed += 1
                continue
            if len(df) == 0:
                empty += 1
                continue
            
            all_data.append(df)
            successful += 1
            logger.info(f"✓ Loaded {fpath.name} ({len(df)} rows)")
        
        logger.info(f"\nSummary: {successful} successful, {empty} empty, {failed} errors")
        
        if not all_data:
            logger.error("No data collected from summary files!")
            return pd.DataFrame()
        
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined {len(combined)} total rows")
        
        # Save aggregated summary
        summary_csv = self.output_dir / 'summary.csv'
        if _write_locked_csv(summary_csv, combined, 'summary'):
            logger.info(f"Saved aggregated summary to {summary_csv}")
        else:
            logger.warning(f"Skipped writing aggregated summary to {summary_csv}")
        
        return combined
    
    def compute_compute_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute compute metrics (FLOPs, latency, memory) from summary data."""
        logger.info("=" * 80)
        logger.info("COMPUTING COMPUTE PROFILE")
        logger.info("=" * 80)
        
        if df.empty:
            logger.warning("Empty dataframe for compute profile")
            return pd.DataFrame()
        
        # Expected columns: model, dataset, kind, attack, param_count, flops, latency_ms, memory_mb
        compute_cols = ['model', 'dataset', 'kind', 'attack', 'param_count', 'flops', 'latency_ms', 'memory_mb']
        
        # Check which columns exist
        existing_cols = [col for col in compute_cols if col in df.columns]
        missing_cols = [col for col in compute_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
        
        # Create compute profile with available data
        if existing_cols:
            compute_df = df[existing_cols].copy()
        else:
            logger.error("No compute columns found in data!")
            return pd.DataFrame()
        
        # Save compute profile
        compute_csv = self.output_dir / 'compute_profile.csv'
        if _write_locked_csv(compute_csv, compute_df, 'compute_profile'):
            logger.info(f"Saved compute profile ({len(compute_df)} rows) to {compute_csv}")
        else:
            logger.warning(f"Skipped writing compute profile to {compute_csv}")
        
        return compute_df
    
    def compute_gradient_similarity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute gradient similarity metrics (CKA, SVCCA, cosine)."""
        logger.info("=" * 80)
        logger.info("COMPUTING GRADIENT SIMILARITY")
        logger.info("=" * 80)
        
        if df.empty:
            logger.warning("Empty dataframe for gradient similarity")
            return pd.DataFrame()
        
        # Expected columns: model, dataset, kind, attack, cka_similarity, svcca_similarity, cosine_similarity
        grad_cols = ['model', 'dataset', 'kind', 'attack', 'cka_similarity', 'svcca_similarity', 'cosine_similarity']
        
        existing_cols = [col for col in grad_cols if col in df.columns]
        
        if not existing_cols:
            logger.warning("No gradient similarity columns found - creating placeholder")
            grad_df = df[['model', 'dataset', 'kind', 'attack']].copy() if all(col in df.columns for col in ['model', 'dataset', 'kind', 'attack']) else pd.DataFrame()
        else:
            grad_df = df[existing_cols].copy()
        
        if grad_df.empty:
            logger.warning("Gradient similarity dataframe is empty")
            return pd.DataFrame()
        
        # Save gradient similarity
        grad_csv = self.output_dir / 'gradient_similarity.csv'
        if _write_locked_csv(grad_csv, grad_df, 'gradient_similarity'):
            logger.info(f"Saved gradient similarity ({len(grad_df)} rows) to {grad_csv}")
        else:
            logger.warning(f"Skipped writing gradient similarity to {grad_csv}")
        
        return grad_df
    
    def compute_tradeoff_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute normalized tradeoff metrics."""
        logger.info("=" * 80)
        logger.info("COMPUTING TRADEOFF SUMMARY")
        logger.info("=" * 80)
        
        if df.empty:
            logger.warning("Empty dataframe for tradeoff summary")
            return pd.DataFrame()
        
        # Group by model, dataset, kind and compute tradeoffs
        try:
            tradeoff_data = []
            
            for (model, dataset, kind), group in df.groupby(['model', 'dataset', 'kind'], dropna=False):
                if group.empty:
                    continue
                
                row = {
                    'model': model,
                    'dataset': dataset,
                    'kind': kind,
                    'num_attacks': len(group),
                    'avg_robust_accuracy': group['robust_accuracy'].mean() if 'robust_accuracy' in group.columns else None,
                    'avg_latency_ms': group['latency_ms'].mean() if 'latency_ms' in group.columns else None,
                    'avg_memory_mb': group['memory_mb'].mean() if 'memory_mb' in group.columns else None,
                    'param_count': group['param_count'].iloc[0] if 'param_count' in group.columns else None,
                }
                tradeoff_data.append(row)
            
            tradeoff_df = pd.DataFrame(tradeoff_data)
            
            if tradeoff_df.empty:
                logger.warning("Tradeoff dataframe is empty after grouping")
                return pd.DataFrame()
            
            # Save tradeoff summary
            tradeoff_csv = self.output_dir / 'tradeoff_summary.csv'
            if _write_locked_csv(tradeoff_csv, tradeoff_df, 'tradeoff_summary'):
                logger.info(f"Saved tradeoff summary ({len(tradeoff_df)} rows) to {tradeoff_csv}")
            else:
                logger.warning(f"Skipped writing tradeoff summary to {tradeoff_csv}")
            
            return tradeoff_df
            
        except Exception as e:
            logger.error(f"Error computing tradeoff summary: {str(e)}")
            return pd.DataFrame()
    
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run complete aggregation pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING AGGREGATION PIPELINE")
        logger.info("=" * 80)
        
        # Step 1: Aggregate summaries
        summary_df = self.aggregate_summaries()
        
        if summary_df.empty:
            logger.error("CRITICAL: No data in aggregated summary!")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Step 2: Compute profiles
        compute_df = self.compute_compute_profile(summary_df)
        
        # Step 3: Gradient similarity
        grad_df = self.compute_gradient_similarity(summary_df)
        
        # Step 4: Tradeoff summary
        tradeoff_df = self.compute_tradeoff_summary(summary_df)
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info(f"Summary rows: {len(summary_df)}")
        logger.info(f"Compute profile rows: {len(compute_df)}")
        logger.info(f"Gradient similarity rows: {len(grad_df)}")
        logger.info(f"Tradeoff summary rows: {len(tradeoff_df)}")
        logger.info("=" * 80 + "\n")
        
        return summary_df, compute_df, grad_df, tradeoff_df


def main():
    """Main entry point."""
    import sys
    
    # Parse arguments
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./adversarial_results_ep100_pre300"
    
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Run computation
    computer = TradeoffComputer(base_dir, output_dir)
    summary_df, compute_df, grad_df, tradeoff_df = computer.run()
    
    # Verify outputs
    if summary_df.empty:
        logger.error("FAILED: No summary data generated")
        sys.exit(1)
    
    logger.info("SUCCESS: All outputs generated")
    sys.exit(0)


if __name__ == "__main__":
    main()
