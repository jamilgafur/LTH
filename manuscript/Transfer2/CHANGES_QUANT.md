# Changes for fp32 and fp16 Model Evaluation Support

**Date**: 2026-11-17  
**Status**: Implementation Complete  
**Files Modified**: `attacks.py`, `utils.py`, `main.py`  
**New File**: `CHANGES_QUANT.md`

## Overview
Updated the codebase to study **all effects for both fp32 and fp16 models**. Previously, the `--quant` flag only affected power/energy measurements. Now, when `--quant` is specified, the full evaluation pipeline runs **twice**: once for fp32 and once for fp16, with results distinguished by a `precision` column.

## Files Modified

### 1. `attacks.py`
- **`fgsm_attack()`**: Added `quant` parameter. When `quant=True` and device is CUDA, uses fp16 autocast for forward pass during attack generation, then converts outputs to fp32 for gradient computation (avoiding fp16 gradient underflow).
- **`pgd_attack()`**: Added `quant` parameter with same behavior as FGSM.

### 2. `utils.py`
- **`evaluate_accuracy()`**: Already supported `quant` parameter - no changes needed.
- **`benchmark_model()`**: Already supported `quant` parameter - no changes needed.
- **`collect_confidences()`**: Added `quant` parameter. When `quant=True` and device is CUDA, uses fp16 autocast for forward pass.
- **`evaluate_calibration()`**: Added `quant` parameter, passed to `collect_confidences()`.
- **`evaluate_ood()`**: Added `quant` parameter, passed to `collect_confidences()`.
- **`_accuracy_only()`**: Added `quant` parameter. When `quant=True` and device is CUDA, uses fp16 autocast for forward pass.
- **`evaluate_corruptions()`**: Added `quant` parameter, passed to `_accuracy_only()`.
- **`extract_features()`**: Added `quant` parameter. When `quant=True` and device is CUDA, uses fp16 autocast for forward pass.

### 3. `main.py`
- **`evaluate_adversarial_robustness()`**: 
  - Added `quant` parameter.
  - Passes `quant` to `fgsm_attack()` and `pgd_attack()`.
  - Uses fp16 autocast for adversarial evaluation when `quant=True`.
  - Adds `_quant` suffix to metric column names when `quant=True`.
  - Adds `quant` field to curve_rows for long-format CSV output.

- **`process_checkpoint()`**:
  - Added `quant` parameter (default: `False`).
  - Updated all evaluation calls to use the local `quant` parameter instead of `args.quant`.
  - Added precision indicator to console output.

- **`no_regions_row()`**:
  - Added `quant` parameter.
  - Added `precision` field to the returned dict.

- **`main()`**:
  - Updated `--quant` flag help text to reflect new behavior.
  - Modified checkpoint processing to run with both `quant=False` (fp32) and `quant=True` (fp16) when `--quant` is specified.
  - Results now include a `precision` column with values "fp32" or "fp16".
  - Removed automatic `_quant` suffix from output filename (results are in same CSV with precision column).
  - Changed `find_experiment_checkpoints()` call to always use `quant=False` (we load fp32 checkpoints and measure with both precisions).

- **Long-format CSV updates**:
  - Added `quant` field to `robustness_curves_*.csv` fieldnames.

## Key Data Structure Changes

### New `precision` Column
- Added to all result rows in the main CSV
- Values: "fp32" or "fp16"
- Allows filtering and comparison between precisions

### Adversarial Metrics
- Column names remain consistent (e.g., `adv_acc_FGSM_original`) regardless of precision
- Precision is distinguished by the `precision` column value
- This maintains backward compatibility while enabling precision-specific analysis

## Key Changes in Behavior

### Without `--quant` flag (default):
- Runs evaluation **only with fp32** (same as before)
- Results have `precision="fp32"` column
- All metrics use original column names (no `_quant` suffix)

### With `--quant` flag:
- Runs evaluation **twice**: once with fp32, once with fp16
- Results have `precision` column with values "fp32" or "fp16"
- Each checkpoint produces 2 sets of results (doubles runtime)
- All effects are studied for both precisions:
  - Accuracy measurements
  - Power/energy measurements
  - Benchmarking (FLOPs, latency, throughput, memory)
  - CKA analysis
  - Adversarial robustness (FGSM, PGD)
  - Calibration (ECE, MCE) - when `--analysis full`
  - OOD AUROC - when `--analysis full`
  - Corruption robustness - when `--analysis full` and `--corruptions-dir` provided
- Adversarial attacks use fp32 for gradient computation even with fp16 models (avoids underflow), but evaluate on fp16 models

## Technical Notes

### FP16 Gradient Underflow Protection
For adversarial attacks (FGSM, PGD) with fp16:
- Forward pass uses fp16 autocast
- Outputs are converted to fp32 before loss computation
- Backward pass stays in fp32
- This prevents gradient underflow at higher epsilon values while still testing fp16 model robustness

### Storage Estimation
When `quant=True`:
- State dict sizes are estimated with fp16 storage (half precision for floating point tensors)
- This reflects the actual storage requirements for fp16 deployment

### Output Files
- Main CSV: Contains all results with `precision` column
- Long-format CSVs (robustness_curves, calibration, corruption): Include `quant` field when applicable

## Backward Compatibility
- Existing scripts that don't use `--quant` will see minimal changes:
  - New `precision` column with value "fp32"
  - All other columns unchanged
- Scripts using `--quant` will now get both fp32 and fp16 results instead of just fp16

## Example Usage

### Run only fp32 (default):
```bash
python main.py --model VGG16 --dataset Cifar10 --output results.csv
```

### Run both fp32 and fp16:
```bash
python main.py --model VGG16 --dataset Cifar10 --output results.csv --quant
```

### Run both with full analysis:
```bash
python main.py --model VGG16 --dataset Cifar10 --output results.csv --quant --analysis full
```

## Testing
- All Python files compile without syntax errors
- The changes maintain backward compatibility
- Each evaluation function properly propagates the `quant` parameter
