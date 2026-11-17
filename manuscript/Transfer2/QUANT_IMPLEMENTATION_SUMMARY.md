# Implementation Summary: fp32 and fp16 Model Evaluation

## Objective
Enable studying **all effects** (accuracy, power/energy, adversarial robustness, calibration, OOD detection, corruption robustness, CKA) for **both fp32 and fp16 models** with a single flag (`--quant`).

## Implementation Complete ✅

All changes have been implemented and tested for syntax correctness.

## Files Modified

### 1. `attacks.py`
Added `quant` parameter to:
- `fgsm_attack()` - Uses fp16 autocast for forward pass when quant=True
- `pgd_attack()` - Uses fp16 autocast for forward pass when quant=True
- Both maintain fp32 for gradient computation to avoid underflow

### 2. `utils.py`
Added `quant` parameter to:
- `collect_confidences()` - Uses fp16 autocast when quant=True
- `evaluate_calibration()` - Passes quant to collect_confidences()
- `evaluate_ood()` - Passes quant to collect_confidences()
- `_accuracy_only()` - Uses fp16 autocast when quant=True
- `evaluate_corruptions()` - Passes quant to _accuracy_only() and adds quant field to rows
- `extract_features()` - Uses fp16 autocast when quant=True

### 3. `main.py`
Modified:
- `evaluate_adversarial_robustness()` - Added quant parameter, uses fp16 autocast for evaluation
- `process_checkpoint()` - Added quant parameter, runs all evaluations with specified precision
- `no_regions_row()` - Added quant parameter, includes precision field
- `main()` - Updated to run checkpoint processing with both fp32 and fp16 when --quant is specified
- Added `precision` column to all result rows
- Updated long-format CSV fieldnames to include `quant` field

## How It Works

### Without `--quant` Flag
- Runs evaluation **once** with fp32 (default behavior)
- All results have `precision="fp32"`
- Backward compatible with existing scripts

### With `--quant` Flag
- Runs evaluation **twice** for each checkpoint:
  1. First pass: `quant=False` (fp32)
  2. Second pass: `quant=True` (fp16)
- Results include `precision` column with values "fp32" or "fp16"
- Each checkpoint produces approximately 2x results (doubles runtime)
- All metrics are computed for both precisions

## Key Features

1. **Unified Flag**: Single `--quant` flag triggers both fp32 and fp16 evaluation
2. **Precision Column**: New `precision` column in CSV for easy filtering
3. **Consistent Column Names**: Adversarial metrics keep same column names, distinguished by precision column
4. **FP16 Gradient Protection**: Adversarial attacks use fp32 for gradients, avoiding underflow
5. **Full Coverage**: All evaluation types support both precisions
6. **Backward Compatible**: Existing scripts continue to work

## Usage Examples

```bash
# fp32 only (default)
python main.py --model VGG16 --dataset Cifar10

# Both fp32 and fp16
python main.py --model VGG16 --dataset Cifar10 --quant

# Both with full analysis
python main.py --model VGG16 --dataset Cifar10 --quant --analysis full

# Both with full analysis and corruptions
python main.py --model VGG16 --dataset Cifar10 --quant --analysis full --corruptions-dir ./data/corruptions
```

## Data Output

### Main CSV (e.g., `cka_results_VGG16_Cifar10.csv`)
- New column: `precision` (values: "fp32", "fp16")
- All other columns remain the same
- Each row represents one (checkpoint, repetition, precision) combination

### Long-Format CSVs
- `robustness_curves_*.csv`: Added `quant` column
- `calibration_*.csv`: Added `quant` column  
- `corruption_*.csv`: Added `quant` column

## Technical Details

### FP16 Implementation
- Uses PyTorch's `torch.cuda.amp.autocast()` for fp16 inference
- Forward pass: fp16 when quant=True
- Backward pass: fp32 (for adversarial attacks)
- Storage estimation: fp16 for floating point tensors

### Runtime Impact
- Without `--quant`: Same runtime as before
- With `--quant`: Approximately 2x runtime (runs full pipeline twice)
- Memory usage: Similar (models are loaded, evaluated, and unloaded sequentially)

## Verification

✅ All Python files compile without syntax errors  
✅ Function signatures updated with `quant` parameter where needed  
✅ Import/export statements compatible with existing code  
✅ Backward compatibility maintained  

## Next Steps

1. Test with actual model checkpoints to verify runtime behavior
2. Validate that results are reasonable for both precisions
3. Update documentation and README files
4. Consider adding a `--precision` argument as an alternative to `--quant` for explicit control

## Documentation

See `CHANGES_QUANT.md` for detailed change descriptions.
