"""Activation-variance probing and variance-based collapse-region discovery."""

import copy
import gc
import warnings

import numpy as np
import torch

from .core import collapse_only
from .graph_utils import count_trainable_params

# ==============================================================================
# Dynamic Collapse Logic
# ==============================================================================

def get_layer_variances(model, dummy_input):
    """
    Enhanced hook to capture variance. 
    Handles 1x1 spatial dimensions to prevent NaN.
    """
    variances = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                # Check if we have spatial dimensions (NCHW) and if they are > 1x1
                if out.ndim == 4:
                    if out.shape[2] > 1 and out.shape[3] > 1:
                        # Standard spatial variance
                        variances[name] = out.var(dim=[2, 3], unbiased=False).mean().item()
                    else:
                        # Fallback for 1x1 tensors: calculate variance across channel dim safely
                        variances[name] = out.var(dim=1, unbiased=False).mean().item() if out.shape[1] > 1 else 0.0
                else:
                    # Non-spatial (e.g., Linear layer output)
                    variances[name] = out.var(unbiased=False).item()
            return None
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(make_hook(name)))
            
    # Run probe
    model.eval()
    with torch.no_grad():
        try:
            model(dummy_input)
        except Exception as e:
            print(f"[ERROR] Variance probe failed: {e}")
        
    # Cleanup
    for h in hooks:
        h.remove()
        
    return variances

def discover_regions(model, cnn_layers, variances, input_shape=(1, 3, 224, 224), window_size=3, veto_fraction=0.25):
    from fvcore.nn import FlopCountAnalysis  # optional dependency, only used by discovery

    print(f"\n[DEBUG] {'='*40}")
    print(f"[DEBUG] === Starting Dynamic Config Generator ===")
    print(f"[DEBUG] {'='*40}")

    device = next(model.parameters()).device
    module_dict = dict(model.named_modules())

    # =====================================================================
    # HELPER 1: Base Metrics Calculation
    # =====================================================================
    def get_base_metrics():
        base_params = count_trainable_params(model)
        dummy_input = torch.randn(1, *input_shape[1:]).to(device)
        model.eval()
        try:
            with torch.no_grad():
                base_flops = FlopCountAnalysis(model, dummy_input).unsupported_ops_warnings(False).total()
            check_flops = True
        except Exception as e:
            base_flops = float('inf')
            check_flops = False
            print(f"[WARN] Failed to calculate base FLOPs: {e}")
        return base_params, base_flops, check_flops, dummy_input

    # =====================================================================
    # HELPER 2: Structural Validity & LCA Logic
    # =====================================================================
    def get_lca_path(start_name, end_name):
        start_parts = start_name.split('.') if start_name else []
        end_parts = end_name.split('.') if end_name else []
        common = []
        if start_name == end_name and len(start_parts) > 0:
            return '.'.join(start_parts[:-1])
        for a, b in zip(start_parts, end_parts):
            if a == b: common.append(a)
            else: break
        return '.'.join(common)

    def is_valid_lca(lca_mod):
        if lca_mod is None: return True
        if isinstance(lca_mod, (torch.nn.Sequential, torch.nn.ModuleList)): return True
        
        # [FIX] Allow the root model explicitly so stages can be bridged
        if lca_mod.__class__.__name__ == type(model).__name__: return True
        
        mod_name = type(lca_mod).__name__
        
        # [FIX] Broad, universal whitelist for all major architectures
        allowed_substrings = ['Inception', 'Block', 'Stage', 'Layer', 'Residual', 'AnyNetX']
        if any(sub in mod_name for sub in allowed_substrings): 
            return True
            
        if hasattr(lca_mod, 'children') and list(lca_mod.children()):
            return False
            
        return True

    # [FIX] Helper to count strictly parametric layers
    def get_parametric_count(layer_names):
        count = 0
        for ln in layer_names:
            mod = module_dict.get(ln)
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                count += 1
        return count

    def format_region(layer_names):
        if not layer_names:
            return "<empty>"
        return f"{layer_names[0]} -> {layer_names[-1]}"

    # =====================================================================
    # HELPER 3: Phase 1 & 2 - Variance Normalization & Expansion
    # =====================================================================

    def extract_and_expand_candidates():
            # Phase 1: Rolling Variance
            rolling_means = []
            for i in range(len(variances)):
                start = max(0, i - window_size)
                end = min(len(variances), i + window_size + 1)
                rolling_means.append(np.mean(variances[start:end]))
                
            h_values = []
            for var, mean_var in zip(variances, rolling_means):
                if var - mean_var < 0: h_values.append(max((var - mean_var) / mean_var, -1.0))
                else: h_values.append(min((var - mean_var) / mean_var, 1.0))

            num_layers = len(cnn_layers)
            veto_idx = int(num_layers * veto_fraction)
            
            R_candidates_raw = []
            current_set = []
            status_map = {}
            
            print(f"\n[DEBUG] --- Phase 2A: Scanning {num_layers} layers for Raw Candidates ---")
            for i, h in enumerate(h_values):
                layer_name = cnn_layers[i]
                mod = module_dict.get(layer_name)
                
                is_classifier = any(term in layer_name.lower() for term in ['classifier', 'fc', 'aux'])
                is_parametric = isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear))
                is_valid_target = isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU, torch.nn.BatchNorm2d)) and not is_classifier
                
                if i < veto_idx: 
                    status = "Veto"
                elif not is_valid_target: 
                    status = "Rejected"
                elif h >= 0 or np.isnan(h): 
                    # [CRITICAL FIX] Non-parametric layers act as "Passive" bridges so they don't fracture blocks
                    status = "Passive" if not is_parametric else "Rejected"
                else: 
                    status = "Active"
                    
                status_map[layer_name] = status

                if status in {"Rejected", "Veto"}:
                    reason = ""
                    if status == "Veto":
                        reason = f"veto window (idx={i}, cutoff={veto_idx})"
                    elif not is_valid_target:
                        reason = f"invalid target type={type(mod).__name__ if mod is not None else 'None'}"
                    elif h >= 0 or np.isnan(h):
                        reason = f"non-negative or NaN h={h:.6f}"
                    print(
                        f"[DEBUG] [SCAN] Layer {i:02d} | {layer_name:<45} | h={h:.6f} | "
                        f"status={status:<8} | reason={reason}"
                    )
                elif status == "Passive":
                    print(
                        f"[DEBUG] [SCAN] Layer {i:02d} | {layer_name:<45} | h={h:.6f} | "
                        f"status=Passive  | reason=bridge layer"
                    )
                
                # Allow Active layers, and Passive layers if they are attached to an active chain
                if status == "Active" or (status == "Passive" and len(current_set) > 0):
                    current_set.append(layer_name)
                else:
                    if get_parametric_count(current_set) >= 2: 
                        print(
                            f"[DEBUG] [RAW] Accepted raw candidate {format_region(current_set)} "
                            f"| layers={len(current_set)} | parametric={get_parametric_count(current_set)}"
                        )
                        R_candidates_raw.append(current_set)
                    else:
                        if current_set:
                            print(
                                f"[DEBUG] [RAW] Discarding short fragment {format_region(current_set)} "
                                f"| layers={len(current_set)} | parametric={get_parametric_count(current_set)}"
                            )
                        for orphan in current_set: status_map[orphan] = "Gap"
                    current_set = []
                    
            if get_parametric_count(current_set) >= 2:
                print(
                    f"[DEBUG] [RAW] Accepted raw candidate {format_region(current_set)} "
                    f"| layers={len(current_set)} | parametric={get_parametric_count(current_set)}"
                )
                R_candidates_raw.append(current_set)
            else:
                if current_set:
                    print(
                        f"[DEBUG] [RAW] Discarding trailing fragment {format_region(current_set)} "
                        f"| layers={len(current_set)} | parametric={get_parametric_count(current_set)}"
                    )
                for orphan in current_set: status_map[orphan] = "Gap"

            # Phase 2B: Structural LCA Expansion
            R_candidates = []
            print(f"\n[DEBUG] --- Phase 2B: Starting Structural LCA Expansion ---")
            for r in R_candidates_raw:
                start_idx = cnn_layers.index(r[0])
                end_idx = cnn_layers.index(r[-1])
                print(
                    f"[DEBUG] [EXPAND] Starting from raw region {format_region(r)} "
                    f"| layers={len(r)} | parametric={get_parametric_count(r)}"
                )
                
                trapped = False
                while True:
                    lca_path = get_lca_path(cnn_layers[start_idx], cnn_layers[end_idx])
                    lca_mod = module_dict.get(lca_path) if lca_path else model
                    
                    if is_valid_lca(lca_mod) and get_parametric_count(cnn_layers[start_idx : end_idx + 1]) >= 2:
                        break 
                        
                    expanded = False
                    # [FIX] Allow expansion into Passive layers
                    if end_idx + 1 < len(cnn_layers) and status_map.get(cnn_layers[end_idx+1]) in ["Active", "Passive"]:
                        end_idx += 1; expanded = True
                        print(f"[DEBUG] [EXPAND] Extended right to {cnn_layers[end_idx]}")
                    elif start_idx > 0 and start_idx - 1 >= veto_idx and status_map.get(cnn_layers[start_idx-1]) in ["Active", "Passive"]:
                        start_idx -= 1; expanded = True
                        print(f"[DEBUG] [EXPAND] Extended left to {cnn_layers[start_idx]}")
                            
                    if not expanded:
                        trapped = True
                        break 
                
                if not trapped:
                    expanded_r = cnn_layers[start_idx : end_idx + 1]
                    if get_parametric_count(expanded_r) >= 2:
                        print(
                            f"[DEBUG] [EXPAND] Final candidate {format_region(expanded_r)} "
                            f"| layers={len(expanded_r)} | parametric={get_parametric_count(expanded_r)}"
                        )
                        R_candidates.append(expanded_r)
                    else:
                        print(f"[DEBUG] [!] Discarding orphan fragment (size {len(expanded_r)}): {expanded_r}")
                        for name in expanded_r: status_map[name] = "Rejected (Orphan)"
                else:
                    print(f"[DEBUG] [-] Trapped region: {r[0]}. Marking as Gap.")
                    for idx in range(start_idx, end_idx + 1): status_map[cnn_layers[idx]] = "Gap"

            return R_candidates, status_map
    # =====================================================================
    # HELPER 4: Phase 3 - Surrogate Compilation & BFS Shaving
    # =====================================================================
    def evaluate_candidates(R_candidates, base_params, base_flops, check_flops, dummy_input):
        experiment_regions = {}
        set_counter = 0
        R_final = []
        tested_regions = set()
        queue = R_candidates.copy()
        
        while queue:
            r = queue.pop(0) 
            # [CRITICAL FIX] Trim non-parametric layers from boundaries
            while r and not isinstance(module_dict.get(r[0]), (torch.nn.Conv2d, torch.nn.Linear)):
                r.pop(0)
            while r and not isinstance(module_dict.get(r[-1]), (torch.nn.Conv2d, torch.nn.Linear)):
                r.pop(-1)

            # [FIX] Strict parametric size enforcement: Ignore anything shaved too small
            if get_parametric_count(r) < 2: continue
                
            start_name = r[0]
            end_name = r[-1]
            region_tuple = (start_name, end_name)
            
            if region_tuple in tested_regions: continue
            tested_regions.add(region_tuple)
            
            try:
                print(f"    [STEP] Testing sequence: {start_name} -> {end_name} (Length: {len(r)})")
                test_model = copy.deepcopy(model).to(device)
                
                collapsed_model = collapse_only(
                    model=test_model,
                    compression_set={"test": region_tuple},
                    input_shape=input_shape,
                    device=device,
                    dry_run=False 
                )
                
                collapsed_model.eval()
                with torch.no_grad(): _ = collapsed_model(dummy_input)

                new_params = count_trainable_params(collapsed_model)
                try:
                    new_flops = FlopCountAnalysis(collapsed_model, dummy_input).unsupported_ops_warnings(False).total()
                except Exception:
                    new_flops = float('inf')

                params_saved = base_params - new_params
                flops_saved = base_flops - new_flops if check_flops and np.isfinite(base_flops) and np.isfinite(new_flops) else None
                param_pct = (params_saved / base_params * 100.0) if base_params else 0.0
                flop_pct = (flops_saved / base_flops * 100.0) if flops_saved is not None and base_flops not in {0, float('inf')} else None

                if new_params < base_params and (not check_flops or new_flops < base_flops):
                    flop_msg = (
                        f" | FLOPs ({base_flops:,.0f} -> {new_flops:,.0f}, saved {flops_saved:,.0f}, {flop_pct:.4f}%)"
                        if flop_pct is not None else ""
                    )
                    print(
                        f"        [✓] ACCEPTED: region={start_name} -> {end_name} | "
                        f"Params ({base_params:,} -> {new_params:,}, saved {params_saved:,}, {param_pct:.4f}%)"
                        f"{flop_msg}"
                    )
                    experiment_regions[f"Set_{set_counter}"] = region_tuple
                    R_final.append(region_tuple)
                    set_counter += 1
                else:
                    flop_msg = (
                        f" | FLOPs ({base_flops:,.0f} -> {new_flops:,.0f})"
                        if check_flops and np.isfinite(new_flops) and np.isfinite(base_flops) else ""
                    )
                    print(
                        f"        [X] FAILED TO REDUCE: region={start_name} -> {end_name} | "
                        f"Params ({base_params:,} -> {new_params:,}){flop_msg}. Shaving..."
                    )
                    if len(r) > 2:
                        left_shaved = r[1:]
                        right_shaved = r[:-1]
                        print(
                            f"        [DEBUG] [SHAVE] Left candidate {format_region(left_shaved)} | "
                            f"parametric={get_parametric_count(left_shaved)}"
                        )
                        print(
                            f"        [DEBUG] [SHAVE] Right candidate {format_region(right_shaved)} | "
                            f"parametric={get_parametric_count(right_shaved)}"
                        )
                        if get_parametric_count(left_shaved) >= 2: queue.append(left_shaved)
                        if get_parametric_count(right_shaved) >= 2: queue.append(right_shaved)

            except Exception as e:
                print(f"        [!] CRASH ({type(e).__name__}: {e}).")
            finally:
                if 'test_model' in locals(): del test_model
                if 'collapsed_model' in locals(): del collapsed_model
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
        return experiment_regions, R_final
    
    # =====================================================================
    # HELPER 5: Phase 4 - Greedy Integration
    # =====================================================================
    def greedy_integration(R_final, experiment_regions, base_params, base_flops, check_flops, dummy_input):
        if len(R_final) > 1:
            # Force the combined region to be the literal set of all valid sets
            experiment_regions["Dynamic_Region_All_Combined"] = R_final
            print(f"[INFO] Forced {len(R_final)} sets into 'Dynamic_Region_All_Combined'.")
            
        return experiment_regions
    # def greedy_integration(R_final, experiment_regions, base_params, base_flops, check_flops, dummy_input):
    #     if len(R_final) <= 1: return experiment_regions
        
    #     print(f"\n[INFO] Phase 4: Validating 'Dynamic_Region_All_Combined' (Greedy Integration)...")
    #     safe_combined_regions = []
    #     current_best_params = base_params
    #     current_best_flops = base_flops
        
    #     for region_tuple in R_final:
    #         candidate_combo = safe_combined_regions + [region_tuple]
    #         try:
    #             test_model = copy.deepcopy(model).to(device)
    #             combo_dict = {f"comb_{i}": reg for i, reg in enumerate(candidate_combo)}
                
    #             collapsed_combined = collapse_only(
    #                 model=test_model, compression_set=combo_dict,
    #                 input_shape=input_shape, device=device, dry_run=False 
    #             )
                
    #             collapsed_combined.eval()
    #             with torch.no_grad(): _ = collapsed_combined(dummy_input)
                    
    #             new_params_combo = count_trainable_params(collapsed_combined)
    #             try:
    #                 new_flops_combo = FlopCountAnalysis(collapsed_combined, dummy_input).unsupported_ops_warnings(False).total()
    #             except Exception:
    #                 new_flops_combo = float('inf')

    #             param_decreased = new_params_combo < current_best_params
    #             flop_decreased = check_flops and new_flops_combo < current_best_flops
                
    #             if not (param_decreased or flop_decreased):
    #                 raise ValueError(f"Diminishing returns.")
                
    #             print(f"        [+] Merged safely: {region_tuple[0]} -> {region_tuple[1]}")
    #             current_best_params = new_params_combo
    #             current_best_flops = new_flops_combo
    #             safe_combined_regions.append(region_tuple)
                
    #         except Exception as e:
    #             print(f"        [-] Merge rejected: Adding {region_tuple[0]} -> {region_tuple[1]} broke constraint")
    #         finally:
    #             if 'test_model' in locals(): del test_model
    #             if 'collapsed_combined' in locals(): del collapsed_combined
    #             gc.collect()
    #             if torch.cuda.is_available(): torch.cuda.empty_cache()
                
    #     if len(safe_combined_regions) > 1:
    #         experiment_regions["Dynamic_Region_All_Combined"] = safe_combined_regions
            
    #     return experiment_regions

    # =====================================================================
    # MAIN EXECUTION FLOW
    # =====================================================================
    base_params, base_flops, check_flops, dummy_input = get_base_metrics()
    print(f"[DEBUG] Base model parameters: {base_params:,} | Base FLOPs: {base_flops:,}")
    
    R_candidates, status_map = extract_and_expand_candidates()
    
    print(f"\n[INFO] Phase 3: Evaluating {len(R_candidates)} Candidate Regions (with Smart Boundary Shaving)...")
    experiment_regions, R_final = evaluate_candidates(R_candidates, base_params, base_flops, check_flops, dummy_input)
    
    experiment_regions = greedy_integration(R_final, experiment_regions, base_params, base_flops, check_flops, dummy_input)

    print(f"\n[DEBUG] --- Final Accepted Regions Summary ---")
    if R_final:
        for idx, (start_name, end_name) in enumerate(R_final):
            start_idx = cnn_layers.index(start_name)
            end_idx = cnn_layers.index(end_name)
            region_layers = cnn_layers[start_idx:end_idx + 1]
            print(
                f"[DEBUG] [FINAL] Set_{idx}: {start_name} -> {end_name} | "
                f"layers={len(region_layers)} | parametric={get_parametric_count(region_layers)}"
            )
    else:
        print("[DEBUG] [FINAL] No regions accepted.")
    
    experiment_regions["Status_Map"] = status_map
    experiment_regions["Control_Continuted"] = None
    print(f"\n[DEBUG] === Finished Dynamic Config Generator ===")
    return experiment_regions


def get_dynamic_experiment_config(*args, **kwargs):
    """Deprecated alias for :func:`discover_regions`."""
    warnings.warn(
        "get_dynamic_experiment_config is deprecated; use discover_regions instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return discover_regions(*args, **kwargs)
