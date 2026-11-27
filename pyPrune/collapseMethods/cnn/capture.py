# capture.py -- observing a model: parameter/zero statistics and capturing
# the activations that enter a layer (simulation hooks).
import logging
from typing import Tuple

import torch
import torch.nn as nn

from .containers import get_layer

logger = logging.getLogger(__name__)


def count_zeros(tensor) -> int:
    return torch.sum(tensor == 0).item()


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def layer_stats(model) -> str:
    """Layer-wise zero-parameter statistics as formatted text."""
    lines = ["Layer-wise zero parameter stats:"]
    for name, param in model.named_parameters():
        if param.requires_grad:
            zeros = count_zeros(param)
            total = param.numel()
            lines.append(f"{name}: {zeros}/{total} zeros ({100 * zeros / total:.2f}%)")
    return "\n".join(lines)


def _simulate_input_hook(model: nn.Module, target_layer_path: str, input_shape: Tuple[int, ...], device='cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run a single forward with a dummy input and capture the activation that is
    the *input to* the target layer (registered as forward hook on the target layer).
    Returns (dummy_input, captured_activation).
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_shape).to(device)

    target_module = get_layer(model, target_layer_path)

    captured = {}
    def hook(module, inp, out):
        # store a detached copy of the input to the target module
        captured['in'] = inp[0].detach()
    handle = target_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(dummy_input)
    finally:
        handle.remove()
    if 'in' not in captured:
        raise RuntimeError(f"Failed to capture activation at {target_layer_path}.")
    return dummy_input, captured['in']


def _capture_preblock_activation(model, start_layer_name, input_shape, conv_layers, layer_type, device, debug):
    logger.debug(f"Attempting to capture activation before '{start_layer_name}' using simulation hook...")
    try:
        dummy_input, x = _simulate_input_hook(model, start_layer_name, input_shape, device=device)
        if debug:
            logger.debug(f"Successfully captured activation → shape: {tuple(x.shape)}")
    except Exception as e:
        logger.warning(f"Hook failed: {e}")
        logger.warning(f"Falling back to dummy tensor initialization.")
        if layer_type == nn.Conv2d:
            in_ch = conv_layers[0].in_channels if hasattr(conv_layers[0], 'in_channels') else input_shape[1]
            H, W = input_shape[-2], input_shape[-1]
            x = torch.randn(1, in_ch, H, W, device=device)
        else:
            in_feat = conv_layers[0].in_features if hasattr(conv_layers[0], 'in_features') else input_shape[1]
            x = torch.randn(1, in_feat, device=device)
        logger.debug(f"Created fallback tensor of shape {tuple(x.shape)}")

    pre_params = count_trainable_params(model)
    if debug:
        logger.debug(f"Total trainable parameters before collapse: {pre_params:,}")

    logger.debug(f"Activation capture complete.")
    return x, pre_params
