import torch
import torch.nn as nn
from collections import OrderedDict
from utils import count_trainable_params
from typing import Optional

def _is_int_str(s):
    try:
        int(s)
        return True
    except:
        return False

def get_layer(model, layer_name):
    if layer_name == "":
        return model
    layer_parts = layer_name.split('.')
    layer = model
    for part in layer_parts:
        if _is_int_str(part):
            layer = layer[int(part)]
        else:
            layer = getattr(layer, part)
    return layer

def _get_container_and_subname(layer_name):
    if layer_name == "":
        return "", ""
    layer_parts = layer_name.split('.')
    if len(layer_parts) == 1:
        return "", layer_parts[0]
    container = '.'.join(layer_parts[:-1])
    subname = layer_parts[-1]
    return container, subname

def _find_layer_indices(named_layers, start_layer_name, end_layer_name):
    start_idx = end_idx = None
    for i, (name, _) in enumerate(named_layers):
        if name == start_layer_name:
            start_idx = i
        if name == end_layer_name:
            end_idx = i
    return start_idx, end_idx

def _update_container(model, container_path, new_container):
    parts = container_path.split('.')
    parent = model
    for part in parts[:-1]:
        if _is_int_str(part):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if _is_int_str(last):
        parent[int(last)] = new_container
    else:
        setattr(parent, last, new_container)

def disable_inplace_relu(model):
    to_replace = [name for name, module in list(model.named_modules())
                  if isinstance(module, nn.ReLU) and getattr(module, 'inplace', False)]
    for name in to_replace:
        container_name, subname = _get_container_and_subname(name)
        parent = get_layer(model, container_name) if container_name != "" else model
        new_relu = nn.ReLU(inplace=False)
        if _is_int_str(subname):
            parent[int(subname)] = new_relu
        else:
            setattr(parent, subname, new_relu)

def collapse_only(model_weights_1, compression_set, model_class, model_kwargs=None,
                  input_shape=(1, 3, 32, 32), device='cpu'):
    model_kwargs = model_kwargs or {}
    model = model_class(**model_kwargs)
    checkpoint = torch.load(model_weights_1, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    for start, end in compression_set:
        model = _collapse_block(model, start, end, input_shape, device=device)

    disable_inplace_relu(model)
    model.to(device)
    return model

def _collapse_block(model, start_layer_name, end_layer_name, input_shape, device='cpu'):
    start_container_name, start_subname = _get_container_and_subname(start_layer_name)
    end_container_name, end_subname = _get_container_and_subname(end_layer_name)
    container = get_layer(model, start_container_name)
    named_layers = list(container.named_children())
    start_idx, end_idx = _find_layer_indices(named_layers, start_subname, end_subname)
    if start_idx is None or end_idx is None:
        raise ValueError(f"Layers '{start_layer_name}' or '{end_layer_name}' not found.")

    # Forward through pre-block layers to get real input
    dummy_input = torch.randn((1, *input_shape[1:]), device=device) if len(input_shape) == 4 else torch.randn(1, *input_shape, device=device)
    x = dummy_input
    for i in range(start_idx):
        x = named_layers[i][1](x)
    in_channels = x.shape[1]

    # Forward through block to get output
    for i in range(start_idx, end_idx + 1):
        x = named_layers[i][1](x)
    out_channels = x.shape[1]
    H, W = x.shape[-2:]

    last_layer = named_layers[end_idx][1]
    collapsed_layers = []

    if isinstance(last_layer, nn.Conv2d):
        # 1x1 conv to match last layer out_channels
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=last_layer.out_channels,
            kernel_size=1,
            stride=last_layer.stride,
            padding=0,
            bias=last_layer.bias is not None
        )
        collapsed_layers.append(conv)
        # preserve BN/ReLU at the end of the block if present
        if isinstance(last_layer, nn.BatchNorm2d):
            collapsed_layers.append(nn.BatchNorm2d(last_layer.out_channels))
        elif isinstance(last_layer, nn.ReLU):
            collapsed_layers.append(nn.ReLU(inplace=False))
        # always add adaptive pooling to match classifier input
        collapsed_layers.append(nn.AdaptiveAvgPool2d((H, W)))

    elif isinstance(last_layer, nn.Linear):
        collapsed_layers.append(nn.Linear(in_features=in_channels, out_features=last_layer.out_features))

    # Replace old layers with new sequential
    updated_layers = named_layers[:start_idx] + [(f"collapsed_{start_idx}", nn.Sequential(*collapsed_layers))] + named_layers[end_idx + 1:]
    collapsed_seq = nn.Sequential(OrderedDict(updated_layers))
    _update_container(model, start_container_name, collapsed_seq)
    model.to(device)
    return model
