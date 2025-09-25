import torch
import torch.nn as nn
from collections import OrderedDict
from utils import count_trainable_params, layer_stats

# ===============================
# Layer Collapse Helpers
# ===============================

def _find_layer_indices(named_layers, start_layer_name, end_layer_name):
    start_idx = end_idx = None
    for i, (name, _) in enumerate(named_layers):
        if name == start_layer_name:
            start_idx = i
        if name == end_layer_name:
            end_idx = i
    return start_idx, end_idx

def _simulate_input(model, section_name, start_idx, input_shape):
    dummy_input = torch.randn(input_shape).to(next(model.parameters()).device)
    x = dummy_input

    if section_name == "features":
        for layer in list(model.features.children())[:start_idx]:
            x = layer(x)
    else:
        for layer in model.features:
            x = layer(x)
        x = torch.flatten(x, 1)
        for layer in list(model.classifier.children())[:start_idx]:
            x = layer(x)

    return dummy_input, x

def _build_collapsed_block(layer_type, in_features, out_features, output_shape):
    if layer_type == nn.Conv2d:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.AdaptiveAvgPool2d((1, 1))
        )
    elif layer_type == nn.Linear:
        flattened_input = in_features * output_shape[-1] * output_shape[-2]
        return nn.Linear(flattened_input, out_features)
    else:
        raise NotImplementedError("Unsupported layer type for collapsing.")

def _replace_layers(named_layers, start_idx, end_idx, new_block):
    new_layers = []
    for i, (name, layer) in enumerate(named_layers):
        if i == start_idx:
            new_layers.append((f"collapsed_{named_layers[start_idx][0]}_to_{named_layers[end_idx][0]}", new_block))
        elif start_idx < i <= end_idx:
            continue  # skip collapsed layers
        elif i > end_idx and isinstance(layer, nn.MaxPool2d):
            print(f"Removing MaxPool2d after collapsed block: {name}")
            continue  # remove dangerous MaxPool2d
        else:
            new_layers.append((name, layer))
    return nn.Sequential(OrderedDict(new_layers))


# ===============================
# Main Collapse Function
# ===============================

def _collapse_block(model, start_layer_name, end_layer_name, input_shape):
    print(f"\nCollapsing layers from '{start_layer_name}' to '{end_layer_name}'...")
    containers = {
        "features": model.features,
        "classifier": model.classifier,
    }

    for section_name, container in containers.items():
        named_layers = list(container.named_children())
        start_idx, end_idx = _find_layer_indices(named_layers, start_layer_name, end_layer_name)

        if start_idx is not None and end_idx is not None:
            assert start_idx <= end_idx, "Start index must be <= end index"

            full_block = named_layers[start_idx:end_idx + 1]
            selected_layers = [layer for _, layer in full_block if isinstance(layer, (nn.Conv2d, nn.Linear))]

            if len(selected_layers) < 2:
                raise ValueError("Need at least 2 Conv2d or Linear layers to collapse.")
            
            layer_type = type(selected_layers[0])
            if not all(isinstance(l, layer_type) for l in selected_layers):
                raise ValueError("Cannot collapse mixed layer types.")

            dummy_input, x = _simulate_input(model, section_name, start_idx, input_shape)

            in_features = x.shape[1] if layer_type == nn.Linear else selected_layers[0].in_channels
            for layer in selected_layers:
                x = layer(x)
            out_features = x.shape[1] if layer_type == nn.Linear else selected_layers[-1].out_channels

            print(f"Input shape: {dummy_input.shape} → Output shape: {x.shape}")

            collapsed_block = _build_collapsed_block(layer_type, in_features, out_features, x.shape)
            updated_container = _replace_layers(named_layers, start_idx, end_idx, collapsed_block)

            if section_name == "features":
                model.features = updated_container
            else:
                model.classifier = updated_container

            print(f"Collapsed {section_name} layers {start_layer_name} → {end_layer_name}")
            print(f"New trainable params: {count_trainable_params(model)}")
            return model

    print(f"New structure:\n{layer_stats(model)}")
    raise ValueError(f"Layer names '{start_layer_name}' or '{end_layer_name}' not found.")


def collapse_only(model_weights_1, compression_set, model_class, model_kwargs=None, input_shape=(1, 3, 32, 32)):
    model_kwargs = model_kwargs or {}

    model = model_class(**model_kwargs)
    checkpoint = torch.load(model_weights_1, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    for start, end in compression_set:
        model = _collapse_block(model, start, end, input_shape)

    return model
