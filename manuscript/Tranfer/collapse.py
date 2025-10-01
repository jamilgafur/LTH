import torch
import torch.nn as nn
from collections import OrderedDict
from utils import count_trainable_params, layer_stats

# ===============================
# Layer Collapse Helpers
# ===============================

def get_layer(model, layer_name):
    """
    Dynamically access a layer in the model based on the full dot-separated name.
    """
    layer_parts = layer_name.split('.')
    layer = model
    for part in layer_parts:
        layer = getattr(layer, part)
    return layer

def compression_set(layers):
    """
    Example of compressing layers. Replace this with actual compression logic.
    """
    for layer in layers:
        print(f"Compressing layer: {layer}")
        # Add compression logic here (e.g., pruning, quantization)

def collapse_only(model_weights_1, compression_set, model_class, model_kwargs=None, input_shape=(1, 3, 32, 32), device='cpu'):
    model_kwargs = model_kwargs or {}
    num_classes = model_kwargs.get('num_classes', 200)

    print(f"Loading model with weights from '{model_weights_1}'...")
    model = model_class(**model_kwargs)
    checkpoint = torch.load(model_weights_1, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    for start, end in compression_set:
        print(f"\n--- Starting collapse for block: {start} to {end} ---")
        model = _collapse_block(model, start, end, input_shape)

    print("\n--- Adjusting classifier input features AFTER collapsing all blocks ---")
    adjust_classifier_input_features(model, input_shape, num_classes=num_classes, device=device)

    print(f"Collapse complete. Total trainable params: {count_trainable_params(model)}")
    print(f"Final model structure:\n{layer_stats(model)}")

    return model

def _collapse_block(model, start_layer_name, end_layer_name, input_shape):
    print(f"\nCollapsing layers from '{start_layer_name}' to '{end_layer_name}'...")

    # Gather available layers for collapsing
    build_layer_names = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.ReLU, nn.AdaptiveAvgPool2d)):
            build_layer_names.append(name)
    print(f"Available layers for collapsing: {build_layer_names}")

    start_container_name, start_subname = _get_container_and_subname(start_layer_name)
    end_container_name, end_subname = _get_container_and_subname(end_layer_name)
    
    # Make sure both layers are within the same container
    if start_container_name != end_container_name:
        raise ValueError(f"Start and end layers must be in the same container. Found: {start_container_name}, {end_container_name}")

    container = get_layer(model, start_container_name)
    named_layers = list(container.named_children())

    # Find layer indices for start and end within container
    start_idx, end_idx = _find_layer_indices(named_layers, start_subname, end_subname)
    
    if start_idx is not None and end_idx is not None:
        assert start_idx <= end_idx, "Start index must be <= end index"
        print(f"Collapsing in section '{start_container_name}' from index {start_idx} to {end_idx}")

        full_block = named_layers[start_idx:end_idx + 1]
        selected_layers = [layer for _, layer in full_block if isinstance(layer, (nn.Conv2d, nn.Linear))]

        if len(selected_layers) < 2:
            raise ValueError("Need at least 2 Conv2d or Linear layers to collapse.")

        layer_type = type(selected_layers[0])
        if not all(isinstance(l, layer_type) for l in selected_layers):
            raise ValueError("Cannot collapse mixed layer types.")

        dummy_input, x = _simulate_input(model, start_container_name, start_idx, input_shape)
        print(f"  Simulated input shape before collapsing block: {x.shape}")

        in_features = x.shape[1] if layer_type == nn.Linear else selected_layers[0].in_channels
        print(f"  in_features determined as: {in_features}")

        # Forward through each selected layer to track output shape
        for i, layer in enumerate(selected_layers):
            x = layer(x)
            print(f"    After selected layer {i} ({type(layer).__name__}) output shape: {x.shape}")

        out_features = x.shape[1] if layer_type == nn.Linear else selected_layers[-1].out_channels
        print(f"  out_features determined as: {out_features}")

        # Build the collapsed block
        collapsed_block = _build_collapsed_block(layer_type, in_features, out_features, x.shape)
        updated_container = _replace_layers(named_layers, start_idx, end_idx, collapsed_block)

        # Dynamically update container in model (instead of hardcoding features/classifier)
        _update_container(model, start_container_name, updated_container)

        print(f"Collapsed {start_container_name} layers {start_layer_name} → {end_layer_name}")
        print(f"New trainable params: {count_trainable_params(model)}")
        print(f"Model structure after collapse:\n{layer_stats(model)}")
        return model

    raise ValueError(f"Layer names '{start_layer_name}' or '{end_layer_name}' not found.")

def _simulate_input(model, section_name, start_idx, input_shape):
    dummy_input = torch.randn(input_shape).to(next(model.parameters()).device)
    print(f"Simulating input for section '{section_name}' with shape {input_shape}...")
    x = dummy_input

    container = get_layer(model, section_name)

    # Forward through container layers before start_idx to get input to collapsed block
    for i, layer in enumerate(list(container.children())[:start_idx]):
        x = layer(x)
        print(f"  After {section_name} layer {i} output shape: {x.shape}")

    return dummy_input, x

def _build_collapsed_block(layer_type, in_features, out_features, output_shape):
    print(f"Building collapsed block of type {layer_type.__name__} with in_features={in_features}, out_features={out_features}, output_shape={output_shape}")
    if layer_type == nn.Conv2d:
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    elif layer_type == nn.Linear:
        in_features_dynamic = output_shape[1]
        print(f"  Linear collapsed block in_features_dynamic = {in_features_dynamic}")
        block = nn.Linear(in_features_dynamic, out_features)
    else:
        raise NotImplementedError(f"Unsupported layer type for collapsing: {layer_type}")
    return block

def _replace_layers(named_layers, start_idx, end_idx, new_block):
    print(f"Replacing layers {start_idx} to {end_idx} with collapsed block...")
    new_layers = []
    for i, (name, layer) in enumerate(named_layers):
        if i == start_idx:
            new_name = f"collapsed_{named_layers[start_idx][0]}_to_{named_layers[end_idx][0]}"
            print(f"  Inserting new block as '{new_name}'")
            new_layers.append((new_name, new_block))
        elif start_idx < i <= end_idx:
            print(f"  Removing layer '{name}' at index {i}")
            continue
        elif i > end_idx and isinstance(layer, nn.MaxPool2d):
            print(f"  Removing MaxPool2d '{name}' at index {i} after collapsed block")
            continue
        else:
            new_layers.append((name, layer))
    return nn.Sequential(OrderedDict(new_layers))

def adjust_classifier_input_features(model, input_shape, num_classes=200, device='cpu'):
    model.eval()  # Switch entire model to eval mode

    with torch.no_grad():
        model.to(device)
        dummy_input = torch.randn(input_shape).to(device)
        features_output = model.features(dummy_input)
        flattened_size = features_output.view(features_output.size(0), -1).size(1)

        print(f"[DEBUG] Flattened features size: {flattened_size}")

        model.classifier = nn.Sequential(
            nn.Linear(flattened_size, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
    model.train()  # Switch back to train mode after adjustment

def _get_container_and_subname(layer_name):
    """
    Extracts the container (e.g., 'features', 'classifier' or any module path) and the subname 
    (e.g., 'conv1') from the full layer name (e.g., 'features.conv1').
    """
    layer_parts = layer_name.split('.')
    container = '.'.join(layer_parts[:-1])  # All but last part (parent module path)
    subname = layer_parts[-1]               # Last part (layer/module name)
    return container, subname

def _find_layer_indices(named_layers, start_layer_name, end_layer_name):
    start_idx = end_idx = None
    print(f"Finding indices for layers '{start_layer_name}' to '{end_layer_name}'...")

    for i, (name, _) in enumerate(named_layers):
        if name == start_layer_name:
            start_idx = i
            print(f"  Found start layer '{start_layer_name}' at index {start_idx}")
        if name == end_layer_name:
            end_idx = i
            print(f"  Found end layer '{end_layer_name}' at index {end_idx}")

    if start_idx is None or end_idx is None:
        print(f"Warning: Could not find one or both layer names '{start_layer_name}', '{end_layer_name}'")

    return start_idx, end_idx

def _update_container(model, container_path, new_container):
    """
    Replace the module at `container_path` in `model` with `new_container`.
    container_path is a dot-separated string specifying nested modules.
    """
    parts = container_path.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_container)
