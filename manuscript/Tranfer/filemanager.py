import os
import json
import torch
from pyPrune.models.Vgg16 import VGG16
# -------------------------
# Checkpoint and Naming
# -------------------------
def get_checkpoint_filename(workflow, exp_name, model_type, epochs):
    exp_tag = exp_name.replace(" ", "_").replace("-", "_")
    filename = f"{workflow}_{exp_tag}_{model_type}_epochs{epochs}.pth"

    return filename

import os
import json

def save_metrics_json(workflow, experiment, metrics_dict, base_dir="metrics"):
    """
    Save metrics for a given workflow and experiment into a JSON file.

    Args:
        workflow (str): Name of the workflow.
        experiment (str): Experiment identifier.
        metrics_dict (dict): Dictionary containing metrics to save (e.g., accuracy, loss, inference_time, etc).
        base_dir (str, optional): Directory to save metrics JSON files. Defaults to "metrics".
    """
    json_path = os.path.join(base_dir, f"{workflow}_metrics.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    if workflow not in data:
        data[workflow] = {}

    data[workflow][experiment] = metrics_dict

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[✓] Saved metrics to {json_path}")

def load_metrics_json(workflow, experiment):
    json_path = f"metrics/{workflow}_metrics.json"
    if not os.path.exists(json_path):
        return [], []

    with open(json_path, "r") as f:
        data = json.load(f)
    if workflow in data and experiment in data[workflow]:
        return data[workflow][experiment]["accuracy"], data[workflow][experiment]["loss"]
    return [], []

def load_model_from_checkpoint(ckpt_path, collapse_range, device, model_class=VGG16, model_kwargs=None):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model_kwargs = model_kwargs or {}
    model = model_class(**model_kwargs)
    import pdb; pdb.set_trace()
    if collapse_range is not None:
        model = collapse_block(model, *collapse_range)

    sd = torch.load(ckpt_path)['model']
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model
