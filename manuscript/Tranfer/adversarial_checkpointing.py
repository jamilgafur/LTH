"""Checkpoint and architecture reconstruction helpers for adversarial analysis."""

from __future__ import annotations

import glob
import json
import os
import re

import torch
import torch.nn as nn

from collapse import _wrap_pools_safe, collapse_only
from pyPrune.models.ConvNetX import ConvNeXt
from pyPrune.models.InceptionNet import InceptionNet
from pyPrune.models.MobileNet import MobileNet
from pyPrune.models.RegNetX import RegNetX_400MF
from pyPrune.models.Vgg16 import VGG16
from pyPrune.models.XceptionNet import XceptionNet

MODEL_ORDER = [
    "VGG16",
    "RegNetX_400MF",
    "InceptionNet",
    "MobileNet",
    "XceptionNet",
    "ConvNeXt",
]
DATASET_ORDER = ["Cifar10", "Cifar100", "imagenet", "tinyimagenet"]


class CheckpointManager:
    """Handles checkpoint discovery and model reconstruction."""

    @staticmethod
    def get_checkpoint_path(model: str, dataset: str, kind: str):
        prefix = f"{model}_{dataset}_"
        dirs = [d for d in os.listdir(".") if os.path.isdir(d) and d.startswith(prefix)]
        if not dirs:
            return None

        target_dir = dirs[0]
        for d in dirs:
            if "epochs100_pretrain300" in d:
                target_dir = d
                break

        filename = (
            "final_JF_Control.pt"
            if kind == "Original"
            else "final_JF_Dynamic_Region_All_Combined.pt"
        )
        full_path = os.path.abspath(os.path.join(target_dir, "checkpoints", filename))
        return full_path if os.path.exists(full_path) else None

    @staticmethod
    def get_model_kwargs(model_name: str, num_classes: int, one_batch: torch.Tensor | None) -> dict:
        kwargs = {"num_classes": num_classes}
        if one_batch is not None:
            kwargs["one_batch"] = one_batch
        if model_name == "InceptionNet":
            kwargs["aux_logits"] = False
        return kwargs

    @staticmethod
    def get_discovered_regions_path(model_name: str, dataset_name: str, ckpt_path: str) -> str | None:
        run_dir = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
        base_dir = os.path.dirname(os.path.abspath(__file__))

        match = re.search(r"(epochs\d+_pretrain\d+)", run_dir)
        if match:
            epoch_tag = match.group(1)
            candidate = os.path.join(
                base_dir,
                f"{model_name}_{dataset_name}_{epoch_tag}_JF_discovered_regions.json",
            )
            if os.path.exists(candidate):
                return candidate

        matches = sorted(
            glob.glob(
                os.path.join(
                    base_dir,
                    f"{model_name}_{dataset_name}_epochs*_pretrain*_JF_discovered_regions.json",
                )
            )
        )
        return matches[0] if matches else None

    @classmethod
    def get_compression_set_for_checkpoint(cls, model_name: str, dataset_name: str, ckpt_path: str):
        json_path = cls.get_discovered_regions_path(model_name, dataset_name, ckpt_path)
        if not json_path:
            raise FileNotFoundError(
                f"No discovered regions JSON found for {model_name}/{dataset_name} at {ckpt_path}"
            )

        with open(json_path, "r") as handle:
            regions = json.load(handle)

        compression_set = regions.get("Dynamic_Region_All_Combined")
        if not compression_set:
            raise KeyError(f"Dynamic_Region_All_Combined not found in {json_path}")

        print(f"[DEBUG] Discovered regions path: {json_path}")
        print(f"[DEBUG] Collapse ranges loaded: {len(compression_set)}")
        return compression_set

    @classmethod
    def load_model(cls, model_name: str, num_classes: int, one_batch: torch.Tensor | None = None) -> nn.Module:
        mapping = {
            "VGG16": VGG16,
            "RegNetX_400MF": RegNetX_400MF,
            "ConvNeXt": ConvNeXt,
            "InceptionNet": InceptionNet,
            "MobileNet": MobileNet,
            "XceptionNet": XceptionNet,
        }
        if model_name not in mapping:
            raise ValueError(f"Unsupported model: {model_name}")
        kwargs = cls.get_model_kwargs(model_name, num_classes, one_batch)
        return mapping[model_name](**kwargs)

    @classmethod
    def build_model_for_checkpoint(
        cls,
        model_name: str,
        dataset_name: str,
        kind: str,
        num_classes: int,
        one_batch: torch.Tensor,
        ckpt_path: str,
        device: str,
    ) -> nn.Module:
        model = cls.load_model(model_name, num_classes, one_batch=one_batch)
        _wrap_pools_safe(model)

        if kind == "Finetuned":
            compression_set = cls.get_compression_set_for_checkpoint(model_name, dataset_name, ckpt_path)
            print(f"[DEBUG] Rebuilding collapsed architecture for {model_name} ({dataset_name})")
            model = collapse_only(
                model=model,
                compression_set=compression_set,
                input_shape=one_batch.shape,
                device=device,
                dry_run=False,
                debug=False,
                handle_skips=True,
            )
        else:
            model = model.to(device)

        return model

    @classmethod
    def discover_checkpoints(cls):
        entries = []
        print("\n" + "=" * 70)
        print("[CHECKPOINT DISCOVERY] Scanning for available models and datasets...")
        print("=" * 70)

        for model in MODEL_ORDER:
            for dataset in DATASET_ORDER:
                for kind in ("Finetuned", "Original"):
                    path = cls.get_checkpoint_path(model, dataset, kind)
                    if path:
                        print(
                            f"  [FOUND] Model: {model:<15} | Dataset: {dataset:<10} | Kind: {kind:<10}"
                        )
                        entries.append((model, dataset, kind, path))

        print("=" * 70)
        print(f"[CHECKPOINT DISCOVERY] Total valid checkpoint pairs found: {len(entries)}")
        print("=" * 70 + "\n")
        return entries
