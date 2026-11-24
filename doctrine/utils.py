"""
NonNULL Framework V1.0 - System Utilities, Multi-EOS Helper & Hierarchical Config Loader
File: doctrine/utils.py
"""

import os
import random
from typing import Any, Dict, List, Tuple
import yaml
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Locks all random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"--- [UTILS] Seed locked at: {seed} ---")


def get_eos_token_ids(tokenizer: Any) -> List[int]:
    """
    Collects all valid end-of-turn / end-of-sequence token IDs across model families
    (Qwen <|im_end|>, Llama 3.1/3.2 <|eot_id|> & <|end_of_text|>, generic eos_token_id).
    """
    ids: List[int] = []
    eos_attr = getattr(tokenizer, "eos_token_id", None)
    if eos_attr is not None:
        if isinstance(eos_attr, (list, tuple)):
            ids.extend(int(x) for x in eos_attr if x is not None)
        else:
            ids.append(int(eos_attr))

    for tok in ("<|eot_id|>", "<|im_end|>", "<|end_of_text|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", -1):
                if tid not in ids:
                    ids.append(tid)
        except Exception:
            pass
    return ids


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Counts trainable parameters (VAN / Adapter) vs total parameters (VU / Backbone)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges modular YAML configuration dictionaries."""
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Loads master YAML config and automatically resolves Hydra-style 'defaults'
    sub-configs (configs/model/*.yaml, configs/meta_controller/*.yaml, configs/trainer/*.yaml).
    """
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        master_cfg = yaml.safe_load(f) or {}

    base_dir = os.path.dirname(os.path.abspath(config_path))
    combined: Dict[str, Any] = {}

    defaults = master_cfg.get("defaults", [])
    if isinstance(defaults, list):
        for item in defaults:
            if isinstance(item, dict):
                for folder, filename in item.items():
                    sub_path = os.path.join(base_dir, str(folder), f"{filename}.yaml")
                    if os.path.exists(sub_path):
                        with open(sub_path, "r", encoding="utf-8") as sf:
                            sub_cfg = yaml.safe_load(sf) or {}
                        combined = _deep_merge(combined, sub_cfg)

    return _deep_merge(combined, master_cfg)
