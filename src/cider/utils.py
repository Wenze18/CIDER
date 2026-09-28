from __future__ import annotations
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def safe_torch_load(path: str | Path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def cosine_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-08
) -> torch.Tensor:
    pred_n = pred / (pred.norm(dim=1, keepdim=True) + eps)
    target_n = target / (target.norm(dim=1, keepdim=True) + eps)
    return 1.0 - (pred_n * target_n).sum(dim=1).mean()


@torch.no_grad()
def batch_cosine(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-08) -> float:
    pred_n = pred / (pred.norm(dim=1, keepdim=True) + eps)
    target_n = target / (target.norm(dim=1, keepdim=True) + eps)
    return float((pred_n * target_n).sum(dim=1).mean().item())


@torch.no_grad()
def batch_pearson(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-08
) -> float:
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    target_c = target - target.mean(dim=1, keepdim=True)
    num = (pred_c * target_c).sum(dim=1)
    den = pred_c.norm(dim=1) * target_c.norm(dim=1) + eps
    return float((num / den).mean().item())


class ResponseDataset(Dataset):

    def __init__(self, cache: dict, split: str):
        if split not in cache["splits"]:
            raise ValueError(
                f"Unknown split {split}. Available: {list(cache['splits'])}"
            )
        self.idx = torch.as_tensor(cache["splits"][split], dtype=torch.long)
        self.X = cache["X"]
        self.dose = cache["dose"]
        self.cell = cache["cell"]
        self.time = cache["time"]
        self.tokens = cache["tokens"]
        self.smiles = cache.get("smiles", None)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = int(self.idx[i])
        item = {
            "ctp": self.X[j],
            "dose": self.dose[j],
            "cell": self.cell[j],
            "time": self.time[j],
            "tokens": self.tokens[j],
            "row_index": torch.tensor(j, dtype=torch.long),
        }
        return item


def make_loader(
    cache: dict, split: str, batch_size: int, shuffle: bool, num_workers: int = 0
) -> DataLoader:
    ds = ResponseDataset(cache, split)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


def top_k_top_p_filtering(
    logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0
) -> torch.Tensor:
    logits = logits.clone()
    if top_k and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_values = values[..., -1, None]
        logits = torch.where(
            logits < min_values, torch.full_like(logits, -float("inf")), logits
        )
    if top_p and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        sorted_indices_to_remove = cumulative > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -float("inf"))
    return logits
