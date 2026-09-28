from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
from tqdm import tqdm
from cider.utils import get_device, safe_torch_load, set_seed
from cider.sampling import (
    generate_ids_batch,
    load_fp_model,
    load_generator,
    soft_tanimoto,
)
from cider.training.fingerprint_diffusion import (
    FingerprintDenoiser,
    sample_fp as sample_diff_fp,
)
from cider.training.fingerprint import load_or_build_fps, morgan_bits
from cider.training.forward_generator import (
    canonicalize,
    ids_to_smiles,
    mol_from_smiles,
    mol_props,
    scaffold_smiles,
    tanimoto,
)

RDLogger.DisableLog("rdApp.*")


def load_diff_model(path: str, cache: dict, device):
    ckpt = safe_torch_load(path, map_location=device)
    a = ckpt["args"]
    cfg = cache["config"]
    model = FingerprintDenoiser(
        cfg["y_dim"],
        cfg["n_cell"],
        cfg["n_time"],
        a["fp_dim"],
        a["d_model"],
        a["depth"],
        a["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return (model, int(a["fp_dim"]), a.get("target", "x0"))


@torch.no_grad()
def ctp_nn_fps(cache, fps, idx: int, device, k: int):
    train_idx_cpu = torch.as_tensor(cache["splits"]["train"], dtype=torch.long)
    train_x = cache["X"][train_idx_cpu].float()
    query = cache["X"][idx].float().view(1, -1)
    score = (
        torch.nn.functional.normalize(query, dim=1)
        @ torch.nn.functional.normalize(train_x, dim=1).T
    )
    pos = torch.topk(score.squeeze(0), k=k).indices
    nn_idx = train_idx_cpu[pos]
    return fps[nn_idx].to(device)


def hard_tanimoto_to_matrix(bits: np.ndarray, mat: torch.Tensor) -> float:
    b = torch.as_tensor(bits, dtype=mat.dtype, device=mat.device)
    inter = (mat * b[None, :]).sum(dim=1)
    denom = mat.sum(dim=1) + b.sum() - inter
    return float((inter / denom.clamp_min(1e-06)).max().item())
