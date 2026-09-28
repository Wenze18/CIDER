from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
from cider.utils import get_device, safe_torch_load, set_seed
from cider.utils import top_k_top_p_filtering
from cider.training.fingerprint import FingerprintPredictor, morgan_bits
from cider.training.forward_generator import (
    RetrievalIndex,
    canonicalize,
    forward_model,
    ids_to_smiles,
    make_model_from_checkpoint,
    mol_from_smiles,
    mol_props,
    scaffold_smiles,
    tanimoto,
)

RDLogger.DisableLog("rdApp.*")


def morgan_fp(mol):
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def maccs_fp(mol):
    if mol is None:
        return None
    return MACCSkeys.GenMACCSKeys(mol)


def load_fp_model(path: str, cache: dict, device):
    ckpt = safe_torch_load(path, map_location=device)
    args = ckpt["args"]
    cfg = cache["config"]
    model = FingerprintPredictor(
        cfg["y_dim"],
        cfg["n_cell"],
        cfg["n_time"],
        args["d_model"],
        args["fp_dim"],
        args["dropout"],
        args["depth"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return (model, int(args["fp_dim"]))


def load_generator(path: str, cache: dict, device):
    ckpt = safe_torch_load(path, map_location=device)
    if ckpt.get("method") in {"fame", "gex2sgen", "transgem", "gxrnn"}:
        from cider.training.generation_baselines import make_model_from_lit_checkpoint

        model = make_model_from_lit_checkpoint(ckpt).to(device)
    elif ckpt.get("method") == "retrieval_xattn":
        from cider.training.retrieval_attention import RetrievalCrossAttentionGenerator

        m = ckpt["model_args"]
        model = RetrievalCrossAttentionGenerator(
            y_dim=m["y_dim"],
            vocab_size=m["vocab_size"],
            pad_id=m["pad_id"],
            n_cell=m["n_cell"],
            n_time=m["n_time"],
            max_len=m["max_len"],
            d_model=m["d_model"],
            nhead=m["nhead"],
            num_layers=m["num_layers"],
            dropout=m["dropout"],
            noise_dim=m.get("noise_dim", 64),
            cond_depth=m.get("cond_depth", 3),
            retrieval_k=m.get("retrieval_k", 4),
            cond_tokens=m.get("cond_tokens", 4),
            retrieval_len=m.get("retrieval_len", m["max_len"]),
        ).to(device)
    else:
        model = make_model_from_checkpoint(ckpt).to(device)
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as exc:
        if all(
            (
                k.startswith("fp_head.")
                for k in ckpt["model"]
                if k not in model.state_dict()
            )
        ):
            model.load_state_dict(ckpt["model"], strict=False)
        else:
            raise exc
    model.eval()
    retriever = None
    retrieval_k = 0
    if ckpt.get("method") in {"retrieval_adaln", "retrieval_xattn"}:
        retriever = RetrievalIndex(cache, device)
        retrieval_k = int(ckpt["model_args"].get("retrieval_k", 4))
    return (model, ckpt, retriever, retrieval_k)


def soft_tanimoto(prob: torch.Tensor, bits: np.ndarray) -> float:
    b = torch.as_tensor(bits, dtype=prob.dtype, device=prob.device)
    inter = (prob * b).sum()
    denom = prob.sum() + b.sum() - inter
    return float((inter / denom.clamp_min(1e-06)).item())


@torch.no_grad()
def generate_ids_batch(
    model,
    ctp,
    dose,
    cell,
    time,
    cfg,
    n: int,
    temperature: float,
    top_k: int,
    top_p: float,
    fixed_zero_noise: bool,
    retrieved_tokens=None,
):
    ctp = ctp.repeat(n, 1)
    dose = dose.repeat(n)
    cell = cell.repeat(n)
    time = time.repeat(n)
    if retrieved_tokens is not None:
        retrieved_tokens = retrieved_tokens.repeat(n, 1, 1)
    ids = torch.full((n, 1), int(cfg["bos_id"]), dtype=torch.long, device=ctp.device)
    finished = torch.zeros(n, dtype=torch.bool, device=ctp.device)
    for _ in range(cfg["max_len"] - 1):
        if retrieved_tokens is not None and hasattr(model, "make_memory"):
            z = None
            if fixed_zero_noise and hasattr(model, "noise_dim"):
                z = torch.zeros(
                    ctp.size(0),
                    int(model.noise_dim),
                    device=ctp.device,
                    dtype=ctp.dtype,
                )
            logits = model(ctp, dose, cell, time, ids, retrieved_tokens, z=z)
        else:
            logits = forward_model(
                model,
                ctp,
                dose,
                cell,
                time,
                ids,
                fixed_zero_noise,
                use_prior=True,
                retrieved_tokens=retrieved_tokens,
            )
        nxt_logits = logits[:, -1, :] / max(temperature, 1e-06)
        nxt_logits[:, cfg["pad_id"]] = -float("inf")
        nxt_logits[:, cfg["unk_id"]] = -float("inf")
        filt = top_k_top_p_filtering(nxt_logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(filt, dim=-1)
        nxt = torch.multinomial(probs, 1).squeeze(1)
        nxt = torch.where(finished, torch.full_like(nxt, int(cfg["eos_id"])), nxt)
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        finished |= nxt.eq(int(cfg["eos_id"]))
        if bool(finished.all().item()):
            break
    return ids.detach().cpu().tolist()
