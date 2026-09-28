from __future__ import annotations
from cider.paths import resolve_path
import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm
from cider.reverse import predict_tabular_group, predict_token, read_manifest
from cider.utils import (
    batch_cosine,
    batch_pearson,
    get_device,
    safe_torch_load,
    set_seed,
)
from cider.sampling import generate_ids_batch, load_fp_model, load_generator
from cider.chemistry import (
    build_train_reference,
    canonicalize_smiles,
    mol_props,
    scaffold_smiles,
    tanimoto_between,
    max_tanimoto_to_train,
)
from cider.training.fingerprint import load_or_build_fps
from cider.training.forward_generator import ids_to_smiles
from cider.training.structural_ranker import (
    StructuralReranker,
    make_row_candidates,
    parse_repeat_seeds,
    train_set_canonical,
)
from cider.proposal import (
    exploratory_score,
    load_versions,
    source_smiles,
    source_tanimoto,
)

RDLogger.DisableLog("rdApp.*")


def parse_indices(
    cache: dict, split: str, rows: int, sample_rows: bool, row_seed: int
) -> list[int]:
    indices = list(map(int, cache["splits"][split]))
    if sample_rows:
        rng = np.random.default_rng(row_seed)
        n = min(rows, len(indices))
        return [int(x) for x in rng.choice(indices, size=n, replace=False)]
    return indices[:rows]


def rank01(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(values)
    out = np.zeros_like(values, dtype=np.float64)
    if ok.sum() == 0:
        return out
    vals = values[ok]
    order = np.argsort(vals)
    ranks = np.empty_like(vals, dtype=np.float64)
    ranks[order] = np.arange(len(vals), dtype=np.float64)
    if len(vals) > 1:
        ranks = ranks / float(len(vals) - 1)
    else:
        ranks[:] = 0.5
    if not higher_is_better:
        ranks = 1.0 - ranks
    out[ok] = ranks
    return out


def train_set_smiles(cache: dict) -> set[str]:
    out = set()
    for idx in cache["splits"]["train"]:
        smi, ok = canonicalize_smiles(cache["smiles"][int(idx)])
        if ok:
            out.add(smi)
    return out


@torch.no_grad()
def predict_response_ensemble(
    rows: list[dict], cache: dict, manifest: dict, batch_size: int, device
) -> torch.Tensor:
    weights = manifest["weights"]
    tabular_pred = predict_tabular_group(
        manifest["tabular_checkpoints"], rows, cache, batch_size, device
    )
    transformer_pred = predict_token(
        manifest["token_checkpoints"]["transformer"], rows, batch_size, device
    )
    perceiver_pred = predict_token(
        manifest["token_checkpoints"]["perceiver"], rows, batch_size, device
    )
    return (
        float(weights["tabular"]) * tabular_pred
        + float(weights["transformer"]) * transformer_pred
        + float(weights["perceiver"]) * perceiver_pred
    )


def reverse_metric_arrays(
    pred: torch.Tensor, target: torch.Tensor, cos_weight: float
) -> dict[str, np.ndarray]:
    target = target.to(pred.dtype)
    mse = torch.mean((pred - target) ** 2, dim=1)
    cosine = torch.nn.functional.cosine_similarity(pred, target, dim=1, eps=1e-08)
    pc = pred - pred.mean(dim=1, keepdim=True)
    tc = target - target.mean(dim=1, keepdim=True)
    pearson = torch.nn.functional.cosine_similarity(pc, tc, dim=1, eps=1e-08)
    cycle = mse + cos_weight * (1.0 - cosine)
    return {
        "reverse_pearson": pearson.cpu().numpy(),
        "reverse_cosine": cosine.cpu().numpy(),
        "reverse_mse": mse.cpu().numpy(),
        "cycle_score": cycle.cpu().numpy(),
    }


def reverse_rows_for_candidates(
    cache: dict, idx: int, smiles_list: list[str]
) -> list[dict]:
    out = []
    for j, smi in enumerate(smiles_list):
        out.append(
            {
                "source": "csv",
                "input_id": f"{idx}_{j}",
                "target_index": "",
                "smiles": smi,
                "dose": float(cache["dose"][idx].item()),
                "cell": int(cache["cell"][idx].item()),
                "time": int(cache["time"][idx].item()),
                "tokens": None,
            }
        )
    from cider.reverse import encode_smiles

    for row in out:
        row["tokens"] = encode_smiles(row["smiles"], cache)
    return out


def add_molecule_fields(
    row: dict,
    cache: dict,
    idx: int,
    train_can_set: set,
    train_scaffold_set: set,
    train_fps: list,
) -> dict:
    smi = row["smiles"]
    props = mol_props(smi)
    row.update(props)
    scaf = scaffold_smiles(smi)
    row["scaffold"] = scaf
    row["novel_to_train"] = bool(smi not in train_can_set)
    row["scaffold_novel_to_train"] = (
        bool(scaf not in train_scaffold_set) if scaf else False
    )
    src = source_smiles(cache, idx)
    row["source_smiles"] = src
    row["tanimoto_to_source"] = tanimoto_between(smi, src) if src else np.nan
    row["max_tanimoto_to_train"] = max_tanimoto_to_train(smi, train_fps)
    return row


@torch.no_grad()
def conservative_pool(
    args, config: dict, cache: dict, idx: int, state: dict, device
) -> list[dict]:
    cfg = cache["config"]
    candidates, _ = make_row_candidates(
        cache,
        cfg,
        idx,
        state["generators"],
        state["fp_model"],
        None,
        None,
        None,
        state["fps"],
        state["train_set"],
        state["candidate_args"],
        device,
        state["repeat_seeds"],
    )
    rows = []
    for smi, item in candidates.items():
        f = item["feature"]
        base = float(args.w_fp * f[-17].item() + args.w_nn * f[-15].item())
        if state["reranker"] is not None:
            learned = float(
                state["reranker"](f.view(1, -1).to(device)).detach().cpu().item()
            )
            base += float(args.w_learned) * learned
        rows.append({"smiles": smi, "raw_smiles": smi, "base_score": base})
    return rows


@torch.no_grad()
def exploratory_pool(
    args, config: dict, cache: dict, idx: int, state: dict, device
) -> list[dict]:
    cfg = cache["config"]
    ctp = cache["X"][idx].to(device).view(1, -1)
    dose = cache["dose"][idx].to(device).view(1)
    cell = cache["cell"][idx].to(device).view(1)
    time = cache["time"][idx].to(device).view(1)
    row_index = torch.tensor([idx], dtype=torch.long, device=device)
    retriever = state["retriever"]
    retrieval_k = state["retrieval_k"]
    retrieved_tokens = (
        retriever.retrieve_tokens(ctp, row_index, retrieval_k)
        if retriever is not None and retrieval_k > 0
        else None
    )
    generated = generate_ids_batch(
        state["generator"],
        ctp,
        dose,
        cell,
        time,
        cfg,
        args.candidates,
        args.temperature,
        args.top_k,
        args.top_p,
        args.fixed_zero_noise,
        retrieved_tokens=retrieved_tokens,
    )
    train_can = state["train_can"]
    weights = dict(config["rank_weights"])
    rows = {}
    for ids in generated:
        raw = ids_to_smiles(
            ids, cache["vocab"]["itos"], cfg["bos_id"], cfg["eos_id"], cfg["pad_id"]
        )
        can, ok = canonicalize_smiles(raw)
        if not ok or can in rows:
            continue
        props = mol_props(can)
        novel = can not in train_can
        tan_src = source_tanimoto(cache, idx, can)
        rows[can] = {
            "smiles": can,
            "raw_smiles": raw,
            "base_score": exploratory_score(props, novel, tan_src, weights),
        }
    return list(rows.values())


def setup_conservative(args, config: dict, cache: dict, device) -> dict:
    loaded = safe_torch_load(
        resolve_path(config["structural_reranker_ckpt"]), map_location="cpu"
    )
    saved_args = loaded.get("args", {})
    candidate_args = SimpleNamespace(
        fp_dim=int(config["fp_dim"]),
        nn_k=int(config["nn_k"]),
        candidates_per_generator=int(args.candidates_per_generator),
        keep=int(args.keep),
        repeat_seeds=args.repeat_seeds,
        diff_samples=2,
        diff_steps=12,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        fixed_zero_noise=args.fixed_zero_noise,
        w_fp=float(args.w_fp),
        w_nn=float(args.w_nn),
        w_learned=float(args.w_learned),
        use_bits=bool(saved_args.get("use_bits", False)),
        use_pred_fp=bool(saved_args.get("use_pred_fp", False)),
        use_nn_mean=bool(saved_args.get("use_nn_mean", False)),
        use_ctp=bool(saved_args.get("use_ctp", True)),
    )
    reranker = StructuralReranker(
        int(loaded["feature_dim"]),
        int(saved_args.get("hidden", 256)),
        int(saved_args.get("depth", 3)),
        float(saved_args.get("dropout", 0.1)),
    ).to(device)
    reranker.load_state_dict(loaded["model"])
    reranker.eval()
    return {
        "generators": [
            load_generator(resolve_path(p), cache, device)
            for p in config["generator_ckpts"]
        ],
        "fp_model": load_fp_model(resolve_path(config["fp_ckpt"]), cache, device)[0],
        "fps": load_or_build_fps(
            cache, int(config["fp_dim"]), resolve_path(config["fp_cache"]), fp_workers=8
        )[0],
        "train_set": train_set_canonical(cache),
        "candidate_args": candidate_args,
        "repeat_seeds": parse_repeat_seeds(args.repeat_seeds, args.seed),
        "reranker": reranker,
    }


def setup_exploratory(args, config: dict, cache: dict, device) -> dict:
    generator, _, retriever, retrieval_k = load_generator(
        resolve_path(config["generator_ckpt"]), cache, device
    )
    return {
        "generator": generator,
        "retriever": retriever,
        "retrieval_k": retrieval_k,
        "train_can": train_set_smiles(cache),
    }


def select_with_cycle(
    pool: list[dict], metrics: dict[str, np.ndarray], args
) -> list[int]:
    if not pool:
        return []
    base = np.array([float(r["base_score"]) for r in pool], dtype=np.float64)
    base_rank = rank01(base, True)
    score = (
        args.w_base_rank * base_rank
        + args.w_cycle_rank * rank01(metrics["cycle_score"], False)
        + args.w_pearson_rank * rank01(metrics["reverse_pearson"], True)
        + args.w_cosine_rank * rank01(metrics["reverse_cosine"], True)
        + args.w_mse_rank * rank01(metrics["reverse_mse"], False)
    )
    if args.base_rank_floor > 0:
        score = score.copy()
        score[base_rank < float(args.base_rank_floor)] = -np.inf
    order = np.argsort(-score)
    selected = []
    seen = set()
    for pos in order:
        smi = pool[int(pos)]["smiles"]
        if smi in seen:
            continue
        selected.append(int(pos))
        seen.add(smi)
        if len(selected) >= args.keep:
            break
    return selected
