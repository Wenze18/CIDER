from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import selfies as sf
except Exception as e:
    raise RuntimeError("Please install selfies first: pip install selfies") from e
from cider.utils import set_seed

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def infer_col(
    obs: pd.DataFrame,
    candidates: List[str],
    contains_any: List[str],
    required: bool = True,
) -> Optional[str]:
    cols = list(obs.columns)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for key in contains_any:
        hits = [c for c in cols if key.lower() in c.lower()]
        if hits:
            hits = sorted(hits, key=lambda x: ("unit" in x.lower(), len(x)))
            return hits[0]
    if required:
        raise ValueError(
            f"Could not infer required column. Tried candidates {candidates} and substrings {contains_any}. Available obs columns:\n{cols}"
        )
    return None


def parse_float(x) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    m = re.search("[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?", s)
    return float(m.group(0)) if m else np.nan


def normalize_time_value(x) -> str:
    if pd.isna(x):
        return "NA"
    s = str(x).strip()
    try:
        v = float(s)
        if abs(v - int(v)) < 1e-08:
            return str(int(v))
        return str(v)
    except Exception:
        return s


def dense_x(X):
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def smiles_to_token_ids(
    smiles: str, stoi: Dict[str, int], max_len: int
) -> Optional[List[int]]:
    try:
        selfies = sf.encoder(smiles)
        toks = list(sf.split_selfies(selfies))
    except Exception:
        return None
    ids = [stoi["<bos>"]]
    for tok in toks:
        ids.append(stoi.get(tok, stoi["<unk>"]))
    ids.append(stoi["<eos>"])
    if len(ids) > max_len:
        return None
    ids = ids + [stoi["<pad>"]] * (max_len - len(ids))
    return ids


def build_selfies_vocab(
    smiles_list: List[str], max_len: int
) -> Tuple[Dict[str, int], List[str], List[int]]:
    token_set = set()
    valid_indices = []
    lengths = []
    for i, smi in enumerate(tqdm(smiles_list, desc="Encoding SELFIES for vocab")):
        try:
            s = sf.encoder(str(smi))
            toks = list(sf.split_selfies(s))
            L = len(toks) + 2
            if L <= max_len:
                token_set.update(toks)
                valid_indices.append(i)
                lengths.append(L)
        except Exception:
            continue
    itos = SPECIAL_TOKENS + sorted(token_set)
    stoi = {tok: i for i, tok in enumerate(itos)}
    return (stoi, itos, valid_indices)


def group_split(smiles: List[str], seed: int, val_frac: float, test_frac: float):
    unique = np.array(sorted(set(smiles)))
    train_u, temp_u = train_test_split(
        unique, test_size=val_frac + test_frac, random_state=seed
    )
    rel_test = test_frac / (val_frac + test_frac)
    val_u, test_u = train_test_split(temp_u, test_size=rel_test, random_state=seed)
    train_set, val_set, test_set = (set(train_u), set(val_u), set(test_u))
    train_idx, val_idx, test_idx = ([], [], [])
    for i, smi in enumerate(smiles):
        if smi in train_set:
            train_idx.append(i)
        elif smi in val_set:
            val_idx.append(i)
        else:
            test_idx.append(i)
    return (np.array(train_idx), np.array(val_idx), np.array(test_idx))


def main():
    p = argparse.ArgumentParser(description="Prepare LINCS L1000 profiles for CIDER.")
    p.add_argument("--h5ad", default="data/processed/lincs_cp_landmark_all.h5ad")
    p.add_argument("--out", default="data/processed/cache.pt")
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--smiles_col", default=None)
    p.add_argument("--dose_col", default=None)
    p.add_argument("--cell_col", default=None)
    p.add_argument("--time_col", default=None)
    p.add_argument(
        "--no_group_split",
        action="store_true",
        help="Random row split instead of molecule-level split.",
    )
    args = p.parse_args()
    set_seed(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Reading {args.h5ad}")
    adata = ad.read_h5ad(args.h5ad)
    obs = adata.obs.reset_index(drop=True).copy()
    print("shape:", adata.shape)
    smiles_col = args.smiles_col or infer_col(
        obs,
        [
            "canonical_smiles",
            "smiles",
            "SMILES",
            "pert_smiles",
            "pubchem_smiles",
            "rdkit_smiles",
        ],
        ["smiles"],
    )
    dose_col = args.dose_col or infer_col(
        obs,
        [
            "dose",
            "pert_dose",
            "pert_dose_value",
            "dose_um",
            "pert_dose_um",
            "dose_value",
        ],
        ["dose"],
    )
    cell_col = args.cell_col or infer_col(
        obs,
        ["cell", "cell_id", "cell_line", "cell_iname", "cell_mfc_name", "cell_name"],
        ["cell"],
    )
    time_col = args.time_col or infer_col(
        obs,
        [
            "time",
            "time_value",
            "pert_time",
            "pert_time_value",
            "time_h",
            "duration",
            "pert_itime",
        ],
        ["time", "duration"],
    )
    print("Using columns:")
    print(f"  SMILES: {smiles_col}")
    print(f"  dose:   {dose_col}")
    print(f"  cell:   {cell_col}")
    print(f"  time:   {time_col}")

    def convert_dose_to_um(value, unit):
        if pd.isna(value):
            return np.nan
        v = float(value)
        u = str(unit).strip().lower().replace("µ", "u")
        if u in ["um", "uM".lower(), "micromolar", ""]:
            return v
        if u in ["nm", "nanomolar"]:
            return v * 0.001
        if u in ["mm", "millimolar"]:
            return v * 1000.0
        if u in ["pm", "picomolar"]:
            return v * 1e-06
        return np.nan

    def convert_time_to_hours(value, unit):
        if pd.isna(value):
            return np.nan
        v = float(value)
        u = str(unit).strip().lower()
        if u in ["h", "hr", "hrs", "hour", "hours"]:
            return v
        if u in ["m", "min", "mins", "minute", "minutes"]:
            return v / 60.0
        if u in ["d", "day", "days"]:
            return v * 24.0
        if u in ["w", "wk", "week", "weeks"]:
            return v * 24.0 * 7.0
        return np.nan

    smiles_raw = obs[smiles_col].astype(str).replace({"nan": np.nan, "None": np.nan})
    cell_raw = obs[cell_col].astype(str).fillna("NA")
    dose_um = np.array(
        [convert_dose_to_um(v, u) for v, u in zip(obs["dose_value"], obs["dose_unit"])],
        dtype=np.float32,
    )
    time_h = np.array(
        [
            convert_time_to_hours(v, u)
            for v, u in zip(obs["time_value"], obs["time_unit"])
        ],
        dtype=np.float32,
    )
    dose_raw = pd.Series(dose_um)
    time_raw = pd.Series(
        [
            str(int(x)) if np.isfinite(x) and abs(x - int(x)) < 1e-08 else str(x)
            for x in time_h
        ]
    )
    base_keep = (
        smiles_raw.notna()
        & np.isfinite(dose_raw.values)
        & cell_raw.notna()
        & time_raw.notna()
    )
    base_idx = np.where(base_keep.values)[0]
    print(f"Rows after metadata filtering: {len(base_idx):,} / {adata.n_obs:,}")
    smiles_base = smiles_raw.iloc[base_idx].astype(str).tolist()
    stoi, itos, valid_local = build_selfies_vocab(smiles_base, max_len=args.max_len)
    valid_idx = base_idx[np.array(valid_local, dtype=int)]
    print(f"Rows after SELFIES/max_len filtering: {len(valid_idx):,}")
    print(f"Vocab size: {len(itos)}")
    smiles = smiles_raw.iloc[valid_idx].astype(str).tolist()
    token_rows = []
    for smi in tqdm(smiles, desc="Tokenizing molecules"):
        ids = smiles_to_token_ids(smi, stoi, args.max_len)
        if ids is None:
            raise RuntimeError("Unexpected tokenization failure after filtering.")
        token_rows.append(ids)
    tokens = torch.tensor(token_rows, dtype=torch.long)
    dose_vals = dose_raw.iloc[valid_idx].to_numpy(dtype=np.float32)
    dose_log = np.log10(np.maximum(dose_vals, 1e-12)).astype(np.float32)
    cell_vals = cell_raw.iloc[valid_idx].astype(str).tolist()
    time_vals = time_raw.iloc[valid_idx].astype(str).tolist()
    cell_itos = sorted(set(cell_vals))
    time_itos = sorted(set(time_vals))
    cell_stoi = {x: i for i, x in enumerate(cell_itos)}
    time_stoi = {x: i for i, x in enumerate(time_itos)}
    cell_ids = torch.tensor([cell_stoi[x] for x in cell_vals], dtype=torch.long)
    time_ids = torch.tensor([time_stoi[x] for x in time_vals], dtype=torch.long)
    if args.no_group_split:
        all_idx = np.arange(len(valid_idx))
        train_idx, temp_idx = train_test_split(
            all_idx, test_size=args.val_frac + args.test_frac, random_state=args.seed
        )
        rel_test = args.test_frac / (args.val_frac + args.test_frac)
        val_idx2, test_idx = train_test_split(
            temp_idx, test_size=rel_test, random_state=args.seed
        )
        val_idx = val_idx2
    else:
        train_idx, val_idx, test_idx = group_split(
            smiles, seed=args.seed, val_frac=args.val_frac, test_frac=args.test_frac
        )
    print(
        f"Split rows: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}"
    )
    print("Loading CTP matrix")
    X = dense_x(adata.X[valid_idx]).astype(np.float32)
    train_mean = X[train_idx].mean(axis=0, keepdims=True).astype(np.float32)
    train_std = X[train_idx].std(axis=0, keepdims=True).astype(np.float32)
    train_std[train_std < 1e-06] = 1.0
    X_norm = ((X - train_mean) / train_std).astype(np.float32)
    dose_mean = float(dose_log[train_idx].mean())
    dose_std = float(dose_log[train_idx].std())
    if dose_std < 1e-06:
        dose_std = 1.0
    dose_norm = ((dose_log - dose_mean) / dose_std).astype(np.float32)
    cache = {
        "X": torch.tensor(X_norm, dtype=torch.float32),
        "dose": torch.tensor(dose_norm, dtype=torch.float32),
        "cell": cell_ids,
        "time": time_ids,
        "tokens": tokens,
        "smiles": smiles,
        "raw_h5ad_indices": valid_idx.astype(int).tolist(),
        "splits": {
            "train": train_idx.astype(int).tolist(),
            "val": val_idx.astype(int).tolist(),
            "test": test_idx.astype(int).tolist(),
        },
        "vocab": {"stoi": stoi, "itos": itos},
        "cell_vocab": {"stoi": cell_stoi, "itos": cell_itos},
        "time_vocab": {"stoi": time_stoi, "itos": time_itos},
        "normalization": {
            "ctp_mean": train_mean.squeeze(0),
            "ctp_std": train_std.squeeze(0),
            "dose_log_mean": dose_mean,
            "dose_log_std": dose_std,
        },
        "columns": {
            "smiles_col": smiles_col,
            "dose_col": dose_col,
            "cell_col": cell_col,
            "time_col": time_col,
        },
        "config": {
            "max_len": args.max_len,
            "y_dim": int(X.shape[1]),
            "vocab_size": len(itos),
            "pad_id": stoi["<pad>"],
            "bos_id": stoi["<bos>"],
            "eos_id": stoi["<eos>"],
            "unk_id": stoi["<unk>"],
            "n_cell": len(cell_itos),
            "n_time": len(time_itos),
        },
    }
    torch.save(cache, out)
    meta_path = out.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "n_rows": len(valid_idx),
                "columns": cache["columns"],
                "config": cache["config"],
                "n_cells": len(cell_itos),
                "n_times": len(time_itos),
                "splits": {k: len(v) for k, v in cache["splits"].items()},
            },
            f,
            indent=2,
        )
    print(f"Saved cache: {out}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
