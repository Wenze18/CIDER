from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn
from tqdm import tqdm

try:
    import selfies as sf
except Exception as e:
    raise RuntimeError("Please install selfies first: pip install selfies") from e
try:
    from rdkit import Chem, DataStructs, RDConfig, RDLogger
    from rdkit.Chem import Descriptors, QED, Lipinski, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
except Exception as e:
    raise RuntimeError(
        "This evaluation script requires RDKit. Install it with: conda install -c conda-forge rdkit"
    ) from e
try:
    from rdkit.Contrib.SA_Score import sascorer
except Exception:
    try:
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer
    except Exception:
        sascorer = None
from cider.models import ForwardGenerator, ReversePredictor
from cider.utils import get_device, safe_torch_load, set_seed, top_k_top_p_filtering


def decode_ids(
    ids: List[int], itos: List[str], bos_id: int, eos_id: int, pad_id: int
) -> str:
    toks = []
    for i in ids:
        i = int(i)
        if i in (bos_id, pad_id):
            continue
        if i == eos_id:
            break
        tok = itos[i]
        if tok.startswith("<") and tok.endswith(">"):
            continue
        toks.append(tok)
    selfies = "".join(toks)
    try:
        return sf.decoder(selfies)
    except Exception:
        return ""


def canonicalize_smiles(smiles: str) -> Tuple[str, bool]:
    if not smiles:
        return ("", False)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (smiles, False)
    return (Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), True)


def mol_from_smiles(smiles: str):
    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def morgan_fp_from_smiles(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def scaffold_smiles(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None:
            return ""
        return Chem.MolToSmiles(scaf, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def mol_props(smiles: str) -> Dict[str, object]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return {
            "mw": np.nan,
            "logp": np.nan,
            "qed": np.nan,
            "sas": np.nan,
            "hbd": np.nan,
            "hba": np.nan,
            "tpsa": np.nan,
            "rotatable_bonds": np.nan,
            "heavy_atoms": np.nan,
            "rings": np.nan,
            "lipinski_violations": np.nan,
            "lipinski": False,
        }
    mw = float(Descriptors.MolWt(mol))
    logp = float(Descriptors.MolLogP(mol))
    qed = float(QED.qed(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    rotb = int(Lipinski.NumRotatableBonds(mol))
    heavy_atoms = int(mol.GetNumHeavyAtoms())
    rings = int(rdMolDescriptors.CalcNumRings(mol))
    sas = float(sascorer.calculateScore(mol)) if sascorer is not None else np.nan
    lipinski_violations = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    lipinski = lipinski_violations == 0
    return {
        "mw": mw,
        "logp": logp,
        "qed": qed,
        "sas": sas,
        "hbd": hbd,
        "hba": hba,
        "tpsa": tpsa,
        "rotatable_bonds": rotb,
        "heavy_atoms": heavy_atoms,
        "rings": rings,
        "lipinski_violations": lipinski_violations,
        "lipinski": lipinski,
    }


def tanimoto_between(smiles_a: str, smiles_b: str) -> float:
    fp_a = morgan_fp_from_smiles(smiles_a)
    fp_b = morgan_fp_from_smiles(smiles_b)
    if fp_a is None or fp_b is None:
        return np.nan
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def max_tanimoto_to_train(smiles: str, train_fps: List) -> float:
    fp = morgan_fp_from_smiles(smiles)
    if fp is None or len(train_fps) == 0:
        return np.nan
    sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
    return float(max(sims)) if sims else np.nan


def internal_diversity(smiles_list: List[str]) -> float:
    fps = []
    for smi in smiles_list:
        fp = morgan_fp_from_smiles(smi)
        if fp is not None:
            fps.append(fp)
    if len(fps) < 2:
        return np.nan
    sims = []
    for i in range(len(fps) - 1):
        sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :]))
    if not sims:
        return np.nan
    return float(1.0 - np.mean(sims))


def smiles_to_token_tensor(smiles: str, cache: dict, device) -> Optional[torch.Tensor]:
    stoi = cache["vocab"]["stoi"]
    cfg = cache["config"]
    try:
        selfies = sf.encoder(smiles)
        toks = list(sf.split_selfies(selfies))
    except Exception:
        return None
    ids = [stoi["<bos>"]]
    ids.extend([stoi.get(tok, stoi["<unk>"]) for tok in toks])
    ids.append(stoi["<eos>"])
    max_len = cfg["max_len"]
    if len(ids) > max_len:
        ids = ids[: max_len - 1] + [stoi["<eos>"]]
    ids = ids + [stoi["<pad>"]] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long, device=device)


def get_split_indices(cache: dict, split: str) -> List[int]:
    if "splits" in cache and split in cache["splits"]:
        return list(map(int, cache["splits"][split]))
    raise KeyError(
        f"Cannot find split='{split}' in cache['splits']. Available keys: {list(cache.keys())}"
    )


def build_train_reference(
    cache: dict, max_train_fps: int, seed: int
) -> Tuple[set, set, List[str], List]:
    train_idx = get_split_indices(cache, "train")
    train_can_set = set()
    train_scaffold_set = set()
    train_smiles_unique = []
    for i in tqdm(train_idx, desc="Building train reference set"):
        smi = cache["smiles"][int(i)]
        can, valid = canonicalize_smiles(smi)
        if not valid:
            continue
        if can not in train_can_set:
            train_can_set.add(can)
            train_smiles_unique.append(can)
            scaf = scaffold_smiles(can)
            if scaf:
                train_scaffold_set.add(scaf)
    fps_smiles = train_smiles_unique
    if max_train_fps > 0 and len(fps_smiles) > max_train_fps:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(fps_smiles), size=max_train_fps, replace=False)
        fps_smiles = [fps_smiles[int(j)] for j in pick]
    train_fps = []
    for smi in tqdm(fps_smiles, desc="Computing train fingerprints"):
        fp = morgan_fp_from_smiles(smi)
        if fp is not None:
            train_fps.append(fp)
    return (train_can_set, train_scaffold_set, train_smiles_unique, train_fps)


def safe_mean(x) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(x.mean()) if len(x) else np.nan


def safe_median(x) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(x.median()) if len(x) else np.nan


def safe_min(x) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(x.min()) if len(x) else np.nan


def safe_max(x) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(x.max()) if len(x) else np.nan


def topk_mean(df: pd.DataFrame, col: str, k: int, ascending: bool) -> float:
    if col not in df.columns or len(df) == 0:
        return np.nan
    vals = pd.to_numeric(df[col], errors="coerce")
    tmp = df.assign(_score=vals).dropna(subset=["_score"])
    if len(tmp) == 0:
        return np.nan
    return float(
        tmp.sort_values("_score", ascending=ascending).head(k)["_score"].mean()
    )
