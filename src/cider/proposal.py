from __future__ import annotations
from cider.paths import resolve_path
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm
from cider.utils import get_device, safe_torch_load, set_seed
from cider.sampling import generate_ids_batch, load_fp_model, load_generator
from cider.chemistry import (
    canonicalize_smiles,
    max_tanimoto_to_train,
    mol_props,
    tanimoto_between,
)
from cider.training.fingerprint import load_or_build_fps
from cider.training.forward_generator import ids_to_smiles
from cider.training.structural_ranker import (
    StructuralReranker,
    make_row_candidates,
    parse_repeat_seeds,
    train_set_canonical,
)

RDLogger.DisableLog("rdApp.*")
DEFAULT_VERSION_CONFIG = "configs/forward.json"


def load_versions(path: str | Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def source_smiles(cache: dict, idx: int) -> str:
    smi, ok = canonicalize_smiles(cache["smiles"][idx])
    return smi if ok else ""


def source_tanimoto(cache: dict, idx: int, smiles: str) -> float:
    src = source_smiles(cache, idx)
    if not src:
        return float("nan")
    return tanimoto_between(smiles, src)


def exploratory_score(
    props: dict, novel: bool, tan_to_source: float, weights: dict
) -> float:
    qed = float(props.get("qed", np.nan))
    sas = float(props.get("sas", np.nan))
    lipinski = bool(props.get("lipinski", False))
    if math.isnan(qed):
        qed = 0.0
    if math.isnan(sas):
        sas = 10.0
    if math.isnan(tan_to_source):
        tan_to_source = 0.0
    sas_quality = max(0.0, min(1.0, (10.0 - sas) / 9.0))
    return (
        weights["qed"] * qed
        + weights["sas"] * sas_quality
        + weights["lipinski"] * float(lipinski)
        + weights["novel"] * float(novel)
        + weights["source_tanimoto"] * tan_to_source
    )
