from __future__ import annotations
from cider.paths import resolve_path
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from tqdm import tqdm
from cider.reverse import read_manifest
from cider.utils import get_device, safe_torch_load, set_seed
from cider.selection import (
    add_molecule_fields,
    exploratory_pool,
    conservative_pool,
    predict_response_ensemble,
    rank01,
    reverse_metric_arrays,
    reverse_rows_for_candidates,
    select_with_cycle,
    setup_exploratory,
    setup_conservative,
)
from cider.chemistry import build_train_reference
from cider.proposal import load_versions

RDLogger.DisableLog("rdApp.*")
PRESETS = {
    "conservative": {
        "description": "Conservative molecular generation with structural and response-based selection.",
        "candidates_per_generator": 8,
        "repeat_seeds": "42,7,13",
        "candidates": 128,
        "keep": 8,
        "base_rank_floor": 0.0,
        "w_fp": 1.0,
        "w_nn": 1.5,
        "w_learned": 1.5,
        "w_base_rank": 1.0,
        "w_cycle_rank": 0.75,
        "w_pearson_rank": 0.75,
        "w_cosine_rank": 0.25,
        "w_mse_rank": 0.25,
        "temperature": 0.95,
        "top_k": 80,
        "top_p": 0.97,
    },
    "exploratory": {
        "description": "Exploratory molecular generation with chemical-quality constraints and response-based selection.",
        "candidates": 512,
        "candidates_per_generator": 8,
        "repeat_seeds": "42,7,13",
        "keep": 8,
        "base_rank_floor": 0.9,
        "w_fp": 1.0,
        "w_nn": 1.5,
        "w_learned": 1.5,
        "w_base_rank": 1.0,
        "w_cycle_rank": 1.0,
        "w_pearson_rank": 1.0,
        "w_cosine_rank": 0.25,
        "w_mse_rank": 0.25,
        "temperature": 0.95,
        "top_k": 80,
        "top_p": 0.97,
    },
}


def parse_indices(cache: dict, args) -> list[int]:
    if args.row_indices.strip():
        return [int(x) for x in args.row_indices.split(",") if x.strip()]
    indices = list(map(int, cache["splits"][args.split]))
    if args.sample_rows:
        rng = np.random.default_rng(args.row_seed)
        return [
            int(x)
            for x in rng.choice(
                indices, size=min(args.rows, len(indices)), replace=False
            )
        ]
    return indices[: args.rows]


def apply_preset(args):
    preset = PRESETS[args.version]
    for key, value in preset.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, value)
    return preset


def write_csv(path: str | Path, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def selected_rows_for_index(
    args, cache, manifest, idx: int, state: dict, make_pool, device, train_ref
):
    train_can_set, train_scaffold_set, train_fps = train_ref
    pool = make_pool(args, state["version_config"], cache, idx, state, device)
    if not pool:
        return []
    smiles = [r["smiles"] for r in pool]
    reverse_rows = reverse_rows_for_candidates(cache, idx, smiles)
    pred = predict_response_ensemble(
        reverse_rows, cache, manifest, args.batch_size, device
    )
    target = cache["X"][idx].float().view(1, -1).expand(pred.size(0), -1)
    metrics = reverse_metric_arrays(pred, target, args.cos_weight)
    selected = select_with_cycle(pool, metrics, args)
    base_rank = rank01(
        np.array([r["base_score"] for r in pool], dtype=np.float64), True
    )
    rows = []
    seen = set()
    for rank, pos in enumerate(selected, start=1):
        item = dict(pool[pos])
        item.update({k: float(v[pos]) for k, v in metrics.items()})
        item.update(
            {
                "version": args.version,
                "target_index": int(idx),
                "rank": int(rank),
                "smiles": item["smiles"],
                "base_score": float(item["base_score"]),
                "base_rank01": float(base_rank[pos]),
                "cycle_rerank_score": float(
                    args.w_base_rank * base_rank[pos]
                    + args.w_cycle_rank * rank01(metrics["cycle_score"], False)[pos]
                    + args.w_pearson_rank
                    * rank01(metrics["reverse_pearson"], True)[pos]
                    + args.w_cosine_rank * rank01(metrics["reverse_cosine"], True)[pos]
                    + args.w_mse_rank * rank01(metrics["reverse_mse"], False)[pos]
                ),
                "unique_in_condition": item["smiles"] not in seen,
            }
        )
        seen.add(item["smiles"])
        add_molecule_fields(
            item, cache, idx, train_can_set, train_scaffold_set, train_fps
        )
        rows.append(item)
    return rows


def main():
    p = argparse.ArgumentParser(
        description="Molecular generation with response-based cycle reranking."
    )
    p.add_argument(
        "--version",
        required=True,
        choices=["conservative", "exploratory"],
    )
    p.add_argument("--versions", default="configs/forward.json")
    p.add_argument("--reverse_manifest", default="configs/reverse.json")
    p.add_argument("--cache", default="data/processed/cache.pt")
    p.add_argument("--out", required=True, help="Output CSV path.")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--sample_rows", action="store_true")
    p.add_argument("--row_seed", type=int, default=20260507)
    p.add_argument(
        "--row_indices",
        default="",
        help="Comma-separated cache row indices. Overrides split/rows/sample_rows.",
    )
    p.add_argument("--keep", type=int, default=None)
    p.add_argument("--candidates", type=int, default=None)
    p.add_argument("--candidates_per_generator", type=int, default=None)
    p.add_argument("--repeat_seeds", default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--fixed_zero_noise", action="store_true")
    p.add_argument("--w_fp", type=float, default=None)
    p.add_argument("--w_nn", type=float, default=None)
    p.add_argument("--w_learned", type=float, default=None)
    p.add_argument("--w_base_rank", type=float, default=None)
    p.add_argument("--w_cycle_rank", type=float, default=None)
    p.add_argument("--w_pearson_rank", type=float, default=None)
    p.add_argument("--w_cosine_rank", type=float, default=None)
    p.add_argument("--w_mse_rank", type=float, default=None)
    p.add_argument("--base_rank_floor", type=float, default=None)
    p.add_argument("--cos_weight", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--max_train_fps", type=int, default=50000)
    p.add_argument("--train_ref_cache", default="data/features/train_reference.pkl")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--write_config",
        default="",
        help="Optional JSON path for the resolved inference configuration.",
    )
    args = p.parse_args()
    preset = apply_preset(args)
    set_seed(args.seed)
    device = get_device(args.device)
    cache = safe_torch_load(resolve_path(args.cache), map_location="cpu")
    versions = load_versions(resolve_path(args.versions))
    version_config = versions[args.version]
    manifest = read_manifest(resolve_path(args.reverse_manifest))
    indices = parse_indices(cache, args)
    train_ref_path = resolve_path(args.train_ref_cache)
    if train_ref_path.exists():
        with train_ref_path.open("rb") as f:
            train_ref = pickle.load(f)
    else:
        train_can_set, train_scaffold_set, _, train_fps = build_train_reference(
            cache, args.max_train_fps, args.seed
        )
        train_ref = (train_can_set, train_scaffold_set, train_fps)
        train_ref_path.parent.mkdir(parents=True, exist_ok=True)
        with train_ref_path.open("wb") as f:
            pickle.dump(train_ref, f)
    if args.version == "conservative":
        state = setup_conservative(args, version_config, cache, device)
        make_pool = conservative_pool
    else:
        state = setup_exploratory(args, version_config, cache, device)
        make_pool = exploratory_pool
    state["version_config"] = version_config
    rows = []
    for idx in tqdm(indices, desc=f"{args.version} cycle inference"):
        rows.extend(
            selected_rows_for_index(
                args, cache, manifest, int(idx), state, make_pool, device, train_ref
            )
        )
    write_csv(args.out, rows)
    resolved = {
        "version": args.version,
        "preset": preset,
        "args": vars(args),
        "n_conditions": len(indices),
        "n_output_rows": len(rows),
    }
    config_path = (
        Path(args.write_config)
        if args.write_config
        else Path(args.out).with_suffix(".config.json")
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": args.out,
                "config": str(config_path),
                "n_conditions": len(indices),
                "n_output_rows": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
