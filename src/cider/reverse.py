from __future__ import annotations
from cider.paths import resolve_path
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import selfies as sf
except Exception:
    sf = None
from cider.models import ReversePredictor
from cider.utils import (
    batch_cosine,
    batch_pearson,
    get_device,
    safe_torch_load,
    set_seed,
)
from cider.training.ensemble import make_token_model
from cider.training.reverse_tabular import (
    TabularReversePredictor,
    build_or_load_feature_cache,
    molecule_features,
)

DEFAULT_MANIFEST = "configs/reverse.json"


def read_manifest(path: str | Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def encode_smiles(smiles: str, cache: dict) -> torch.Tensor:
    cfg = cache["config"]
    stoi = cache["vocab"]["stoi"]
    ids = [cfg["bos_id"]]
    if sf is not None:
        try:
            toks = list(sf.split_selfies(sf.encoder(smiles)))
        except Exception:
            toks = []
    else:
        toks = []
    ids.extend((stoi.get(tok, cfg["unk_id"]) for tok in toks))
    ids.append(cfg["eos_id"])
    ids = ids[: cfg["max_len"]]
    ids.extend([cfg["pad_id"]] * (cfg["max_len"] - len(ids)))
    return torch.tensor(ids, dtype=torch.long)


def parse_rows(cache: dict, args) -> tuple[list[dict], torch.Tensor | None]:
    if args.input_csv:
        rows = []
        with open(args.input_csv, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                rows.append(
                    {
                        "source": "csv",
                        "input_id": row.get("id", str(i)),
                        "target_index": "",
                        "smiles": row["smiles"],
                        "dose": float(row["dose"]),
                        "cell": int(row["cell"]),
                        "time": int(row["time"]),
                        "tokens": encode_smiles(row["smiles"], cache),
                    }
                )
        return (rows, None)
    if args.row_indices:
        indices = [int(x) for x in args.row_indices.split(",") if x.strip()]
    else:
        indices = list(map(int, cache["splits"][args.split]))
        if args.sample_rows:
            rng = np.random.default_rng(args.row_seed)
            indices = [
                int(x)
                for x in rng.choice(
                    indices, size=min(args.rows, len(indices)), replace=False
                )
            ]
        else:
            indices = indices[: args.rows]
    rows = []
    targets = []
    for idx in indices:
        rows.append(
            {
                "source": "cache",
                "input_id": str(idx),
                "target_index": idx,
                "smiles": cache["smiles"][idx],
                "dose": float(cache["dose"][idx].item()),
                "cell": int(cache["cell"][idx].item()),
                "time": int(cache["time"][idx].item()),
                "tokens": cache["tokens"][idx].long(),
            }
        )
        targets.append(cache["X"][idx].float())
    return (rows, torch.stack(targets) if targets else None)


class RowDataset(Dataset):

    def __init__(self, rows: list[dict], features_by_bits: dict[int, torch.Tensor]):
        self.rows = rows
        self.features_by_bits = features_by_bits

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        item = {
            "tokens": row["tokens"],
            "dose": torch.tensor(row["dose"], dtype=torch.float32),
            "cell": torch.tensor(row["cell"], dtype=torch.long),
            "time": torch.tensor(row["time"], dtype=torch.long),
        }
        for bits, feats in self.features_by_bits.items():
            item[f"features_{bits}"] = feats[i]
        return item


def make_external_features(
    rows: list[dict], feature_cache_path: str, morgan_bits: int
) -> torch.Tensor:
    obj = safe_torch_load(feature_cache_path, map_location="cpu")
    raw = np.stack(
        [molecule_features(row["smiles"], morgan_bits) for row in rows]
    ).astype(np.float32)
    mean = obj["mean"].astype(np.float32)
    std = obj["std"].astype(np.float32)
    std[std < 1e-06] = 1.0
    return torch.tensor(((raw - mean) / std).astype(np.float16), dtype=torch.float16)


def features_for_bits(
    rows: list[dict], cache: dict, ckpt_args: dict, morgan_bits: int
) -> torch.Tensor:
    feature_cache = resolve_path(f"data/features/rdkit_{morgan_bits}.pt")
    if rows and rows[0]["source"] == "cache":
        all_features = build_or_load_feature_cache(
            cache, Path(feature_cache), morgan_bits
        )
        idx = torch.tensor([int(row["target_index"]) for row in rows], dtype=torch.long)
        return all_features[idx]
    return make_external_features(rows, feature_cache, morgan_bits)


@torch.no_grad()
def predict_tabular_group(
    paths: list[str], rows: list[dict], cache: dict, batch_size: int, device
) -> torch.Tensor:
    preds = []
    prepared = []
    features_by_bits = {}
    for path in paths:
        ckpt = safe_torch_load(resolve_path(path), map_location="cpu")
        bits = int(ckpt["args"]["morgan_bits"])
        if bits not in features_by_bits:
            features_by_bits[bits] = features_for_bits(rows, cache, ckpt["args"], bits)
        prepared.append((path, ckpt, bits))
    loader = DataLoader(
        RowDataset(rows, features_by_bits), batch_size=batch_size, shuffle=False
    )
    for path, ckpt, bits in prepared:
        model = TabularReversePredictor(**ckpt["model_args"]).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        parts = []
        for batch in loader:
            pred = model(
                batch[f"features_{bits}"].to(device),
                batch["dose"].to(device),
                batch["cell"].to(device),
                batch["time"].to(device),
            )
            parts.append(pred.cpu())
        preds.append(torch.cat(parts))
    return torch.stack(preds, dim=0).mean(dim=0)


@torch.no_grad()
def predict_token(path: str, rows: list[dict], batch_size: int, device) -> torch.Tensor:
    ckpt = safe_torch_load(resolve_path(path), map_location="cpu")
    model = make_token_model(ckpt).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    loader = DataLoader(RowDataset(rows, {}), batch_size=batch_size, shuffle=False)
    parts = []
    for batch in loader:
        pred = model.forward_tokens(
            batch["tokens"].to(device),
            batch["dose"].to(device),
            batch["cell"].to(device),
            batch["time"].to(device),
        )
        parts.append(pred.cpu())
    return torch.cat(parts)


def write_meta(path: Path, rows: list[dict], pred: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fieldnames = [
            "input_id",
            "target_index",
            "smiles",
            "dose",
            "cell",
            "time",
            "pred_mean",
            "pred_std",
            "pred_l2",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, y in zip(rows, pred):
            writer.writerow(
                {
                    "input_id": row["input_id"],
                    "target_index": row["target_index"],
                    "smiles": row["smiles"],
                    "dose": row["dose"],
                    "cell": row["cell"],
                    "time": row["time"],
                    "pred_mean": float(y.mean().item()),
                    "pred_std": float(y.std().item()),
                    "pred_l2": float(y.norm().item()),
                }
            )


def metric_payload(pred: torch.Tensor, target: torch.Tensor | None) -> dict:
    if target is None:
        return {}
    return {
        "n": int(pred.size(0)),
        "mse": float(torch.nn.functional.mse_loss(pred, target).item()),
        "pearson": batch_pearson(pred, target),
        "cosine": batch_cosine(pred, target),
    }


def main():
    p = argparse.ArgumentParser(
        description="Predict gene-expression responses with the CIDER cross-family ensemble."
    )
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--cache", default="data/processed/cache.pt")
    p.add_argument("--out_prefix", required=True)
    p.add_argument(
        "--input_csv",
        default="",
        help="Optional CSV with columns: smiles,dose,cell,time and optional id.",
    )
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--rows", type=int, default=16)
    p.add_argument("--row_indices", default="")
    p.add_argument("--sample_rows", action="store_true")
    p.add_argument("--row_seed", type=int, default=20260507)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    manifest = read_manifest(args.manifest)
    cache = safe_torch_load(resolve_path(args.cache), map_location="cpu")
    rows, target = parse_rows(cache, args)
    device = get_device(args.device)
    weights = manifest["weights"]
    tabular_pred = predict_tabular_group(
        manifest["tabular_checkpoints"], rows, cache, args.batch_size, device
    )
    transformer_pred = predict_token(
        manifest["token_checkpoints"]["transformer"], rows, args.batch_size, device
    )
    perceiver_pred = predict_token(
        manifest["token_checkpoints"]["perceiver"], rows, args.batch_size, device
    )
    pred = (
        float(weights["tabular"]) * tabular_pred
        + float(weights["transformer"]) * transformer_pred
        + float(weights["perceiver"]) * perceiver_pred
    )
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(prefix) + ".pred.npy", pred.numpy().astype(np.float32))
    write_meta(Path(str(prefix) + ".meta.csv"), rows, pred)
    metrics = metric_payload(pred, target)
    payload = {
        "model": manifest["name"],
        "n_inputs": len(rows),
        "pred_npy": str(prefix) + ".pred.npy",
        "meta_csv": str(prefix) + ".meta.csv",
        "metrics_json": str(prefix) + ".metrics.json",
        "metrics": metrics,
    }
    Path(str(prefix) + ".metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
