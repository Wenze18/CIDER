from __future__ import annotations
import argparse
from cider.paths import resolve_path
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from cider.utils import batch_cosine, batch_pearson, get_device, safe_torch_load
from cider.training.reverse_tabular import (
    FeatureDataset,
    TabularReversePredictor,
    build_or_load_feature_cache,
)


@torch.no_grad()
def predict_checkpoint(
    ckpt_path: str, cache: dict, split: str, batch_size: int, device: torch.device
):
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    args = ckpt["args"]
    features = build_or_load_feature_cache(
        cache,
        Path(resolve_path(f"data/features/rdkit_{int(args['morgan_bits'])}.pt")),
        int(args["morgan_bits"]),
    )
    loader = DataLoader(
        FeatureDataset(cache, features, split),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    model = TabularReversePredictor(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    preds = []
    targets = []
    for batch in loader:
        pred = model(
            batch["features"].to(device, non_blocking=True),
            batch["dose"].to(device, non_blocking=True),
            batch["cell"].to(device, non_blocking=True),
            batch["time"].to(device, non_blocking=True),
        ).cpu()
        preds.append(pred)
        targets.append(batch["ctp"])
    return (torch.cat(preds), torch.cat(targets))


def main():
    p = argparse.ArgumentParser(
        description="Evaluate averaged tabular reverse checkpoints."
    )
    p.add_argument("--cache", default="data/processed/cache.pt")
    p.add_argument("--split", default="val")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--out_json", default="outputs/training/reverse/ensemble_val_metrics.json"
    )
    p.add_argument("checkpoints", nargs="+")
    args = p.parse_args()
    device = get_device(args.device)
    cache = safe_torch_load(args.cache, map_location="cpu")
    preds = []
    target = None
    per_checkpoint = []
    for ckpt_path in args.checkpoints:
        pred, y = predict_checkpoint(
            ckpt_path, cache, args.split, args.batch_size, device
        )
        if target is None:
            target = y
        preds.append(pred)
        per_checkpoint.append(
            {
                "checkpoint": ckpt_path,
                "mse": F.mse_loss(pred, target).item(),
                "pearson": batch_pearson(pred, target),
                "cosine": batch_cosine(pred, target),
            }
        )
    ensemble = torch.stack(preds, dim=0).mean(dim=0)
    result = {
        "split": args.split,
        "checkpoints": args.checkpoints,
        "per_checkpoint": per_checkpoint,
        "ensemble": {
            "mse": F.mse_loss(ensemble, target).item(),
            "pearson": batch_pearson(ensemble, target),
            "cosine": batch_cosine(ensemble, target),
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
