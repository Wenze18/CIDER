from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, MACCSkeys, rdMolDescriptors
from cider.utils import (
    batch_cosine,
    batch_pearson,
    get_device,
    safe_torch_load,
    set_seed,
)

RDLogger.DisableLog("rdApp.*")
DESC_FNS = [
    Descriptors.MolWt,
    Descriptors.MolLogP,
    Descriptors.TPSA,
    Descriptors.NumHAcceptors,
    Descriptors.NumHDonors,
    Descriptors.NumRotatableBonds,
    Descriptors.RingCount,
    Descriptors.FractionCSP3,
    rdMolDescriptors.CalcNumAliphaticRings,
    rdMolDescriptors.CalcNumAromaticRings,
    rdMolDescriptors.CalcNumSaturatedRings,
    rdMolDescriptors.CalcNumHeteroatoms,
    rdMolDescriptors.CalcNumHeavyAtoms,
    rdMolDescriptors.CalcNumAmideBonds,
    rdMolDescriptors.CalcNumBridgeheadAtoms,
    rdMolDescriptors.CalcNumSpiroAtoms,
    rdMolDescriptors.CalcNumAtomStereoCenters,
    rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters,
    Lipinski.NumHeteroatoms,
    Lipinski.NumAromaticRings,
    Lipinski.NumSaturatedRings,
    Lipinski.NumAliphaticRings,
]


def cosine_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-08
) -> torch.Tensor:
    pred_n = pred / (pred.norm(dim=1, keepdim=True) + eps)
    target_n = target / (target.norm(dim=1, keepdim=True) + eps)
    return 1.0 - (pred_n * target_n).sum(dim=1).mean()


def pearson_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-08
) -> torch.Tensor:
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    target_c = target - target.mean(dim=1, keepdim=True)
    corr = (pred_c * target_c).sum(dim=1) / (
        pred_c.norm(dim=1) * target_c.norm(dim=1) + eps
    )
    return 1.0 - corr.mean()


def bitvect_to_array(fp, n_bits: int) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def molecule_features(smiles: str, morgan_bits: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((morgan_bits * 2 + 167 + len(DESC_FNS),), dtype=np.float32)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=morgan_bits)
    fp3 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=morgan_bits)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    desc = []
    for fn in DESC_FNS:
        try:
            v = float(fn(mol))
        except Exception:
            v = 0.0
        if not np.isfinite(v):
            v = 0.0
        desc.append(v)
    return np.concatenate(
        [
            bitvect_to_array(fp2, morgan_bits),
            bitvect_to_array(fp3, morgan_bits),
            bitvect_to_array(maccs, 167),
            np.asarray(desc, dtype=np.float32),
        ]
    ).astype(np.float32)


def build_or_load_feature_cache(
    cache: dict, path: Path, morgan_bits: int
) -> torch.Tensor:
    if path.exists():
        obj = safe_torch_load(path, map_location="cpu")
        if obj.get("morgan_bits") == morgan_bits and obj.get("n_rows") == len(
            cache["smiles"]
        ):
            return obj["features"]
    feats = []
    for smi in tqdm(cache["smiles"], desc="RDKit features"):
        feats.append(molecule_features(smi, morgan_bits))
    X = np.stack(feats).astype(np.float32)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-06] = 1.0
    X = ((X - mean) / std).astype(np.float16)
    features = torch.tensor(X, dtype=torch.float16)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "mean": mean.squeeze(0).astype(np.float32),
            "std": std.squeeze(0).astype(np.float32),
            "morgan_bits": morgan_bits,
            "n_rows": len(cache["smiles"]),
        },
        path,
    )
    return features


class FeatureDataset(Dataset):

    def __init__(self, cache: dict, features: torch.Tensor, split: str):
        self.idx = torch.as_tensor(cache["splits"][split], dtype=torch.long)
        self.features = features
        self.y = cache["X"]
        self.dose = cache["dose"]
        self.cell = cache["cell"]
        self.time = cache["time"]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return {
            "features": self.features[j],
            "ctp": self.y[j],
            "dose": self.dose[j],
            "cell": self.cell[j],
            "time": self.time[j],
        }


class ResidualBlock(nn.Module):

    def __init__(self, dim: int, hidden_mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * hidden_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class TabularReversePredictor(nn.Module):

    def __init__(
        self,
        in_dim: int,
        y_dim: int,
        n_cell: int,
        n_time: int,
        d_model: int = 512,
        blocks: int = 6,
        hidden_mult: int = 2,
        dropout: float = 0.1,
        ensemble_size: int = 8,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.feat = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.dose = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.cell = nn.Embedding(n_cell, d_model)
        self.time = nn.Embedding(n_time, d_model)
        self.ctx = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(d_model, hidden_mult, dropout) for _ in range(blocks)]
        )
        self.base = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, y_dim))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, y_dim * ensemble_size),
        )

    def forward_members(self, features, dose, cell, time):
        h_m = self.feat(features.float())
        if dose.ndim == 1:
            dose = dose[:, None]
        h_d = self.dose(dose.float())
        h_c = self.cell(cell.long())
        h_t = self.time(time.long())
        h_ctx = self.ctx(torch.cat([h_d, h_c, h_t], dim=-1))
        h = self.fuse(
            torch.cat([h_m, h_ctx, h_m * h_ctx, torch.abs(h_m - h_ctx)], dim=-1)
        )
        h = self.blocks(h)
        B = h.size(0)
        delta = self.head(h).view(B, self.ensemble_size, -1)
        base = self.base(h_ctx)[:, None, :]
        return delta + base

    def forward(self, features, dose, cell, time):
        return self.forward_members(features, dose, cell, time).mean(dim=1)


def profile_loss(pred, target, args):
    mse = F.mse_loss(pred, target)
    huber = F.smooth_l1_loss(pred, target, beta=args.huber_beta)
    cos = cosine_loss(pred, target)
    pear = pearson_loss(pred, target)
    loss = (
        args.mse_weight * mse
        + args.huber_weight * huber
        + args.cos_weight * cos
        + args.pearson_weight * pear
    )
    return (loss, {"mse": mse, "huber": huber, "cos": cos, "pear": pear})


def make_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float):

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * min(1, progress))
        )

    return LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, optimizer, scheduler, device, train: bool, args):
    model.train(train)
    total_loss = total_mse = total_huber = total_cos = total_pear = n = 0
    pbar = tqdm(loader, desc="train tabular" if train else "eval tabular", leave=False)
    amp_enabled = args.amp and device.type == "cuda"
    for batch in pbar:
        feat = batch["features"].to(device, non_blocking=True)
        ctp = batch["ctp"].to(device, non_blocking=True)
        dose = batch["dose"].to(device, non_blocking=True)
        cell = batch["cell"].to(device, non_blocking=True)
        time = batch["time"].to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if args.feature_dropout > 0:
                mask = torch.rand_like(feat.float()) < args.feature_dropout
                feat = feat.masked_fill(mask, 0)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp_enabled
        ):
            members = model.forward_members(feat, dose, cell, time)
            pred = members.mean(dim=1)
            loss, parts = profile_loss(pred, ctp, args)
            if args.member_loss_weight > 0:
                member_loss = 0.0
                for k in range(members.size(1)):
                    member_loss = (
                        member_loss + profile_loss(members[:, k], ctp, args)[0]
                    )
                loss = loss + args.member_loss_weight * member_loss / members.size(1)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
        bs = ctp.size(0)
        total_loss += float(loss.item()) * bs
        total_mse += float(parts["mse"].item()) * bs
        total_huber += float(parts["huber"].item()) * bs
        total_cos += batch_cosine(pred.detach().float(), ctp.detach().float()) * bs
        total_pear += batch_pearson(pred.detach().float(), ctp.detach().float()) * bs
        n += bs
        pbar.set_postfix(
            loss=total_loss / n,
            mse=total_mse / n,
            pearson=total_pear / n,
            cosine=total_cos / n,
        )
    return {
        "loss": total_loss / max(1, n),
        "mse": total_mse / max(1, n),
        "huber": total_huber / max(1, n),
        "cosine": total_cos / max(1, n),
        "pearson": total_pear / max(1, n),
    }


def main():
    p = argparse.ArgumentParser(
        description="Train RDKit-feature tabular-style reverse predictor."
    )
    p.add_argument("--cache", default="data/processed/cache.pt")
    p.add_argument("--feature_cache", default="data/features/rdkit_2048.pt")
    p.add_argument("--out", default="checkpoints/reverse/tabular.pt")
    p.add_argument("--metrics_jsonl", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--morgan_bits", type=int, default=1024)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--hidden_mult", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--ensemble_size", type=int, default=8)
    p.add_argument("--feature_dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--mse_weight", type=float, default=0.7)
    p.add_argument("--huber_weight", type=float, default=0.3)
    p.add_argument("--cos_weight", type=float, default=0.25)
    p.add_argument("--pearson_weight", type=float, default=0.25)
    p.add_argument("--member_loss_weight", type=float, default=0.1)
    p.add_argument("--huber_beta", type=float, default=0.5)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print("device:", device)
    print("args:", vars(args))
    cache = safe_torch_load(args.cache, map_location="cpu")
    cfg = cache["config"]
    features = build_or_load_feature_cache(
        cache, Path(args.feature_cache), args.morgan_bits
    )
    train_loader = DataLoader(
        FeatureDataset(cache, features, "train"),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        FeatureDataset(cache, features, "val"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = TabularReversePredictor(
        in_dim=features.size(1),
        y_dim=cfg["y_dim"],
        n_cell=cfg["n_cell"],
        n_time=cfg["n_time"],
        d_model=args.d_model,
        blocks=args.blocks,
        hidden_mult=args.hidden_mult,
        dropout=args.dropout,
        ensemble_size=args.ensemble_size,
    ).to(device)
    n_params = sum((p0.numel() for p0 in model.parameters()))
    print(f"feature_dim: {features.size(1)} parameters: {n_params:,}")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(
        optimizer,
        int(args.warmup_frac * args.epochs * len(train_loader)),
        args.epochs * len(train_loader),
        args.min_lr_ratio,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = (
        Path(args.metrics_jsonl) if args.metrics_jsonl else out.with_suffix(".jsonl")
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, scheduler, device, True, args)
        with torch.no_grad():
            va = run_epoch(model, val_loader, optimizer, scheduler, device, False, args)
        row = {
            "epoch": epoch,
            "train": tr,
            "val": va,
            "args": vars(args),
            "n_params": n_params,
            "feature_dim": int(features.size(1)),
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch:03d} | train loss {tr['loss']:.4f} mse {tr['mse']:.4f} pear {tr['pearson']:.4f} cos {tr['cosine']:.4f} | val loss {va['loss']:.4f} mse {va['mse']:.4f} pear {va['pearson']:.4f} cos {va['cosine']:.4f}"
        )
        if va["mse"] < best:
            best = va["mse"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_args": {
                        "in_dim": features.size(1),
                        "y_dim": cfg["y_dim"],
                        "n_cell": cfg["n_cell"],
                        "n_time": cfg["n_time"],
                        "d_model": args.d_model,
                        "blocks": args.blocks,
                        "hidden_mult": args.hidden_mult,
                        "dropout": args.dropout,
                        "ensemble_size": args.ensemble_size,
                    },
                    "cache_config": cfg,
                    "epoch": epoch,
                    "val": va,
                    "train": tr,
                    "args": vars(args),
                    "n_params": n_params,
                    "feature_dim": int(features.size(1)),
                    "model_class": "TabularReversePredictor",
                },
                out,
            )
            print(f"  saved best tabular reverse -> {out}")


if __name__ == "__main__":
    main()
