from __future__ import annotations
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.optim import AdamW
from tqdm import tqdm
from cider.utils import get_device, make_loader, safe_torch_load, set_seed
from cider.training.forward_generator import ResidualMLP, StrongConditionEncoder

RDLogger.DisableLog("rdApp.*")


def mol_from_smiles(smiles: str):
    return Chem.MolFromSmiles(smiles) if smiles else None


def morgan_bits(mol, n_bits: int = 2048) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.float32)
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def maccs_fp(mol):
    return MACCSkeys.GenMACCSKeys(mol) if mol is not None else None


def maccs_bits(mol) -> np.ndarray:
    arr = np.zeros((167,), dtype=np.float32)
    if mol is None:
        return arr
    fp = MACCSkeys.GenMACCSKeys(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def scaffold_smiles(mol) -> str:
    if mol is None:
        return ""
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    return (
        Chem.MolToSmiles(scaf, canonical=True, isomericSmiles=True)
        if scaf is not None
        else ""
    )


def featurize_smiles(args):
    smi, fp_dim = args
    mol = mol_from_smiles(smi)
    return (morgan_bits(mol, fp_dim), maccs_bits(mol), scaffold_smiles(mol))


class FingerprintPredictor(nn.Module):

    def __init__(
        self,
        y_dim: int,
        n_cell: int,
        n_time: int,
        d_model: int,
        fp_dim: int,
        dropout: float,
        depth: int,
    ):
        super().__init__()
        self.cond = StrongConditionEncoder(
            y_dim, n_cell, n_time, d_model, dropout, depth=depth
        )
        self.blocks = nn.Sequential(
            *[
                ResidualMLP(d_model, hidden_mult=4, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, fp_dim),
        )

    def forward(self, ctp, dose, cell, time):
        h = self.blocks(self.cond(ctp, dose, cell, time))
        return self.head(h)


def build_fps(cache: dict, fp_dim: int, fp_workers: int):
    jobs = [(smi, fp_dim) for smi in cache["smiles"]]
    if fp_workers > 1:
        with Pool(processes=fp_workers) as pool:
            rows = list(
                tqdm(
                    pool.imap(featurize_smiles, jobs, chunksize=256),
                    total=len(jobs),
                    desc="fingerprints",
                )
            )
    else:
        rows = [featurize_smiles(job) for job in tqdm(jobs, desc="fingerprints")]
    fps, maccs, scafs = zip(*rows)
    return (
        torch.tensor(np.stack(fps), dtype=torch.float32),
        torch.tensor(np.stack(maccs), dtype=torch.float32),
        scafs,
    )


def load_or_build_fps(cache: dict, fp_dim: int, fp_cache: str | None, fp_workers: int):
    if fp_cache:
        path = Path(fp_cache)
        if path.exists():
            payload = safe_torch_load(path, map_location="cpu")
            if payload.get("fp_dim") == fp_dim and payload.get("n_smiles") == len(
                cache["smiles"]
            ):
                return (payload["fps"], payload["maccs"], payload["scafs"])
    fps, maccs, scafs = build_fps(cache, fp_dim, fp_workers)
    if fp_cache:
        path = Path(fp_cache)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "fp_dim": fp_dim,
                "n_smiles": len(cache["smiles"]),
                "fps": fps,
                "maccs": maccs,
                "scafs": scafs,
            },
            path,
        )
    return (fps, maccs, scafs)


def run_epoch(model, loader, fps, optimizer, device, train: bool):
    model.train(train)
    total_loss = total_bit_acc = n = 0
    pbar = tqdm(loader, desc="train FP" if train else "eval FP", leave=False)
    for batch in pbar:
        ctp = batch["ctp"].to(device)
        dose = batch["dose"].to(device)
        cell = batch["cell"].to(device)
        time = batch["time"].to(device)
        row_index = batch["row_index"].to(device)
        target = fps[row_index.cpu()].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(ctp, dose, cell, time)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            bit_acc = logits.sigmoid().ge(0.5).eq(target.bool()).float().mean().item()
        total_loss += float(loss.item()) * ctp.size(0)
        total_bit_acc += bit_acc * ctp.size(0)
        n += ctp.size(0)
        pbar.set_postfix(loss=total_loss / max(n, 1), bit_acc=total_bit_acc / max(n, 1))
    return {"loss": total_loss / max(n, 1), "bit_acc": total_bit_acc / max(n, 1)}


@torch.no_grad()
def retrieval_eval(
    model, cache, fps, maccs, scafs, split: str, device, max_rows: int, topk: int
):
    model.eval()
    train_idx_cpu = torch.as_tensor(cache["splits"]["train"], dtype=torch.long)
    train_idx = train_idx_cpu.to(device)
    train_fps = fps[train_idx_cpu].to(device)
    train_sum = train_fps.sum(dim=1)
    rows = list(map(int, cache["splits"][split]))[:max_rows]
    max_morgan = []
    mean_morgan = []
    max_maccs = []
    hit03 = hit05 = hit07 = scaf_hit = 0
    for idx in tqdm(rows, desc=f"retrieval eval {split}", leave=False):
        ctp = cache["X"][idx].to(device).view(1, -1)
        dose = cache["dose"][idx].to(device).view(1)
        cell = cache["cell"][idx].to(device).view(1)
        time = cache["time"][idx].to(device).view(1)
        prob = model(ctp, dose, cell, time).sigmoid()
        inter = prob @ train_fps.T
        denom = prob.sum(dim=1, keepdim=True) + train_sum[None, :] - inter
        score = inter / denom.clamp_min(1e-06)
        nn_pos = score.topk(topk, dim=1).indices.squeeze(0).tolist()
        nn_idx = [int(train_idx[p].item()) for p in nn_pos]
        gt = fps[idx].to(device)
        cand = fps[nn_idx].to(device)
        hard_inter = (cand * gt[None, :]).sum(dim=1)
        hard_denom = cand.sum(dim=1) + gt.sum() - hard_inter
        sims = (hard_inter / hard_denom.clamp_min(1e-06)).detach().cpu().numpy()
        mmax = float(np.max(sims))
        max_morgan.append(mmax)
        mean_morgan.append(float(np.mean(sims)))
        hit03 += int(mmax >= 0.3)
        hit05 += int(mmax >= 0.5)
        hit07 += int(mmax >= 0.7)
        gt_maccs = maccs[idx].to(device)
        cand_maccs = maccs[nn_idx].to(device)
        mac_inter = (cand_maccs * gt_maccs[None, :]).sum(dim=1)
        mac_denom = cand_maccs.sum(dim=1) + gt_maccs.sum() - mac_inter
        mac_sims = (mac_inter / mac_denom.clamp_min(1e-06)).detach().cpu().numpy()
        max_maccs.append(float(np.max(mac_sims)) if len(mac_sims) else float("nan"))
        gt_scaf = scafs[idx]
        scaf_hit += int(
            any((gt_scaf and scafs[j] and (scafs[j] == gt_scaf) for j in nn_idx))
        )
    n = len(max_morgan)
    return {
        "split": split,
        "rows": n,
        "topk": topk,
        "gt_max_morgan": float(np.mean(max_morgan)),
        "gt_mean_morgan": float(np.mean(mean_morgan)),
        "gt_hit_morgan_0.3": hit03 / max(n, 1),
        "gt_hit_morgan_0.5": hit05 / max(n, 1),
        "gt_hit_morgan_0.7": hit07 / max(n, 1),
        "gt_max_maccs": float(np.nanmean(max_maccs)),
        "gt_scaffold_hit": scaf_hit / max(n, 1),
        "validity": 1.0,
        "novel_unique_per_sample": 0.0,
    }


def main():
    p = argparse.ArgumentParser(
        description="Forward-only CTP-to-fingerprint retrieval baseline."
    )
    p.add_argument("--cache", default="data/processed/cache.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--metrics_jsonl", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--fp_dim", type=int, default=2048)
    p.add_argument("--fp_cache", default="")
    p.add_argument("--cache_only", action="store_true")
    p.add_argument("--fp_workers", type=int, default=1)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--eval_rows", type=int, default=256)
    p.add_argument("--early_stop_patience", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_only", action="store_true")
    args = p.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    cache = safe_torch_load(args.cache, map_location="cpu")
    cfg = cache["config"]
    fps, maccs, scafs = load_or_build_fps(
        cache, args.fp_dim, args.fp_cache or None, args.fp_workers
    )
    if args.cache_only:
        print(
            json.dumps(
                {
                    "fp_cache": args.fp_cache,
                    "fp_dim": args.fp_dim,
                    "n_smiles": len(cache["smiles"]),
                }
            )
        )
        return
    model = FingerprintPredictor(
        cfg["y_dim"],
        cfg["n_cell"],
        cfg["n_time"],
        args.d_model,
        args.fp_dim,
        args.dropout,
        args.depth,
    ).to(device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_jsonl)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        ckpt = safe_torch_load(out, map_location=device)
        model.load_state_dict(ckpt["model"])
        ev = retrieval_eval(
            model, cache, fps, maccs, scafs, "val", device, args.eval_rows, args.topk
        )
        payload = {
            "best_epoch": ckpt.get("epoch", -1),
            "best_val": ckpt.get("val", {}),
            "retrieval_val": ev,
        }
        out.with_suffix(".eval.json").write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        return
    train_loader = make_loader(
        cache, "train", args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = make_loader(
        cache, "val", args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if metrics_path.exists():
        metrics_path.unlink()
    best = float("inf")
    best_epoch = -1
    bad = 0
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, fps, optimizer, device, True)
        with torch.no_grad():
            va = run_epoch(model, val_loader, fps, optimizer, device, False)
        row = {"epoch": epoch, "train": tr, "val": va, "args": vars(args)}
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch:03d} | train BCE {tr['loss']:.4f} bit {tr['bit_acc']:.4f} | val BCE {va['loss']:.4f} bit {va['bit_acc']:.4f}"
        )
        if va["loss"] < best:
            best = va["loss"]
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val": va,
                    "args": vars(args),
                },
                out,
            )
            print(f"  saved best FP retrieval model -> {out}")
        else:
            bad += 1
            if bad >= args.early_stop_patience:
                print(
                    f"  early stopping after epoch {epoch}: best val BCE {best:.6f} at epoch {best_epoch}"
                )
                break
    ckpt = safe_torch_load(out, map_location=device)
    model.load_state_dict(ckpt["model"])
    ev = retrieval_eval(
        model, cache, fps, maccs, scafs, "val", device, args.eval_rows, args.topk
    )
    payload = {"best_epoch": best_epoch, "best_val": ckpt["val"], "retrieval_val": ev}
    out.with_suffix(".eval.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
