from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from cider.utils import get_device, make_loader, safe_torch_load, set_seed
from cider.training.fingerprint import load_or_build_fps
from cider.training.forward_generator import ResidualMLP, StrongConditionEncoder


class TimeEmbedding(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.net[0].in_features // 2
        freqs = torch.exp(
            torch.linspace(
                np.log(1.0), np.log(1000.0), half, device=t.device, dtype=t.dtype
            )
        )
        ang = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.net(emb)


class FingerprintFlow(nn.Module):

    def __init__(
        self,
        y_dim: int,
        n_cell: int,
        n_time: int,
        fp_dim: int,
        d_model: int,
        depth: int,
        dropout: float,
    ):
        super().__init__()
        self.cond = StrongConditionEncoder(
            y_dim, n_cell, n_time, d_model, dropout, depth=3
        )
        self.x_in = nn.Sequential(
            nn.Linear(fp_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.t_emb = TimeEmbedding(d_model)
        self.blocks = nn.Sequential(
            *[
                ResidualMLP(d_model, hidden_mult=4, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, fp_dim),
        )

    def forward(self, x_t, t, ctp, dose, cell, time):
        h = self.x_in(x_t) + self.t_emb(t) + self.cond(ctp, dose, cell, time)
        return self.out(self.blocks(h))


def run_epoch(model, loader, fps, optimizer, device, train: bool):
    model.train(train)
    total_loss = total_cos = n = 0
    for batch in tqdm(loader, desc="train flow" if train else "eval flow", leave=False):
        ctp = batch["ctp"].to(device)
        dose = batch["dose"].to(device)
        cell = batch["cell"].to(device)
        time = batch["time"].to(device)
        row_index = batch["row_index"]
        x1 = fps[row_index].to(device)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.size(0), device=device).clamp(0.0001, 1 - 0.0001)
        x_t = (1.0 - t[:, None]) * x0 + t[:, None] * x1
        target_v = x1 - x0
        if train:
            optimizer.zero_grad(set_to_none=True)
        pred_v = model(x_t, t, ctp, dose, cell, time)
        mse = F.mse_loss(pred_v, target_v)
        cos = F.cosine_similarity(pred_v, target_v, dim=1).mean()
        loss = mse + 0.05 * (1.0 - cos)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += float(loss.item()) * x1.size(0)
        total_cos += float(cos.item()) * x1.size(0)
        n += x1.size(0)
    return {"loss": total_loss / max(n, 1), "cos": total_cos / max(n, 1)}


@torch.no_grad()
def sample_fp(model, ctp, dose, cell, time, fp_dim: int, steps: int, n_samples: int):
    ctp = ctp.repeat(n_samples, 1)
    dose = dose.repeat(n_samples)
    cell = cell.repeat(n_samples)
    time = time.repeat(n_samples)
    x = torch.randn(n_samples, fp_dim, device=ctp.device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n_samples,), (i + 0.5) / steps, device=ctp.device)
        x = x + dt * model(x, t, ctp, dose, cell, time)
    return x.clamp(0.0, 1.0)


@torch.no_grad()
def retrieval_eval(
    model,
    cache,
    fps,
    maccs,
    scafs,
    split: str,
    device,
    rows: int,
    topk: int,
    samples: int,
    steps: int,
):
    model.eval()
    train_idx_cpu = torch.as_tensor(cache["splits"]["train"], dtype=torch.long)
    train_idx = train_idx_cpu.to(device)
    train_fps = fps[train_idx_cpu].to(device)
    train_sum = train_fps.sum(dim=1)
    fp_dim = train_fps.size(1)
    max_morgan = []
    mean_morgan = []
    max_maccs = []
    hit03 = hit05 = hit07 = scaf_hit = 0
    for idx in tqdm(
        list(map(int, cache["splits"][split]))[:rows],
        desc="flow retrieval eval",
        leave=False,
    ):
        ctp = cache["X"][idx].to(device).view(1, -1)
        dose = cache["dose"][idx].to(device).view(1)
        cell = cache["cell"][idx].to(device).view(1)
        time = cache["time"][idx].to(device).view(1)
        probs = sample_fp(model, ctp, dose, cell, time, fp_dim, steps, samples)
        inter = probs @ train_fps.T
        denom = probs.sum(dim=1, keepdim=True) + train_sum[None, :] - inter
        score = inter / denom.clamp_min(1e-06)
        flat = score.reshape(-1)
        nn_pos = torch.topk(flat, k=topk).indices % train_fps.size(0)
        nn_idx = [int(train_idx[int(p)].item()) for p in nn_pos.tolist()]
        gt = fps[idx].to(device)
        cand = fps[nn_idx].to(device)
        hard_inter = (cand * gt[None, :]).sum(dim=1)
        hard_denom = cand.sum(dim=1) + gt.sum() - hard_inter
        sims = (hard_inter / hard_denom.clamp_min(1e-06)).cpu().numpy()
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
        max_maccs.append(
            float(torch.max(mac_inter / mac_denom.clamp_min(1e-06)).item())
        )
        gt_scaf = scafs[idx]
        scaf_hit += int(
            any((gt_scaf and scafs[j] and (scafs[j] == gt_scaf) for j in nn_idx))
        )
    n = len(max_morgan)
    return {
        "split": split,
        "rows": n,
        "topk": topk,
        "samples": samples,
        "steps": steps,
        "gt_max_morgan": float(np.mean(max_morgan)),
        "gt_mean_morgan": float(np.mean(mean_morgan)),
        "gt_hit_morgan_0.3": hit03 / max(n, 1),
        "gt_hit_morgan_0.5": hit05 / max(n, 1),
        "gt_hit_morgan_0.7": hit07 / max(n, 1),
        "gt_max_maccs": float(np.mean(max_maccs)),
        "gt_scaffold_hit": scaf_hit / max(n, 1),
        "validity": 1.0,
        "novel_unique_per_sample": 0.0,
    }


def main():
    p = argparse.ArgumentParser(
        description="Forward conditional flow matching in Morgan fingerprint space."
    )
    p.add_argument("--cache", default="data/processed/cache.pt")
    p.add_argument("--fp_cache", default="data/features/fingerprints_2048.pt")
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
    p.add_argument("--eval_rows", type=int, default=512)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--steps", type=int, default=16)
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
        cache, args.fp_dim, args.fp_cache, fp_workers=8
    )
    model = FingerprintFlow(
        cfg["y_dim"],
        cfg["n_cell"],
        cfg["n_time"],
        args.fp_dim,
        args.d_model,
        args.depth,
        args.dropout,
    ).to(device)
    if args.eval_only:
        ckpt = safe_torch_load(args.out, map_location=device)
        model.load_state_dict(ckpt["model"])
        ev = retrieval_eval(
            model,
            cache,
            fps,
            maccs,
            scafs,
            "val",
            device,
            args.eval_rows,
            args.topk,
            args.samples,
            args.steps,
        )
        payload = {
            "best_epoch": ckpt.get("epoch", -1),
            "best_val": ckpt.get("val", {}),
            "retrieval_val": ev,
        }
        Path(args.out).with_suffix(".eval.json").write_text(
            json.dumps(payload, indent=2)
        )
        print(json.dumps(payload, indent=2))
        return
    train_loader = make_loader(
        cache, "train", args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = make_loader(
        cache, "val", args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_jsonl)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists():
        metrics_path.unlink()
    best = float("inf")
    best_epoch = -1
    bad = 0
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, fps, optimizer, device, True)
        va = run_epoch(model, val_loader, fps, optimizer, device, False)
        row = {"epoch": epoch, "train": tr, "val": va, "args": vars(args)}
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch:03d} | train loss {tr['loss']:.4f} cos {tr['cos']:.4f} | val loss {va['loss']:.4f} cos {va['cos']:.4f}"
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
            print(f"  saved best fp flow -> {out}")
        else:
            bad += 1
            if bad >= args.early_stop_patience:
                print(
                    f"  early stopping after epoch {epoch}: best val loss {best:.6f} at epoch {best_epoch}"
                )
                break
    ckpt = safe_torch_load(out, map_location=device)
    model.load_state_dict(ckpt["model"])
    ev = retrieval_eval(
        model,
        cache,
        fps,
        maccs,
        scafs,
        "val",
        device,
        args.eval_rows,
        args.topk,
        args.samples,
        args.steps,
    )
    payload = {"best_epoch": best_epoch, "best_val": ckpt["val"], "retrieval_val": ev}
    out.with_suffix(".eval.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
