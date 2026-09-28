from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from cider.utils import (
    batch_cosine,
    batch_pearson,
    get_device,
    make_loader,
    safe_torch_load,
    set_seed,
)


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


class ConditionEncoder(nn.Module):

    def __init__(self, n_cell: int, n_time: int, d_model: int, dropout: float):
        super().__init__()
        self.dose = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.cell = nn.Embedding(n_cell, d_model)
        self.time = nn.Embedding(n_time, d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self, dose: torch.Tensor, cell: torch.Tensor, time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dose.ndim == 1:
            dose = dose[:, None]
        d = self.dose(dose.float())
        c = self.cell(cell.long())
        t = self.time(time.long())
        ctx = self.fuse(torch.cat([d, c, t], dim=-1))
        return (ctx, torch.stack([d, c, t], dim=1))


class AdaLNBlock(nn.Module):

    def __init__(self, d_model: int, nhead: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model * 4))

    def _modulate(
        self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return x * (1 + scale[:, None, :]) + shift[:, None, :]

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        s1, g1, s2, g2 = self.mod(ctx).chunk(4, dim=-1)
        h = self._modulate(self.norm1(x), s1, g1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        h = self._modulate(self.norm2(x), s2, g2)
        x = x + self.ffn(h)
        return x


class PerceiverCrossAttention(nn.Module):

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h, _ = self.attn(
            self.q_norm(query),
            self.kv_norm(kv),
            self.kv_norm(kv),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query = query + h
        return query + self.ffn(query)


class PerceiverReversePredictor(nn.Module):

    def __init__(
        self,
        y_dim: int,
        vocab_size: int,
        pad_id: int,
        n_cell: int,
        n_time: int,
        max_len: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        latent_len: int = 64,
        latent_layers: int = 6,
        decoder_layers: int = 2,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        ensemble_size: int = 4,
    ):
        super().__init__()
        self.y_dim = y_dim
        self.pad_id = pad_id
        self.max_len = max_len
        self.ensemble_size = ensemble_size
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.type_emb = nn.Embedding(4, d_model)
        self.cond = ConditionEncoder(
            n_cell=n_cell, n_time=n_time, d_model=d_model, dropout=dropout
        )
        self.latents = nn.Parameter(torch.randn(1, latent_len, d_model) * 0.02)
        self.input_cross = PerceiverCrossAttention(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        self.latent_blocks = nn.ModuleList(
            [
                AdaLNBlock(d_model, nhead, ffn_mult, dropout)
                for _ in range(latent_layers)
            ]
        )
        self.gene_queries = nn.Parameter(torch.randn(1, y_dim, d_model) * 0.02)
        self.decoder_cross = nn.ModuleList(
            [
                PerceiverCrossAttention(d_model=d_model, nhead=nhead, dropout=dropout)
                for _ in range(decoder_layers)
            ]
        )
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.output_mixer = nn.TransformerEncoder(dec_layer, num_layers=1)
        self.baseline = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, y_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, ensemble_size),
        )
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward_members(
        self,
        tokens: torch.Tensor,
        dose: torch.Tensor,
        cell: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        tokens = tokens[:, : self.max_len]
        B, L = tokens.shape
        pad_mask = tokens.eq(self.pad_id)
        x = (
            self.token_emb(tokens.long())
            + self.pos_emb[:, :L, :]
            + self.type_emb.weight[0][None, None, :]
        )
        ctx, cond_tokens = self.cond(dose, cell, time)
        cond_tokens = (
            cond_tokens
            + self.type_emb(torch.tensor([1, 2, 3], device=tokens.device))[None, :, :]
        )
        inp = torch.cat([x, cond_tokens], dim=1)
        input_mask = torch.cat(
            [pad_mask, torch.zeros(B, 3, dtype=torch.bool, device=tokens.device)], dim=1
        )
        z = self.latents.expand(B, -1, -1)
        z = self.input_cross(z, inp, key_padding_mask=input_mask)
        for block in self.latent_blocks:
            z = block(z, ctx)
        q = self.gene_queries.expand(B, -1, -1)
        for cross in self.decoder_cross:
            q = cross(q, z)
        q = self.output_mixer(q)
        delta_members = self.head(q).transpose(1, 2)
        base = self.baseline(ctx)[:, None, :]
        return delta_members + base

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        dose: torch.Tensor,
        cell: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_members(tokens, dose, cell, time).mean(dim=1)


def make_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float):

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * min(1.0, progress))
        )

    return LambdaLR(optimizer, lr_lambda)


def apply_token_dropout(tokens: torch.Tensor, cfg: dict, p: float) -> torch.Tensor:
    if p <= 0:
        return tokens
    keep = (
        tokens.eq(cfg["pad_id"]) | tokens.eq(cfg["bos_id"]) | tokens.eq(cfg["eos_id"])
    )
    drop = (torch.rand_like(tokens.float()) < p) & ~keep
    return torch.where(drop, torch.full_like(tokens, cfg["unk_id"]), tokens)


def profile_loss(
    pred: torch.Tensor, target: torch.Tensor, args
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
    return (loss, {"mse": mse, "huber": huber, "cos_loss": cos, "pear_loss": pear})


def run_epoch(
    model, loader, optimizer, scheduler, device, train: bool, args, cfg: dict
):
    model.train(train)
    total_loss = total_mse = total_huber = total_cos = total_pear = n = 0
    pbar = tqdm(
        loader, desc="train Perceiver" if train else "eval Perceiver", leave=False
    )
    amp_enabled = args.amp and device.type == "cuda"
    for batch in pbar:
        ctp = batch["ctp"].to(device, non_blocking=True)
        dose = batch["dose"].to(device, non_blocking=True)
        cell = batch["cell"].to(device, non_blocking=True)
        time = batch["time"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)
        if train:
            tokens = apply_token_dropout(tokens, cfg, args.token_dropout)
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp_enabled
        ):
            members = model.forward_members(tokens, dose, cell, time)
            pred = members.mean(dim=1)
            loss, parts = profile_loss(pred, ctp, args)
            if args.member_loss_weight > 0:
                member_loss = 0.0
                for k in range(members.size(1)):
                    member_loss = (
                        member_loss + profile_loss(members[:, k, :], ctp, args)[0]
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
        "loss": total_loss / max(n, 1),
        "mse": total_mse / max(n, 1),
        "huber": total_huber / max(n, 1),
        "cosine": total_cos / max(n, 1),
        "pearson": total_pear / max(n, 1),
    }


def main():
    p = argparse.ArgumentParser(description="Train Perceiver reverse predictor.")
    p.add_argument("--cache", default="data/processed/cache.pt")
    p.add_argument("--out", default="checkpoints/reverse/perceiver.pt")
    p.add_argument("--metrics_jsonl", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--mse_weight", type=float, default=0.7)
    p.add_argument("--huber_weight", type=float, default=0.3)
    p.add_argument("--cos_weight", type=float, default=0.25)
    p.add_argument("--pearson_weight", type=float, default=0.25)
    p.add_argument("--member_loss_weight", type=float, default=0.1)
    p.add_argument("--huber_beta", type=float, default=0.5)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--latent_len", type=int, default=64)
    p.add_argument("--latent_layers", type=int, default=6)
    p.add_argument("--decoder_layers", type=int, default=2)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--ensemble_size", type=int, default=4)
    p.add_argument("--token_dropout", type=float, default=0.0)
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
    train_loader = make_loader(
        cache, "train", args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = make_loader(
        cache, "val", args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    model = PerceiverReversePredictor(
        y_dim=cfg["y_dim"],
        vocab_size=cfg["vocab_size"],
        pad_id=cfg["pad_id"],
        n_cell=cfg["n_cell"],
        n_time=cfg["n_time"],
        max_len=cfg["max_len"],
        d_model=args.d_model,
        nhead=args.heads,
        latent_len=args.latent_len,
        latent_layers=args.latent_layers,
        decoder_layers=args.decoder_layers,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        ensemble_size=args.ensemble_size,
    ).to(device)
    n_params = sum((p0.numel() for p0 in model.parameters()))
    print(f"parameters: {n_params:,}")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = make_scheduler(
        optimizer, int(args.warmup_frac * total_steps), total_steps, args.min_lr_ratio
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = (
        Path(args.metrics_jsonl) if args.metrics_jsonl else out.with_suffix(".jsonl")
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            model, train_loader, optimizer, scheduler, device, True, args, cfg
        )
        with torch.no_grad():
            va = run_epoch(
                model, val_loader, optimizer, scheduler, device, False, args, cfg
            )
        row = {
            "epoch": epoch,
            "train": tr,
            "val": va,
            "args": vars(args),
            "n_params": n_params,
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
                        "y_dim": cfg["y_dim"],
                        "vocab_size": cfg["vocab_size"],
                        "pad_id": cfg["pad_id"],
                        "n_cell": cfg["n_cell"],
                        "n_time": cfg["n_time"],
                        "max_len": cfg["max_len"],
                        "d_model": args.d_model,
                        "nhead": args.heads,
                        "latent_len": args.latent_len,
                        "latent_layers": args.latent_layers,
                        "decoder_layers": args.decoder_layers,
                        "ffn_mult": args.ffn_mult,
                        "dropout": args.dropout,
                        "ensemble_size": args.ensemble_size,
                    },
                    "cache_config": cfg,
                    "epoch": epoch,
                    "val": va,
                    "train": tr,
                    "args": vars(args),
                    "n_params": n_params,
                    "model_class": "PerceiverReversePredictor",
                },
                out,
            )
            print(f"  saved best Perceiver -> {out}")


if __name__ == "__main__":
    main()
