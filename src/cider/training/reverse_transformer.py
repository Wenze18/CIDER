from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from cider.models import ContextEncoderNoCTP, MolTransformerEncoder, ReversePredictor
from cider.utils import (
    batch_cosine,
    batch_pearson,
    get_device,
    make_loader,
    safe_torch_load,
    set_seed,
)


def pearson_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-08
) -> torch.Tensor:
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    target_c = target - target.mean(dim=1, keepdim=True)
    corr = (pred_c * target_c).sum(dim=1) / (
        pred_c.norm(dim=1) * target_c.norm(dim=1) + eps
    )
    return 1.0 - corr.mean()


def cosine_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-08
) -> torch.Tensor:
    pred_n = pred / (pred.norm(dim=1, keepdim=True) + eps)
    target_n = target / (target.norm(dim=1, keepdim=True) + eps)
    return 1.0 - (pred_n * target_n).sum(dim=1).mean()


class ResidualMLPBlock(nn.Module):

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class AttentiveMolTransformerEncoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_len: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        cond_prefix_len: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.cond_prefix_len = cond_prefix_len
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len + cond_prefix_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward_tokens(
        self, token_ids: torch.Tensor, prefix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        token_ids = token_ids[:, : self.max_len]
        pad_mask = token_ids.eq(self.pad_id)
        emb = self.token_emb(token_ids.long())
        if prefix is not None:
            emb = torch.cat([prefix, emb], dim=1)
            prefix_mask = torch.zeros(
                token_ids.size(0),
                prefix.size(1),
                dtype=torch.bool,
                device=token_ids.device,
            )
            pad_mask = torch.cat([prefix_mask, pad_mask], dim=1)
        emb = emb + self.pos_emb[:, : emb.size(1), :]
        out = self.encoder(emb, src_key_padding_mask=pad_mask)
        out = self.norm(out)
        if prefix is not None:
            out = out[:, prefix.size(1) :, :]
            pad_mask = pad_mask[:, prefix.size(1) :]
        scores = self.attn(out).squeeze(-1).masked_fill(pad_mask, -10000.0)
        weights = torch.softmax(scores, dim=1)
        return (out * weights[:, :, None]).sum(dim=1)


class TransformerReversePredictor(nn.Module):

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
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        arch: str = "attn",
        cond_prefix_len: int = 0,
        head_blocks: int = 2,
        skip_heads: bool = False,
    ):
        super().__init__()
        self.arch = arch
        self.cond_prefix_len = cond_prefix_len if arch == "prefix" else 0
        if arch == "mean":
            self.mol_encoder = MolTransformerEncoder(
                vocab_size=vocab_size,
                pad_id=pad_id,
                max_len=max_len,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
        else:
            self.mol_encoder = AttentiveMolTransformerEncoder(
                vocab_size=vocab_size,
                pad_id=pad_id,
                max_len=max_len,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                cond_prefix_len=self.cond_prefix_len,
            )
        self.ctx_encoder = ContextEncoderNoCTP(
            n_cell=n_cell, n_time=n_time, d_model=d_model, dropout=dropout
        )
        if self.cond_prefix_len:
            self.prefix_proj = nn.Linear(d_model, self.cond_prefix_len * d_model)
        else:
            self.prefix_proj = None
        in_dim = d_model * 4
        fused_dim = d_model * 4
        self.fuse = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualMLPBlock(fused_dim, mult=2, dropout=dropout)
                for _ in range(head_blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(fused_dim)
        self.out = nn.Linear(fused_dim, y_dim)
        self.skip_heads = skip_heads
        if skip_heads:
            self.ctx_skip = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, y_dim)
            )
            self.mol_skip = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, y_dim)
            )
        else:
            self.ctx_skip = None
            self.mol_skip = None

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        dose: torch.Tensor,
        cell: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        h_ctx = self.ctx_encoder(dose, cell, time)
        prefix = None
        if self.prefix_proj is not None:
            prefix = self.prefix_proj(h_ctx).view(
                tokens.size(0), self.cond_prefix_len, -1
            )
        if self.arch == "mean":
            h_m = self.mol_encoder.forward_tokens(tokens)
        else:
            h_m = self.mol_encoder.forward_tokens(tokens, prefix=prefix)
        h = self.fuse(
            torch.cat([h_m, h_ctx, h_m * h_ctx, torch.abs(h_m - h_ctx)], dim=-1)
        )
        h = self.blocks(h)
        y = self.out(self.out_norm(h))
        if self.ctx_skip is not None and self.mol_skip is not None:
            y = y + self.ctx_skip(h_ctx) + self.mol_skip(h_m)
        return y


def make_model(args, cfg: dict) -> nn.Module:
    common = dict(
        y_dim=cfg["y_dim"],
        vocab_size=cfg["vocab_size"],
        pad_id=cfg["pad_id"],
        n_cell=cfg["n_cell"],
        n_time=cfg["n_time"],
        max_len=cfg["max_len"],
        d_model=args.d_model,
        nhead=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
    )
    if args.arch == "baseline":
        return ReversePredictor(**common)
    return TransformerReversePredictor(
        **common,
        dim_feedforward=args.ffn_dim,
        arch=args.arch,
        cond_prefix_len=args.cond_prefix_len,
        head_blocks=args.head_blocks,
        skip_heads=args.skip_heads,
    )


def make_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float):
    if total_steps <= 0:
        return None

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * min(1.0, progress))
        )

    return LambdaLR(optimizer, lr_lambda)


def apply_token_dropout(
    tokens: torch.Tensor, pad_id: int, bos_id: int, eos_id: int, unk_id: int, p: float
) -> torch.Tensor:
    if p <= 0.0:
        return tokens
    keep = tokens.eq(pad_id) | tokens.eq(bos_id) | tokens.eq(eos_id)
    drop = (torch.rand_like(tokens.float()) < p) & ~keep
    return torch.where(drop, torch.full_like(tokens, unk_id), tokens)


def run_epoch(
    model, loader, optimizer, scheduler, device, train: bool, args, cfg: dict
):
    model.train(train)
    total_loss = total_mse = total_huber = total_cos = total_pear = n = 0
    pbar = tqdm(loader, desc="train F" if train else "eval F", leave=False)
    amp_enabled = args.amp and device.type == "cuda"
    for batch in pbar:
        ctp = batch["ctp"].to(device, non_blocking=True)
        dose = batch["dose"].to(device, non_blocking=True)
        cell = batch["cell"].to(device, non_blocking=True)
        time = batch["time"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)
        if train and args.token_dropout > 0.0:
            tokens = apply_token_dropout(
                tokens,
                cfg["pad_id"],
                cfg["bos_id"],
                cfg["eos_id"],
                cfg["unk_id"],
                args.token_dropout,
            )
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp_enabled
        ):
            pred = model.forward_tokens(tokens, dose, cell, time)
            mse = F.mse_loss(pred, ctp)
            huber = F.smooth_l1_loss(pred, ctp, beta=args.huber_beta)
            cos = cosine_loss(pred, ctp)
            pear = pearson_loss(pred, ctp)
            loss = (
                args.mse_weight * mse
                + args.huber_weight * huber
                + args.cos_weight * cos
                + args.pearson_weight * pear
            )
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        bs = ctp.size(0)
        total_loss += float(loss.item()) * bs
        total_mse += float(mse.item()) * bs
        total_huber += float(huber.item()) * bs
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
    p = argparse.ArgumentParser(description="Train a Transformer response predictor.")
    p.add_argument("--cache", default="data/processed/cache.pt")
    p.add_argument("--out", default="checkpoints/reverse/transformer.pt")
    p.add_argument("--metrics_jsonl", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--mse_weight", type=float, default=1.0)
    p.add_argument("--huber_weight", type=float, default=0.0)
    p.add_argument("--huber_beta", type=float, default=0.5)
    p.add_argument("--cos_weight", type=float, default=0.2)
    p.add_argument("--pearson_weight", type=float, default=0.0)
    p.add_argument(
        "--arch", choices=["baseline", "mean", "attn", "prefix"], default="attn"
    )
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ffn_dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--cond_prefix_len", type=int, default=4)
    p.add_argument("--head_blocks", type=int, default=2)
    p.add_argument("--skip_heads", action="store_true")
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
    model = make_model(args, cfg).to(device)
    n_params = sum((p.numel() for p in model.parameters()))
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
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            train=True,
            args=args,
            cfg=cfg,
        )
        with torch.no_grad():
            va = run_epoch(
                model,
                val_loader,
                optimizer,
                scheduler,
                device,
                train=False,
                args=args,
                cfg=cfg,
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
        if va["loss"] < best:
            best = va["loss"]
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
                        "num_layers": args.layers,
                        "dim_feedforward": args.ffn_dim,
                        "dropout": args.dropout,
                        "arch": args.arch,
                        "cond_prefix_len": args.cond_prefix_len,
                        "head_blocks": args.head_blocks,
                        "skip_heads": args.skip_heads,
                    },
                    "cache_config": cfg,
                    "epoch": epoch,
                    "val": va,
                    "train": tr,
                    "args": vars(args),
                    "n_params": n_params,
                    "model_class": (
                        "ReversePredictor"
                        if args.arch == "baseline"
                        else "TransformerReversePredictor"
                    ),
                },
                out,
            )
            print(f"  saved best Transformer response predictor -> {out}")


if __name__ == "__main__":
    main()
