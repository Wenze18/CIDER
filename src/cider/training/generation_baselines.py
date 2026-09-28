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
from cider.utils import get_device, make_loader, safe_torch_load, set_seed
from cider.training.forward_generator import (
    StrongConditionEncoder,
    forward_model,
    masked_sequence_exact,
)


class LitResidualMLP(nn.Module):

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 3, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class ProfileDenoiser(nn.Module):

    def __init__(self, y_dim: int, d_model: int, dropout: float, depth: int):
        super().__init__()
        self.inp = nn.Sequential(
            nn.LayerNorm(y_dim),
            nn.Linear(y_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[LitResidualMLP(d_model, dropout) for _ in range(depth)]
        )
        self.out = nn.LayerNorm(d_model)

    def forward(self, y):
        return self.out(self.blocks(self.inp(y)))


class GRUConditionalDecoder(nn.Module):

    def __init__(
        self, vocab_size: int, pad_id: int, d_model: int, layers: int, dropout: float
    ):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.gru = nn.GRU(
            d_model,
            d_model,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.token_emb.weight

    def forward(self, input_tokens, h0):
        x = self.token_emb(input_tokens.long())
        out, _ = self.gru(x, h0)
        return self.out(self.norm(out))


class GxRNNGenerator(nn.Module):

    def __init__(
        self,
        y_dim: int,
        vocab_size: int,
        pad_id: int,
        n_cell: int,
        n_time: int,
        max_len: int,
        d_model: int,
        layers: int,
        dropout: float,
        noise_dim: int = 64,
        cond_depth: int = 3,
    ):
        super().__init__()
        self.max_len = max_len
        self.noise_dim = noise_dim
        self.layers = layers
        self.cond = StrongConditionEncoder(
            y_dim, n_cell, n_time, d_model, dropout, depth=cond_depth
        )
        self.noise = nn.Sequential(
            nn.Linear(noise_dim, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.h0 = nn.Sequential(nn.Linear(d_model * 2, d_model * layers), nn.Tanh())
        self.decoder = GRUConditionalDecoder(
            vocab_size, pad_id, d_model, layers, dropout
        )

    def forward(
        self, y, dose, cell, time, input_tokens, z: Optional[torch.Tensor] = None
    ):
        bsz = y.size(0)
        if z is None:
            z = torch.randn(bsz, self.noise_dim, dtype=y.dtype, device=y.device)
        h = torch.cat([self.cond(y, dose, cell, time), self.noise(z)], dim=-1)
        h0 = self.h0(h).view(bsz, self.layers, -1).transpose(0, 1).contiguous()
        return self.decoder(input_tokens, h0)


class Gex2SGenGenerator(nn.Module):

    def __init__(
        self,
        y_dim: int,
        vocab_size: int,
        pad_id: int,
        n_cell: int,
        n_time: int,
        max_len: int,
        d_model: int,
        layers: int,
        dropout: float,
        noise_dim: int = 128,
        cond_depth: int = 3,
    ):
        super().__init__()
        self.max_len = max_len
        self.noise_dim = noise_dim
        self.layers = layers
        self.profile_enc = StrongConditionEncoder(
            y_dim, n_cell, n_time, d_model, dropout, depth=cond_depth
        )
        self.profile_dec = nn.Sequential(
            nn.Linear(noise_dim, d_model), nn.GELU(), nn.Linear(d_model, y_dim)
        )
        self.prior = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, noise_dim * 2)
        )
        self.h0 = nn.Sequential(
            nn.Linear(d_model + noise_dim, d_model * layers), nn.Tanh()
        )
        self.decoder = GRUConditionalDecoder(
            vocab_size, pad_id, d_model, layers, dropout
        )

    @staticmethod
    def split_stats(x):
        mu, logvar = x.chunk(2, dim=-1)
        return (mu, logvar.clamp(-8.0, 6.0))

    def forward(
        self,
        y,
        dose,
        cell,
        time,
        input_tokens,
        z: Optional[torch.Tensor] = None,
        use_prior: bool = False,
    ):
        bsz = y.size(0)
        cond = self.profile_enc(y, dose, cell, time)
        mu, logvar = self.split_stats(self.prior(cond))
        kl = None
        if z is None:
            if self.training and (not use_prior):
                eps = torch.randn_like(mu)
                z = mu + eps * torch.exp(0.5 * logvar)
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
            else:
                z = mu
        h0 = (
            self.h0(torch.cat([cond, z], dim=-1))
            .view(bsz, self.layers, -1)
            .transpose(0, 1)
            .contiguous()
        )
        logits = self.decoder(input_tokens, h0)
        if kl is None:
            return logits
        recon = self.profile_dec(z)
        profile_rec = F.mse_loss(recon, y)
        return (logits, kl, profile_rec)


class TransGEMGenerator(nn.Module):

    def __init__(
        self,
        y_dim: int,
        vocab_size: int,
        pad_id: int,
        n_cell: int,
        n_time: int,
        max_len: int,
        d_model: int,
        layers: int,
        heads: int,
        dropout: float,
        noise_dim: int = 64,
        bins: int = 64,
    ):
        super().__init__()
        self.max_len = max_len
        self.noise_dim = noise_dim
        self.bins = bins
        self.pad_id = pad_id
        self.gene_id = nn.Embedding(y_dim, d_model)
        self.value_bin = nn.Embedding(bins, d_model)
        self.dose = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.cell = nn.Embedding(n_cell, d_model)
        self.time = nn.Embedding(n_time, d_model)
        self.noise = nn.Sequential(
            nn.Linear(noise_dim, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.mem_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.token_emb.weight
        nn.init.normal_(self.pos, std=0.02)

    def _memory(self, y, dose, cell, time, z):
        bsz, y_dim = y.shape
        clipped = y.float().clamp(-10, 10)
        bins = torch.clamp(
            ((clipped + 10.0) / 20.0 * self.bins).long(), 0, self.bins - 1
        )
        gene = self.gene_id.weight[None, :, :].expand(bsz, -1, -1)
        val = self.value_bin(bins)
        mem = gene + val
        if dose.ndim == 1:
            dose = dose[:, None]
        special = torch.stack(
            [
                self.dose(dose.float()),
                self.cell(cell.long()),
                self.time(time.long()),
                self.noise(z),
            ],
            dim=1,
        )
        return self.mem_proj(torch.cat([special, mem], dim=1))

    @staticmethod
    def causal_mask(length, device):
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def forward(
        self, y, dose, cell, time, input_tokens, z: Optional[torch.Tensor] = None
    ):
        bsz, length = input_tokens.shape
        if z is None:
            z = torch.randn(bsz, self.noise_dim, dtype=y.dtype, device=y.device)
        mem = self._memory(y, dose, cell, time, z)
        x = self.token_emb(input_tokens.long()) + self.pos[:, :length, :]
        mask = self.causal_mask(length, y.device)
        key_padding = input_tokens.eq(self.pad_id)
        out = self.decoder(x, mem, tgt_mask=mask, tgt_key_padding_mask=key_padding)
        return self.out(self.norm(out))


class FAMEGenerator(nn.Module):

    def __init__(
        self,
        y_dim: int,
        vocab_size: int,
        pad_id: int,
        n_cell: int,
        n_time: int,
        max_len: int,
        d_model: int,
        layers: int,
        dropout: float,
        noise_dim: int = 128,
        cond_depth: int = 4,
    ):
        super().__init__()
        self.max_len = max_len
        self.noise_dim = noise_dim
        self.layers = layers
        self.profile = ProfileDenoiser(y_dim, d_model, dropout, cond_depth)
        self.dose = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.cell = nn.Embedding(n_cell, d_model)
        self.time = nn.Embedding(n_time, d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 6, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.prior = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, noise_dim * 2)
        )
        self.h0 = nn.Sequential(
            nn.Linear(d_model + noise_dim, d_model * layers), nn.Tanh()
        )
        self.decoder = GRUConditionalDecoder(
            vocab_size, pad_id, d_model, layers, dropout
        )

    @staticmethod
    def split_stats(x):
        mu, logvar = x.chunk(2, dim=-1)
        return (mu, logvar.clamp(-8.0, 6.0))

    def cond(self, y, dose, cell, time):
        if dose.ndim == 1:
            dose = dose[:, None]
        hp = self.profile(y)
        hd = self.dose(dose.float())
        hc = self.cell(cell.long())
        ht = self.time(time.long())
        return self.fuse(torch.cat([hp, hd, hc, ht, hp * hc, hp * ht], dim=-1))

    def forward(
        self,
        y,
        dose,
        cell,
        time,
        input_tokens,
        z: Optional[torch.Tensor] = None,
        use_prior: bool = False,
    ):
        bsz = y.size(0)
        cond = self.cond(y, dose, cell, time)
        mu, logvar = self.split_stats(self.prior(cond))
        kl = None
        if z is None:
            if self.training and (not use_prior):
                eps = torch.randn_like(mu)
                z = mu + eps * torch.exp(0.5 * logvar)
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
            else:
                z = mu
        h0 = (
            self.h0(torch.cat([cond, z], dim=-1))
            .view(bsz, self.layers, -1)
            .transpose(0, 1)
            .contiguous()
        )
        logits = self.decoder(input_tokens, h0)
        if kl is None:
            return logits
        return (logits, kl)


def make_lit_model(
    method: str, cfg: dict, args=None, model_args: Optional[dict] = None
) -> nn.Module:
    if model_args is None:
        model_args = {
            "y_dim": cfg["y_dim"],
            "vocab_size": cfg["vocab_size"],
            "pad_id": cfg["pad_id"],
            "n_cell": cfg["n_cell"],
            "n_time": cfg["n_time"],
            "max_len": cfg["max_len"],
            "d_model": args.d_model,
            "layers": args.layers,
            "heads": args.heads,
            "dropout": args.dropout,
            "noise_dim": args.noise_dim,
            "cond_depth": args.cond_depth,
        }
    if method == "gxrnn":
        return GxRNNGenerator(**{k: v for k, v in model_args.items() if k != "heads"})
    if method == "gex2sgen":
        return Gex2SGenGenerator(
            **{k: v for k, v in model_args.items() if k != "heads"}
        )
    if method == "transgem":
        return TransGEMGenerator(
            **{k: v for k, v in model_args.items() if k != "cond_depth"}
        )
    if method == "fame":
        return FAMEGenerator(**{k: v for k, v in model_args.items() if k != "heads"})
    raise ValueError(method)


def make_model_from_lit_checkpoint(ckpt: dict) -> nn.Module:
    return make_lit_model(ckpt["method"], {}, model_args=ckpt["model_args"])


def make_scheduler(
    optimizer, total_steps: int, warmup_frac: float, min_lr_ratio: float
):
    warmup = int(total_steps * warmup_frac)

    def fn(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * min(1, progress))
        )

    return LambdaLR(optimizer, fn)


def model_forward(model, ctp, dose, cell, time, inp, fixed_zero_noise: bool):
    if fixed_zero_noise and hasattr(model, "noise_dim"):
        z = torch.zeros(
            ctp.size(0), int(model.noise_dim), device=ctp.device, dtype=ctp.dtype
        )
        return model(ctp, dose, cell, time, inp, z=z)
    return model(ctp, dose, cell, time, inp)


def run_epoch(
    model, loader, optimizer, scheduler, device, train: bool, cfg: dict, args
):
    model.train(train)
    total_loss = total_ce = total_acc = total_seq = total_kl = total_prof = n_tok = (
        n_seq
    ) = 0.0
    pbar = tqdm(loader, desc="train" if train else "eval", leave=False)
    amp_enabled = args.amp and device.type == "cuda"
    for batch in pbar:
        ctp = batch["ctp"].to(device, non_blocking=True)
        dose = batch["dose"].to(device, non_blocking=True)
        cell = batch["cell"].to(device, non_blocking=True)
        time = batch["time"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)
        inp = tokens[:, :-1].clone()
        tgt = tokens[:, 1:]
        if train and args.token_dropout > 0:
            drop = (torch.rand_like(inp.float()) < args.token_dropout) & inp.ne(
                cfg["pad_id"]
            )
            drop[:, 0] = False
            inp = inp.masked_fill(drop, cfg["pad_id"])
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp_enabled
        ):
            out = model_forward(
                model, ctp, dose, cell, time, inp, args.fixed_zero_noise
            )
            kl = prof = None
            if isinstance(out, tuple):
                logits = out[0]
                if len(out) > 1:
                    kl = out[1]
                if len(out) > 2:
                    prof = out[2]
            else:
                logits = out
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                ignore_index=cfg["pad_id"],
                label_smoothing=args.label_smoothing,
            )
            loss = ce
            if kl is not None:
                if args.free_bits > 0:
                    kl = torch.clamp(kl, min=args.free_bits)
                loss = loss + args.beta_kl * kl
            if prof is not None:
                loss = loss + args.profile_recon_weight * prof
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
        with torch.no_grad():
            mask = tgt.ne(cfg["pad_id"])
            pred = logits.argmax(dim=-1)
            correct = (pred.eq(tgt) & mask).sum().item()
            count = mask.sum().item()
            seq = masked_sequence_exact(pred, tgt, cfg["pad_id"])
        bsz = tokens.size(0)
        total_loss += float(loss.item()) * max(count, 1)
        total_ce += float(ce.item()) * max(count, 1)
        total_acc += correct
        total_seq += seq
        total_kl += float(kl.item()) * bsz if kl is not None else 0.0
        total_prof += float(prof.item()) * bsz if prof is not None else 0.0
        n_tok += count
        n_seq += bsz
        pbar.set_postfix(loss=total_loss / max(n_tok, 1), acc=total_acc / max(n_tok, 1))
    return {
        "loss": total_loss / max(n_tok, 1),
        "ce": total_ce / max(n_tok, 1),
        "token_acc": total_acc / max(n_tok, 1),
        "seq_acc": total_seq / max(n_seq, 1),
        "kl": total_kl / max(n_seq, 1),
        "profile_recon": total_prof / max(n_seq, 1),
    }


def main():
    p = argparse.ArgumentParser(
        description="Train gene-expression-guided molecular generation baselines."
    )
    p.add_argument("--cache", default="data/processed/cache.pt")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--method", choices=["fame", "gex2sgen", "transgem", "gxrnn"], required=True
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--early_stop_patience", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--noise_dim", type=int, default=128)
    p.add_argument("--cond_depth", type=int, default=3)
    p.add_argument("--label_smoothing", type=float, default=0.02)
    p.add_argument("--token_dropout", type=float, default=0.02)
    p.add_argument("--beta_kl", type=float, default=0.01)
    p.add_argument("--free_bits", type=float, default=0.0)
    p.add_argument("--profile_recon_weight", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--fixed_zero_noise", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    cache = safe_torch_load(args.cache, map_location="cpu")
    cfg = cache["config"]
    train_loader = make_loader(
        cache, "train", args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = make_loader(
        cache, "val", args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    model = make_lit_model(args.method, cfg, args=args).to(device)
    n_params = sum((p0.numel() for p0 in model.parameters()))
    print("device:", device)
    print("method:", args.method)
    print("params:", n_params)
    print("args:", vars(args))
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(
        optimizer, args.epochs * len(train_loader), args.warmup_frac, args.min_lr_ratio
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = out.with_suffix(".jsonl")
    best = float("inf")
    bad = 0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            model, train_loader, optimizer, scheduler, device, True, cfg, args
        )
        with torch.no_grad():
            va = run_epoch(
                model, val_loader, optimizer, scheduler, device, False, cfg, args
            )
        row = {
            "epoch": epoch,
            "train": tr,
            "val": va,
            "args": vars(args),
            "n_params": n_params,
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch:03d} | train CE {tr['ce']:.4f} acc {tr['token_acc']:.4f} | val CE {va['ce']:.4f} acc {va['token_acc']:.4f} seq {va['seq_acc']:.4f}"
        )
        if va["ce"] < best:
            best = va["ce"]
            bad = 0
            best_epoch = epoch
            model_args = {
                "y_dim": cfg["y_dim"],
                "vocab_size": cfg["vocab_size"],
                "pad_id": cfg["pad_id"],
                "n_cell": cfg["n_cell"],
                "n_time": cfg["n_time"],
                "max_len": cfg["max_len"],
                "d_model": args.d_model,
                "layers": args.layers,
                "heads": args.heads,
                "dropout": args.dropout,
                "noise_dim": args.noise_dim,
                "cond_depth": args.cond_depth,
            }
            torch.save(
                {
                    "model": model.state_dict(),
                    "method": args.method,
                    "model_args": model_args,
                    "args": vars(args),
                    "cache_config": cfg,
                    "epoch": epoch,
                    "val": va,
                    "train": tr,
                    "n_params": n_params,
                    "note": "Literature benchmark adaptation trained on train split and selected by validation CE.",
                },
                out,
            )
            print(f"  saved best -> {out}")
        else:
            bad += 1
            if bad >= args.early_stop_patience:
                print(f"early stopping at epoch {epoch}, best epoch {best_epoch}")
                break


if __name__ == "__main__":
    main()
