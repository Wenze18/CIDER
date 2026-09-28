from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

try:
    import selfies as sf
except Exception:
    sf = None
try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import (
        AllChem,
        Descriptors,
        Lipinski,
        MACCSkeys,
        QED,
        rdMolDescriptors,
    )
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
except Exception:
    Chem = None
from cider.models import ForwardGenerator
from cider.utils import (
    get_device,
    make_loader,
    safe_torch_load,
    set_seed,
    top_k_top_p_filtering,
)


class ResidualMLP(nn.Module):

    def __init__(self, dim: int, hidden_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = dim * hidden_mult
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class StrongConditionEncoder(nn.Module):

    def __init__(
        self,
        y_dim: int,
        n_cell: int,
        n_time: int,
        d_model: int,
        dropout: float,
        depth: int = 3,
    ):
        super().__init__()
        self.y_in = nn.Sequential(
            nn.LayerNorm(y_dim),
            nn.Linear(y_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.y_blocks = nn.Sequential(
            *[
                ResidualMLP(d_model, hidden_mult=3, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.dose_net = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.cell_emb = nn.Embedding(n_cell, d_model)
        self.time_emb = nn.Embedding(n_time, d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 6, d_model * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        y: torch.Tensor,
        dose: torch.Tensor,
        cell: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if dose.ndim == 1:
            dose = dose[:, None]
        h_y = self.y_blocks(self.y_in(y))
        h_d = self.dose_net(dose.float())
        h_c = self.cell_emb(cell.long())
        h_t = self.time_emb(time.long())
        h = torch.cat([h_y, h_d, h_c, h_t, h_y * h_c, h_y * h_t], dim=-1)
        return self.fuse(h)


class AdaLNBlock(nn.Module):

    def __init__(self, d_model: int, nhead: int, dropout: float, ffn_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult * 2),
            nn.GLU(dim=-1),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )
        self.cond = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model * 6))
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.cond[-1].weight)
        nn.init.zeros_(self.cond[-1].bias)

    @staticmethod
    def modulate(
        x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return x * (1.0 + scale[:, None, :]) + shift[:, None, :]

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        s1, g1, a1, s2, g2, a2 = self.cond(cond).chunk(6, dim=-1)
        h = self.modulate(self.norm1(x), s1, g1)
        h, _ = self.attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(h) * torch.sigmoid(a1)[:, None, :]
        h = self.modulate(self.norm2(x), s2, g2)
        x = x + self.ffn(h) * torch.sigmoid(a2)[:, None, :]
        return x


class AdaLNForwardGenerator(nn.Module):

    def __init__(
        self,
        y_dim: int,
        vocab_size: int,
        pad_id: int,
        n_cell: int,
        n_time: int,
        max_len: int = 128,
        d_model: int = 384,
        nhead: int = 8,
        num_layers: int = 8,
        dropout: float = 0.1,
        noise_dim: int = 64,
        cond_depth: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.max_len = max_len
        self.noise_dim = noise_dim
        self.cond_encoder = StrongConditionEncoder(
            y_dim, n_cell, n_time, d_model, dropout, depth=cond_depth
        )
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.cond_fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.blocks = nn.ModuleList(
            [AdaLNBlock(d_model, nhead, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        if pad_id is not None:
            with torch.no_grad():
                self.token_emb.weight[pad_id].zero_()

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def forward(
        self,
        y: torch.Tensor,
        dose: torch.Tensor,
        cell: torch.Tensor,
        time: torch.Tensor,
        input_tokens: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, length = input_tokens.shape
        if length > self.max_len:
            input_tokens = input_tokens[:, : self.max_len]
            length = self.max_len
        cond = self.cond_encoder(y, dose, cell, time)
        if z is None:
            z = torch.randn(bsz, self.noise_dim, device=y.device, dtype=y.dtype)
        cond = self.cond_fuse(torch.cat([cond, self.noise_proj(z)], dim=-1))
        x = self.token_emb(input_tokens.long()) + self.pos_emb[:, :length, :]
        attn_mask = self._causal_mask(length, x.device)
        key_padding = input_tokens.eq(self.pad_id)
        for block in self.blocks:
            x = block(x, cond, attn_mask=attn_mask, key_padding_mask=key_padding)
        return self.lm_head(self.norm(x))


class CVAEAdaLNForwardGenerator(nn.Module):

    def __init__(
        self,
        y_dim: int,
        vocab_size: int,
        pad_id: int,
        n_cell: int,
        n_time: int,
        max_len: int = 128,
        d_model: int = 384,
        nhead: int = 8,
        num_layers: int = 8,
        dropout: float = 0.1,
        noise_dim: int = 64,
        cond_depth: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.max_len = max_len
        self.noise_dim = noise_dim
        self.cond_encoder = StrongConditionEncoder(
            y_dim, n_cell, n_time, d_model, dropout, depth=cond_depth
        )
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.posterior_mol = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.prior = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, noise_dim * 2),
        )
        self.posterior = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, noise_dim * 2),
        )
        self.z_proj = nn.Sequential(
            nn.Linear(noise_dim, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.cond_fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.blocks = nn.ModuleList(
            [AdaLNBlock(d_model, nhead, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        if pad_id is not None:
            with torch.no_grad():
                self.token_emb.weight[pad_id].zero_()

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    @staticmethod
    def _split_stats(stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = stats.chunk(2, dim=-1)
        return (mu, logvar.clamp(-8.0, 6.0))

    @staticmethod
    def _sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    @staticmethod
    def kl_normal(q_mu, q_logvar, p_mu, p_logvar) -> torch.Tensor:
        return 0.5 * (
            p_logvar
            - q_logvar
            + (torch.exp(q_logvar) + (q_mu - p_mu).pow(2))
            / torch.exp(p_logvar).clamp_min(1e-08)
            - 1.0
        ).sum(dim=-1)

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        emb = self.token_emb(tokens.long())
        mask = tokens.ne(self.pad_id).float()
        return (emb * mask[:, :, None]).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)

    def _decode(
        self, input_tokens: torch.Tensor, cond: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        bsz, length = input_tokens.shape
        if length > self.max_len:
            input_tokens = input_tokens[:, : self.max_len]
            length = self.max_len
        dec_cond = self.cond_fuse(torch.cat([cond, self.z_proj(z)], dim=-1))
        x = self.token_emb(input_tokens.long()) + self.pos_emb[:, :length, :]
        attn_mask = self._causal_mask(length, x.device)
        key_padding = input_tokens.eq(self.pad_id)
        for block in self.blocks:
            x = block(x, dec_cond, attn_mask=attn_mask, key_padding_mask=key_padding)
        return self.lm_head(self.norm(x))

    def forward(
        self,
        y: torch.Tensor,
        dose: torch.Tensor,
        cell: torch.Tensor,
        time: torch.Tensor,
        input_tokens: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        posterior_tokens: Optional[torch.Tensor] = None,
        use_prior: bool = False,
    ):
        cond = self.cond_encoder(y, dose, cell, time)
        p_mu, p_logvar = self._split_stats(self.prior(cond))
        kl = None
        if z is None:
            if posterior_tokens is not None and (not use_prior):
                mol_h = self.posterior_mol(self._pool_tokens(posterior_tokens))
                q_mu, q_logvar = self._split_stats(
                    self.posterior(torch.cat([cond, mol_h], dim=-1))
                )
                z = self._sample(q_mu, q_logvar)
                kl = self.kl_normal(q_mu, q_logvar, p_mu, p_logvar).mean()
            else:
                z = p_mu
        logits = self._decode(input_tokens, cond, z)
        if kl is None:
            return logits
        return (logits, kl)


class RetrievalIndex:

    def __init__(self, cache: dict, device: torch.device):
        train_idx = torch.as_tensor(cache["splits"]["train"], dtype=torch.long)
        train_x = cache["X"][train_idx].float()
        train_x = F.normalize(train_x, dim=1)
        self.train_idx = train_idx.to(device)
        self.train_x = train_x.to(device)
        self.train_tokens = cache["tokens"][train_idx].to(device)
        max_row = int(torch.as_tensor(cache["splits"]["train"]).max().item())
        max_row = max(
            max_row, int(torch.as_tensor(cache["splits"]["val"]).max().item())
        )
        max_row = max(
            max_row, int(torch.as_tensor(cache["splits"]["test"]).max().item())
        )
        row_to_pos = torch.full((max_row + 1,), -1, dtype=torch.long)
        row_to_pos[train_idx] = torch.arange(train_idx.numel(), dtype=torch.long)
        self.row_to_pos = row_to_pos.to(device)

    @torch.no_grad()
    def retrieve_tokens(
        self, ctp: torch.Tensor, row_index: torch.Tensor, k: int
    ) -> torch.Tensor:
        query = F.normalize(ctp.float(), dim=1)
        sim = query @ self.train_x.T
        row_index = row_index.to(sim.device)
        in_range = row_index < self.row_to_pos.numel()
        pos = torch.full_like(row_index, -1)
        pos[in_range] = self.row_to_pos[row_index[in_range]]
        valid = pos.ge(0)
        if valid.any():
            sim[torch.arange(sim.size(0), device=sim.device)[valid], pos[valid]] = (
                -float("inf")
            )
        nn_idx = sim.topk(k, dim=1).indices
        return self.train_tokens[nn_idx]


class RetrievalAdaLNForwardGenerator(nn.Module):

    def __init__(
        self,
        y_dim: int,
        vocab_size: int,
        pad_id: int,
        n_cell: int,
        n_time: int,
        max_len: int = 128,
        d_model: int = 384,
        nhead: int = 8,
        num_layers: int = 8,
        dropout: float = 0.1,
        noise_dim: int = 64,
        cond_depth: int = 3,
        retrieval_k: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.max_len = max_len
        self.noise_dim = noise_dim
        self.retrieval_k = retrieval_k
        self.cond_encoder = StrongConditionEncoder(
            y_dim, n_cell, n_time, d_model, dropout, depth=cond_depth
        )
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.retr_rank_emb = nn.Parameter(torch.zeros(1, retrieval_k, d_model))
        self.retr_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.retr_norm = nn.LayerNorm(d_model)
        self.cond_fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.blocks = nn.ModuleList(
            [AdaLNBlock(d_model, nhead, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.retr_rank_emb, std=0.02)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        if pad_id is not None:
            with torch.no_grad():
                self.token_emb.weight[pad_id].zero_()

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def _pool_retrieved(self, retrieved_tokens: torch.Tensor) -> torch.Tensor:
        bsz, k, length = retrieved_tokens.shape
        flat = retrieved_tokens.reshape(bsz * k, length)
        emb = self.token_emb(flat.long())
        mask = flat.ne(self.pad_id).float()
        pooled = (emb * mask[:, :, None]).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        pooled = pooled.view(bsz, k, -1)
        return pooled + self.retr_rank_emb[:, :k, :]

    def forward(
        self,
        y: torch.Tensor,
        dose: torch.Tensor,
        cell: torch.Tensor,
        time: torch.Tensor,
        input_tokens: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        retrieved_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, length = input_tokens.shape
        if length > self.max_len:
            input_tokens = input_tokens[:, : self.max_len]
            length = self.max_len
        cond = self.cond_encoder(y, dose, cell, time)
        if z is None:
            z = torch.randn(bsz, self.noise_dim, device=y.device, dtype=y.dtype)
        h_z = self.noise_proj(z)
        if retrieved_tokens is None:
            h_retr = torch.zeros_like(cond)
        else:
            retr = self._pool_retrieved(retrieved_tokens.to(input_tokens.device))
            q = cond[:, None, :]
            h_retr, _ = self.retr_attn(q, retr, retr, need_weights=False)
            h_retr = self.retr_norm(h_retr.squeeze(1))
        dec_cond = self.cond_fuse(torch.cat([cond, h_retr, h_z], dim=-1))
        x = self.token_emb(input_tokens.long()) + self.pos_emb[:, :length, :]
        attn_mask = self._causal_mask(length, x.device)
        key_padding = input_tokens.eq(self.pad_id)
        for block in self.blocks:
            x = block(x, dec_cond, attn_mask=attn_mask, key_padding_mask=key_padding)
        return self.lm_head(self.norm(x))


class MaskGITForwardGenerator(nn.Module):

    def __init__(
        self,
        y_dim: int,
        vocab_size: int,
        pad_id: int,
        n_cell: int,
        n_time: int,
        max_len: int = 128,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dropout: float = 0.1,
        noise_dim: int = 64,
        cond_depth: int = 3,
        prefix_len: int = 32,
        mask_token_id: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.max_len = max_len
        self.noise_dim = noise_dim
        self.prefix_len = prefix_len
        self.mask_token_id = mask_token_id
        self.cond_encoder = StrongConditionEncoder(
            y_dim, n_cell, n_time, d_model, dropout, depth=cond_depth
        )
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.prefix_proj = nn.Linear(d_model * 2, prefix_len * d_model)
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len + prefix_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        nn.init.normal_(self.pos_emb, std=0.02)

    def _make_prefix(
        self,
        y: torch.Tensor,
        dose: torch.Tensor,
        cell: torch.Tensor,
        time: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = y.size(0)
        cond = self.cond_encoder(y, dose, cell, time)
        if z is None:
            z = torch.randn(bsz, self.noise_dim, device=y.device, dtype=y.dtype)
        h_z = self.noise_proj(z)
        prefix = self.prefix_proj(torch.cat([cond, h_z], dim=-1))
        return prefix.view(bsz, self.prefix_len, -1)

    def forward(
        self,
        y: torch.Tensor,
        dose: torch.Tensor,
        cell: torch.Tensor,
        time: torch.Tensor,
        input_tokens: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, length = input_tokens.shape
        if length > self.max_len:
            input_tokens = input_tokens[:, : self.max_len]
            length = self.max_len
        prefix = self._make_prefix(y, dose, cell, time, z=z)
        tok = self.token_emb(input_tokens.long())
        x = torch.cat([prefix, tok], dim=1)
        x = x + self.pos_emb[:, : x.size(1), :]
        prefix_pad = torch.zeros(
            bsz, self.prefix_len, dtype=torch.bool, device=input_tokens.device
        )
        key_padding = torch.cat([prefix_pad, input_tokens.eq(self.pad_id)], dim=1)
        out = self.transformer(x, src_key_padding_mask=key_padding)
        out = self.norm(out[:, self.prefix_len :, :])
        return self.lm_head(out)


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
        noise_dim=args.noise_dim,
    )
    if args.method == "prefix":
        return ForwardGenerator(prefix_len=args.prefix_len, **common)
    if args.method == "adaln":
        return AdaLNForwardGenerator(cond_depth=args.cond_depth, **common)
    if args.method == "cvae_adaln":
        return CVAEAdaLNForwardGenerator(cond_depth=args.cond_depth, **common)
    if args.method == "retrieval_adaln":
        return RetrievalAdaLNForwardGenerator(
            cond_depth=args.cond_depth, retrieval_k=args.retrieval_k, **common
        )
    if args.method == "maskgit":
        return MaskGITForwardGenerator(
            cond_depth=args.cond_depth,
            prefix_len=args.prefix_len,
            mask_token_id=(
                args.mask_token_id if args.mask_token_id >= 0 else cfg["unk_id"]
            ),
            **common,
        )
    raise ValueError(f"Unknown method: {args.method}")


def make_model_from_checkpoint(ckpt: dict) -> nn.Module:
    m = ckpt["model_args"]
    method = ckpt.get("method", "prefix")
    if method == "prefix":
        return ForwardGenerator(
            y_dim=m["y_dim"],
            vocab_size=m["vocab_size"],
            pad_id=m["pad_id"],
            n_cell=m["n_cell"],
            n_time=m["n_time"],
            max_len=m["max_len"],
            d_model=m["d_model"],
            nhead=m["nhead"],
            num_layers=m["num_layers"],
            dropout=m["dropout"],
            prefix_len=m.get("prefix_len", 8),
            noise_dim=m.get("noise_dim", 64),
        )
    if method == "adaln":
        return AdaLNForwardGenerator(
            y_dim=m["y_dim"],
            vocab_size=m["vocab_size"],
            pad_id=m["pad_id"],
            n_cell=m["n_cell"],
            n_time=m["n_time"],
            max_len=m["max_len"],
            d_model=m["d_model"],
            nhead=m["nhead"],
            num_layers=m["num_layers"],
            dropout=m["dropout"],
            noise_dim=m.get("noise_dim", 64),
            cond_depth=m.get("cond_depth", 3),
        )
    if method == "cvae_adaln":
        return CVAEAdaLNForwardGenerator(
            y_dim=m["y_dim"],
            vocab_size=m["vocab_size"],
            pad_id=m["pad_id"],
            n_cell=m["n_cell"],
            n_time=m["n_time"],
            max_len=m["max_len"],
            d_model=m["d_model"],
            nhead=m["nhead"],
            num_layers=m["num_layers"],
            dropout=m["dropout"],
            noise_dim=m.get("noise_dim", 64),
            cond_depth=m.get("cond_depth", 3),
        )
    if method == "retrieval_adaln":
        return RetrievalAdaLNForwardGenerator(
            y_dim=m["y_dim"],
            vocab_size=m["vocab_size"],
            pad_id=m["pad_id"],
            n_cell=m["n_cell"],
            n_time=m["n_time"],
            max_len=m["max_len"],
            d_model=m["d_model"],
            nhead=m["nhead"],
            num_layers=m["num_layers"],
            dropout=m["dropout"],
            noise_dim=m.get("noise_dim", 64),
            cond_depth=m.get("cond_depth", 3),
            retrieval_k=m.get("retrieval_k", 4),
        )
    if method == "maskgit":
        return MaskGITForwardGenerator(
            y_dim=m["y_dim"],
            vocab_size=m["vocab_size"],
            pad_id=m["pad_id"],
            n_cell=m["n_cell"],
            n_time=m["n_time"],
            max_len=m["max_len"],
            d_model=m["d_model"],
            nhead=m["nhead"],
            num_layers=m["num_layers"],
            dropout=m["dropout"],
            noise_dim=m.get("noise_dim", 64),
            cond_depth=m.get("cond_depth", 3),
            prefix_len=m.get("prefix_len", 32),
            mask_token_id=m.get("mask_token_id", 3),
        )
    raise ValueError(f"Unknown checkpoint method: {method}")


def masked_sequence_exact(pred: torch.Tensor, tgt: torch.Tensor, pad_id: int) -> int:
    mask = tgt.ne(pad_id)
    ok = (pred.eq(tgt) | ~mask).all(dim=1)
    return int(ok.sum().item())


def forward_model(
    model,
    ctp,
    dose,
    cell,
    time,
    inp,
    fixed_zero_noise: bool,
    posterior_tokens: Optional[torch.Tensor] = None,
    use_prior: bool = False,
    retrieved_tokens: Optional[torch.Tensor] = None,
):
    if fixed_zero_noise and hasattr(model, "noise_dim"):
        z = torch.zeros(
            ctp.size(0), int(model.noise_dim), device=ctp.device, dtype=ctp.dtype
        )
        if isinstance(model, RetrievalAdaLNForwardGenerator):
            return model(
                ctp, dose, cell, time, inp, z=z, retrieved_tokens=retrieved_tokens
            )
        return model(ctp, dose, cell, time, inp, z=z)
    if isinstance(model, CVAEAdaLNForwardGenerator):
        return model(
            ctp,
            dose,
            cell,
            time,
            inp,
            posterior_tokens=posterior_tokens,
            use_prior=use_prior,
        )
    if isinstance(model, RetrievalAdaLNForwardGenerator):
        return model(ctp, dose, cell, time, inp, retrieved_tokens=retrieved_tokens)
    return model(ctp, dose, cell, time, inp)


def run_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    train: bool,
    pad_id: int,
    label_smoothing: float,
    token_dropout: float,
    fixed_zero_noise: bool,
    beta_kl: float,
    free_bits: float,
    retriever: Optional[RetrievalIndex] = None,
    retrieval_k: int = 0,
):
    model.train(train)
    total_loss = total_acc = total_seq = n_tok = n_seq = 0
    pbar = tqdm(loader, desc="train G" if train else "eval G", leave=False)
    for batch in pbar:
        ctp = batch["ctp"].to(device)
        dose = batch["dose"].to(device)
        cell = batch["cell"].to(device)
        time = batch["time"].to(device)
        tokens = batch["tokens"].to(device)
        row_index = batch["row_index"].to(device)
        inp = tokens[:, :-1].clone()
        tgt = tokens[:, 1:]
        if train and token_dropout > 0:
            drop = (torch.rand_like(inp.float()) < token_dropout) & inp.ne(pad_id)
            drop[:, 0] = False
            inp = inp.masked_fill(drop, pad_id)
        if train:
            optimizer.zero_grad(set_to_none=True)
        retrieved_tokens = None
        if retriever is not None and retrieval_k > 0:
            retrieved_tokens = retriever.retrieve_tokens(ctp, row_index, retrieval_k)
        out = forward_model(
            model,
            ctp,
            dose,
            cell,
            time,
            inp,
            fixed_zero_noise,
            posterior_tokens=tokens if train else None,
            use_prior=not train,
            retrieved_tokens=retrieved_tokens,
        )
        kl = None
        if isinstance(out, tuple):
            logits, kl = out
        else:
            logits = out
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=pad_id,
            label_smoothing=label_smoothing,
        )
        ce_loss = loss
        if kl is not None and beta_kl > 0:
            kl_eff = torch.clamp(kl, min=free_bits)
            loss = loss + beta_kl * kl_eff
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        with torch.no_grad():
            mask = tgt.ne(pad_id)
            pred = logits.argmax(dim=-1)
            correct = (pred.eq(tgt) & mask).sum().item()
            count = mask.sum().item()
            seq_ok = masked_sequence_exact(pred, tgt, pad_id)
        total_loss += float(loss.item()) * max(count, 1)
        total_acc += correct
        total_seq += seq_ok
        n_tok += count
        n_seq += tgt.size(0)
        pbar.set_postfix(loss=total_loss / max(n_tok, 1), acc=total_acc / max(n_tok, 1))
    return {
        "loss": total_loss / max(n_tok, 1),
        "token_acc": total_acc / max(n_tok, 1),
        "seq_exact": total_seq / max(n_seq, 1),
    }


def make_maskgit_inputs(
    tokens: torch.Tensor,
    pad_id: int,
    bos_id: int,
    mask_token_id: int,
    mask_min: float,
    mask_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = tokens.size(0)
    can_mask = tokens.ne(pad_id) & tokens.ne(bos_id)
    ratios = torch.empty(bsz, 1, device=tokens.device).uniform_(mask_min, mask_max)
    mask = (torch.rand(tokens.shape, device=tokens.device) < ratios) & can_mask
    no_mask = mask.sum(dim=1).eq(0)
    if no_mask.any():
        rows = torch.arange(bsz, device=tokens.device)[no_mask]
        for row in rows.tolist():
            pos = torch.where(can_mask[row])[0]
            if pos.numel() > 0:
                pick = pos[torch.randint(pos.numel(), (1,), device=tokens.device)]
                mask[row, pick] = True
    corrupted = tokens.clone()
    corrupted = corrupted.masked_fill(mask, mask_token_id)
    return (corrupted, mask)


def run_epoch_maskgit(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    train: bool,
    cfg: dict,
    label_smoothing: float,
    fixed_zero_noise: bool,
    mask_min: float,
    mask_max: float,
    mask_token_id: int,
):
    model.train(train)
    total_loss = total_acc = total_seq = n_tok = n_seq = 0
    pbar = tqdm(loader, desc="train M" if train else "eval M", leave=False)
    for batch in pbar:
        ctp = batch["ctp"].to(device)
        dose = batch["dose"].to(device)
        cell = batch["cell"].to(device)
        time = batch["time"].to(device)
        tokens = batch["tokens"].to(device)
        inp, denoise_mask = make_maskgit_inputs(
            tokens, cfg["pad_id"], cfg["bos_id"], mask_token_id, mask_min, mask_max
        )
        if train:
            optimizer.zero_grad(set_to_none=True)
        z = None
        if fixed_zero_noise and hasattr(model, "noise_dim"):
            z = torch.zeros(
                ctp.size(0), int(model.noise_dim), device=ctp.device, dtype=ctp.dtype
            )
        logits = model(ctp, dose, cell, time, inp, z=z)
        targets = tokens.masked_fill(~denoise_mask, cfg["pad_id"])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=cfg["pad_id"],
            label_smoothing=label_smoothing,
        )
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct = (pred.eq(tokens) & denoise_mask).sum().item()
            count = denoise_mask.sum().item()
            seq_mask = tokens.ne(cfg["pad_id"])
            seq_ok = (
                (pred.eq(tokens) & denoise_mask | ~denoise_mask | ~seq_mask)
                .all(dim=1)
                .sum()
                .item()
            )
        total_loss += float(loss.item()) * max(count, 1)
        total_acc += correct
        total_seq += int(seq_ok)
        n_tok += count
        n_seq += tokens.size(0)
        pbar.set_postfix(loss=total_loss / max(n_tok, 1), acc=total_acc / max(n_tok, 1))
    return {
        "loss": total_loss / max(n_tok, 1),
        "token_acc": total_acc / max(n_tok, 1),
        "seq_exact": total_seq / max(n_seq, 1),
    }


def ids_to_smiles(ids, itos, bos_id: int, eos_id: int, pad_id: int) -> str:
    if sf is None:
        return ""
    toks = []
    for i in ids:
        i = int(i)
        if i in (bos_id, pad_id):
            continue
        if i == eos_id:
            break
        tok = itos[i]
        if tok.startswith("<") and tok.endswith(">"):
            continue
        toks.append(tok)
    try:
        return sf.decoder("".join(toks))
    except Exception:
        return ""


def canonicalize(smiles: str) -> tuple[str, bool]:
    if not smiles:
        return ("", False)
    if Chem is None:
        return (smiles, True)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (smiles, False)
    return (Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), True)


def mol_from_smiles(smiles: str):
    if Chem is None or not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def morgan_fp(mol):
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def maccs_fp(mol):
    if mol is None:
        return None
    return MACCSkeys.GenMACCSKeys(mol)


def scaffold_smiles(mol) -> str:
    if mol is None:
        return ""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None:
            return ""
        return Chem.MolToSmiles(scaf, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def mol_props(mol) -> Optional[np.ndarray]:
    if mol is None:
        return None
    vals = np.array(
        [
            Descriptors.MolWt(mol) / 100.0,
            Descriptors.MolLogP(mol) / 2.0,
            rdMolDescriptors.CalcTPSA(mol) / 100.0,
            Lipinski.NumHAcceptors(mol) / 5.0,
            Lipinski.NumHDonors(mol) / 3.0,
            QED.qed(mol),
            rdMolDescriptors.CalcNumRings(mol) / 3.0,
        ],
        dtype=np.float32,
    )
    return vals


def tanimoto(fp_a, fp_b) -> float:
    if fp_a is None or fp_b is None:
        return float("nan")
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


@torch.no_grad()
def generate_ids(
    model,
    ctp,
    dose,
    cell,
    time,
    cfg,
    temperature: float,
    top_k: int,
    top_p: float,
    fixed_zero_noise: bool,
    retrieved_tokens: Optional[torch.Tensor] = None,
):
    ids = torch.tensor([[cfg["bos_id"]]], dtype=torch.long, device=ctp.device)
    for _ in range(cfg["max_len"] - 1):
        logits = forward_model(
            model,
            ctp,
            dose,
            cell,
            time,
            ids,
            fixed_zero_noise,
            use_prior=True,
            retrieved_tokens=retrieved_tokens,
        )
        nxt_logits = logits[:, -1, :] / max(temperature, 1e-06)
        nxt_logits[:, cfg["pad_id"]] = -float("inf")
        nxt_logits[:, cfg["unk_id"]] = -float("inf")
        filt = top_k_top_p_filtering(nxt_logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(filt, dim=-1)
        nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)
        if int(nxt.item()) == cfg["eos_id"]:
            break
    return ids.squeeze(0).tolist()


@torch.no_grad()
def generate_ids_maskgit(
    model,
    ctp,
    dose,
    cell,
    time,
    cfg,
    temperature: float,
    top_k: int,
    top_p: float,
    fixed_zero_noise: bool,
    mask_steps: int,
    mask_token_id: int,
    gen_len: Optional[int] = None,
):
    device = ctp.device
    length = int(gen_len) if gen_len is not None else int(cfg["max_len"])
    length = max(2, min(int(cfg["max_len"]), length))
    ids = torch.full((1, length), mask_token_id, dtype=torch.long, device=device)
    ids[:, 0] = cfg["bos_id"]
    unknown = ids.eq(mask_token_id)
    unknown[:, 0] = False
    z = None
    if fixed_zero_noise and hasattr(model, "noise_dim"):
        z = torch.zeros(1, int(model.noise_dim), device=device, dtype=ctp.dtype)
    for step in range(mask_steps):
        logits = model(ctp, dose, cell, time, ids, z=z)
        logits = logits / max(temperature, 1e-06)
        logits[:, :, cfg["pad_id"]] = -float("inf")
        logits[:, :, cfg["unk_id"]] = -float("inf")
        logits[:, :, cfg["bos_id"]] = -float("inf")
        if top_k and top_k > 0:
            flat = logits.reshape(-1, logits.size(-1))
            flat = top_k_top_p_filtering(flat, top_k=top_k, top_p=top_p)
            logits = flat.view_as(logits)
        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view_as(ids)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        conf = conf.masked_fill(~unknown, float("inf"))
        n_unknown = int(unknown.sum().item())
        if n_unknown <= 0:
            break
        if step == mask_steps - 1:
            reveal = n_unknown
        else:
            remain_frac = math.cos(0.5 * math.pi * float(step + 1) / float(mask_steps))
            target_unknown = max(1, int(round((length - 1) * remain_frac)))
            reveal = max(1, n_unknown - target_unknown)
        reveal = min(reveal, n_unknown)
        reveal_idx = torch.topk(conf.view(-1), k=reveal, largest=True).indices
        flat_ids = ids.view(-1)
        flat_sampled = sampled.view(-1)
        flat_unknown = unknown.view(-1)
        flat_ids[reveal_idx] = flat_sampled[reveal_idx]
        flat_unknown[reveal_idx] = False
        ids = flat_ids.view_as(ids)
        unknown = flat_unknown.view_as(unknown)
    ids = ids.masked_fill(ids.eq(mask_token_id), cfg["eos_id"])
    out = ids.squeeze(0).tolist()
    if cfg["eos_id"] not in out:
        out[-1] = cfg["eos_id"]
    return out


@torch.no_grad()
def generation_eval(
    model,
    cache: dict,
    split: str,
    device,
    max_rows: int,
    n_per_row: int,
    temperature: float,
    top_k: int,
    top_p: float,
    fixed_zero_noise: bool,
    retriever: Optional[RetrievalIndex] = None,
    retrieval_k: int = 0,
):
    cfg = cache["config"]
    indices = list(map(int, cache["splits"][split]))[:max_rows]
    train_set = set()
    if Chem is not None:
        for i in cache["splits"]["train"]:
            can, valid = canonicalize(cache["smiles"][int(i)])
            if valid:
                train_set.add(can)
    valid = total = unique_total = novel_total = eos_total = 0
    lengths = []
    gt_max_morgan = []
    gt_mean_morgan = []
    gt_max_maccs = []
    gt_mean_maccs = []
    gt_hit_03 = gt_hit_05 = gt_hit_07 = 0
    gt_scaffold_hit = 0
    gt_prop_min_l1 = []
    gt_eval_rows = 0
    train_lengths = None
    if isinstance(model, MaskGITForwardGenerator):
        train_tokens = cache["tokens"][cache["splits"]["train"]]
        train_lengths = train_tokens.ne(cfg["pad_id"]).sum(dim=1).cpu().numpy()
    for idx in tqdm(indices, desc=f"gen eval {split}", leave=False):
        ctp = cache["X"][idx].to(device).view(1, -1)
        dose = cache["dose"][idx].to(device).view(1)
        cell = cache["cell"][idx].to(device).view(1)
        time = cache["time"][idx].to(device).view(1)
        row_index = torch.tensor([idx], dtype=torch.long, device=device)
        retrieved_tokens = None
        if retriever is not None and retrieval_k > 0:
            retrieved_tokens = retriever.retrieve_tokens(ctp, row_index, retrieval_k)
        gt_mol = (
            mol_from_smiles(cache["smiles"][idx])
            if Chem is not None and "smiles" in cache
            else None
        )
        gt_morgan_fp = morgan_fp(gt_mol)
        gt_maccs_fp = maccs_fp(gt_mol)
        gt_scaf = scaffold_smiles(gt_mol)
        gt_props = mol_props(gt_mol)
        row_morgan = []
        row_maccs = []
        row_scaffold = False
        row_prop_l1 = []
        seen = set()
        for _ in range(n_per_row):
            if isinstance(model, MaskGITForwardGenerator):
                gen_len = (
                    int(np.random.choice(train_lengths))
                    if train_lengths is not None and len(train_lengths)
                    else cfg["max_len"]
                )
                ids = generate_ids_maskgit(
                    model,
                    ctp,
                    dose,
                    cell,
                    time,
                    cfg,
                    temperature,
                    top_k,
                    top_p,
                    fixed_zero_noise,
                    mask_steps=getattr(model, "mask_steps", 12),
                    mask_token_id=getattr(model, "mask_token_id", cfg["unk_id"]),
                    gen_len=gen_len,
                )
            else:
                ids = generate_ids(
                    model,
                    ctp,
                    dose,
                    cell,
                    time,
                    cfg,
                    temperature,
                    top_k,
                    top_p,
                    fixed_zero_noise,
                    retrieved_tokens=retrieved_tokens,
                )
            total += 1
            eos_total += int(cfg["eos_id"] in ids)
            lengths.append(len(ids))
            smi = ids_to_smiles(
                ids, cache["vocab"]["itos"], cfg["bos_id"], cfg["eos_id"], cfg["pad_id"]
            )
            can, ok = canonicalize(smi)
            if ok:
                valid += 1
                gen_mol = mol_from_smiles(can)
                if gt_mol is not None and gen_mol is not None:
                    row_morgan.append(tanimoto(gt_morgan_fp, morgan_fp(gen_mol)))
                    row_maccs.append(tanimoto(gt_maccs_fp, maccs_fp(gen_mol)))
                    gen_scaf = scaffold_smiles(gen_mol)
                    if gt_scaf and gen_scaf and (gen_scaf == gt_scaf):
                        row_scaffold = True
                    gen_props = mol_props(gen_mol)
                    if gt_props is not None and gen_props is not None:
                        row_prop_l1.append(float(np.mean(np.abs(gen_props - gt_props))))
                if can not in seen:
                    unique_total += 1
                    seen.add(can)
                    if not train_set or can not in train_set:
                        novel_total += 1
        if row_morgan:
            mmax = float(np.nanmax(row_morgan))
            gt_max_morgan.append(mmax)
            gt_mean_morgan.append(float(np.nanmean(row_morgan)))
            gt_hit_03 += int(mmax >= 0.3)
            gt_hit_05 += int(mmax >= 0.5)
            gt_hit_07 += int(mmax >= 0.7)
            gt_eval_rows += 1
        if row_maccs:
            gt_max_maccs.append(float(np.nanmax(row_maccs)))
            gt_mean_maccs.append(float(np.nanmean(row_maccs)))
        gt_scaffold_hit += int(row_scaffold)
        if row_prop_l1:
            gt_prop_min_l1.append(float(np.min(row_prop_l1)))
    return {
        "split": split,
        "rows": len(indices),
        "n_per_row": n_per_row,
        "n_total": total,
        "validity": valid / max(total, 1),
        "unique_valid_per_sample": unique_total / max(total, 1),
        "novel_unique_per_sample": novel_total / max(total, 1),
        "eos_rate": eos_total / max(total, 1),
        "mean_length": float(np.mean(lengths)) if lengths else float("nan"),
        "gt_eval_rows": gt_eval_rows,
        "gt_max_morgan": (
            float(np.mean(gt_max_morgan)) if gt_max_morgan else float("nan")
        ),
        "gt_mean_morgan": (
            float(np.mean(gt_mean_morgan)) if gt_mean_morgan else float("nan")
        ),
        "gt_hit_morgan_0.3": gt_hit_03 / max(gt_eval_rows, 1),
        "gt_hit_morgan_0.5": gt_hit_05 / max(gt_eval_rows, 1),
        "gt_hit_morgan_0.7": gt_hit_07 / max(gt_eval_rows, 1),
        "gt_max_maccs": float(np.mean(gt_max_maccs)) if gt_max_maccs else float("nan"),
        "gt_mean_maccs": (
            float(np.mean(gt_mean_maccs)) if gt_mean_maccs else float("nan")
        ),
        "gt_scaffold_hit": gt_scaffold_hit / max(gt_eval_rows, 1),
        "gt_min_property_l1": (
            float(np.mean(gt_prop_min_l1)) if gt_prop_min_l1 else float("nan")
        ),
    }


def make_scheduler(
    optimizer, total_steps: int, warmup_frac: float, min_lr_ratio: float
):
    warmup = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    p = argparse.ArgumentParser(
        description="Train a gene-expression-conditioned molecular generator."
    )
    p.add_argument("--cache", default="data/processed/cache.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--metrics_jsonl", required=True)
    p.add_argument(
        "--method",
        choices=["prefix", "adaln", "cvae_adaln", "retrieval_adaln", "maskgit"],
        default="adaln",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--prefix_len", type=int, default=8)
    p.add_argument("--noise_dim", type=int, default=64)
    p.add_argument("--cond_depth", type=int, default=3)
    p.add_argument("--retrieval_k", type=int, default=4)
    p.add_argument("--mask_token_id", type=int, default=-1)
    p.add_argument("--mask_min", type=float, default=0.15)
    p.add_argument("--mask_max", type=float, default=0.95)
    p.add_argument("--mask_steps", type=int, default=12)
    p.add_argument("--label_smoothing", type=float, default=0.03)
    p.add_argument("--token_dropout", type=float, default=0.0)
    p.add_argument("--fixed_zero_noise", action="store_true")
    p.add_argument("--beta_kl", type=float, default=0.01)
    p.add_argument("--free_bits", type=float, default=0.0)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--early_stop_patience", type=int, default=3)
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)
    p.add_argument("--gen_eval_rows", type=int, default=128)
    p.add_argument("--gen_eval_n", type=int, default=4)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print("device:", device)
    cache = safe_torch_load(args.cache, map_location="cpu")
    cfg = cache["config"]
    if args.mask_token_id < 0:
        args.mask_token_id = cfg["unk_id"]
    retriever = None
    if args.method == "retrieval_adaln":
        retriever = RetrievalIndex(cache, device)
    if args.eval_only:
        ckpt = safe_torch_load(args.out, map_location=device)
        model = make_model_from_checkpoint(ckpt).to(device)
        model.load_state_dict(ckpt["model"])
        if isinstance(model, MaskGITForwardGenerator):
            model.mask_steps = int(args.mask_steps)
        model.eval()
        if ckpt.get("method") == "retrieval_adaln" and retriever is None:
            retriever = RetrievalIndex(cache, device)
        gen_val = generation_eval(
            model,
            cache,
            "val",
            device,
            args.gen_eval_rows,
            args.gen_eval_n,
            args.temperature,
            args.top_k,
            args.top_p,
            args.fixed_zero_noise,
            retriever=retriever,
            retrieval_k=args.retrieval_k,
        )
        eval_payload = {
            "best_epoch": ckpt.get("epoch", -1),
            "best_val_ce": ckpt.get("val", {}).get("loss", float("nan")),
            "val": ckpt.get("val", {}),
            "generation_val": gen_val,
        }
        eval_path = Path(args.out).with_suffix(".eval.json")
        eval_path.write_text(json.dumps(eval_payload, indent=2))
        print(json.dumps(eval_payload, indent=2))
        return
    train_loader = make_loader(
        cache, "train", args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = make_loader(
        cache, "val", args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    model = make_model(args, cfg).to(device)
    if isinstance(model, MaskGITForwardGenerator):
        model.mask_steps = int(args.mask_steps)
    n_params = sum((p.numel() for p in model.parameters()))
    print(f"method: {args.method} parameters: {n_params:,}")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(
        optimizer, args.epochs * len(train_loader), args.warmup_frac, args.min_lr_ratio
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_jsonl)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists():
        metrics_path.unlink()
    best = float("inf")
    best_epoch = -1
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        if args.method == "maskgit":
            tr = run_epoch_maskgit(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                True,
                cfg,
                args.label_smoothing,
                args.fixed_zero_noise,
                args.mask_min,
                args.mask_max,
                args.mask_token_id,
            )
            with torch.no_grad():
                va = run_epoch_maskgit(
                    model,
                    val_loader,
                    optimizer,
                    None,
                    device,
                    False,
                    cfg,
                    0.0,
                    args.fixed_zero_noise,
                    args.mask_min,
                    args.mask_max,
                    args.mask_token_id,
                )
        else:
            tr = run_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                True,
                cfg["pad_id"],
                args.label_smoothing,
                args.token_dropout,
                args.fixed_zero_noise,
                args.beta_kl,
                args.free_bits,
                retriever=retriever,
                retrieval_k=args.retrieval_k,
            )
            with torch.no_grad():
                va = run_epoch(
                    model,
                    val_loader,
                    optimizer,
                    None,
                    device,
                    False,
                    cfg["pad_id"],
                    0.0,
                    0.0,
                    args.fixed_zero_noise,
                    0.0,
                    args.free_bits,
                    retriever=retriever,
                    retrieval_k=args.retrieval_k,
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
            f"epoch {epoch:03d} | train CE {tr['loss']:.4f} acc {tr['token_acc']:.4f} seq {tr['seq_exact']:.4f} | val CE {va['loss']:.4f} acc {va['token_acc']:.4f} seq {va['seq_exact']:.4f}"
        )
        if va["loss"] < best - args.early_stop_min_delta:
            best = va["loss"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "method": args.method,
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
                        "dropout": args.dropout,
                        "prefix_len": args.prefix_len,
                        "noise_dim": args.noise_dim,
                        "cond_depth": args.cond_depth,
                        "retrieval_k": args.retrieval_k,
                        "mask_token_id": args.mask_token_id,
                    },
                    "cache_config": cfg,
                    "epoch": epoch,
                    "val": va,
                    "args": vars(args),
                },
                out,
            )
            print(f"  saved best forward-only G -> {out}")
        else:
            bad_epochs += 1
            if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
                print(
                    f"  early stopping after epoch {epoch}: best val CE {best:.6f} at epoch {best_epoch}, no improvement for {bad_epochs} epochs"
                )
                break
    ckpt = safe_torch_load(out, map_location=device)
    model.load_state_dict(ckpt["model"])
    if isinstance(model, MaskGITForwardGenerator):
        model.mask_steps = int(args.mask_steps)
    model.eval()
    gen_val = generation_eval(
        model,
        cache,
        "val",
        device,
        args.gen_eval_rows,
        args.gen_eval_n,
        args.temperature,
        args.top_k,
        args.top_p,
        args.fixed_zero_noise,
        retriever=retriever,
        retrieval_k=args.retrieval_k,
    )
    eval_payload = {
        "best_epoch": best_epoch,
        "best_val_ce": best,
        "val": ckpt["val"],
        "generation_val": gen_val,
    }
    eval_path = out.with_suffix(".eval.json")
    eval_path.write_text(json.dumps(eval_payload, indent=2))
    print(json.dumps(eval_payload, indent=2))


if __name__ == "__main__":
    main()
