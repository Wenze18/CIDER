from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionEncoder(nn.Module):

    def __init__(
        self,
        y_dim: int,
        n_cell: int,
        n_time: int,
        d_model: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.y_net = nn.Sequential(
            nn.Linear(y_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
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
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
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
        h_y = self.y_net(y)
        h_d = self.dose_net(dose.float())
        h_c = self.cell_emb(cell.long())
        h_t = self.time_emb(time.long())
        return self.fuse(torch.cat([h_y, h_d, h_c, h_t], dim=-1))


class ContextEncoderNoCTP(nn.Module):

    def __init__(
        self, n_cell: int, n_time: int, d_model: int = 256, dropout: float = 0.1
    ):
        super().__init__()
        self.dose_net = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.cell_emb = nn.Embedding(n_cell, d_model)
        self.time_emb = nn.Embedding(n_time, d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self, dose: torch.Tensor, cell: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        if dose.ndim == 1:
            dose = dose[:, None]
        h_d = self.dose_net(dose.float())
        h_c = self.cell_emb(cell.long())
        h_t = self.time_emb(time.long())
        return self.fuse(torch.cat([h_d, h_c, h_t], dim=-1))


class MolTransformerEncoder(nn.Module):

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
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
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
        nn.init.normal_(self.pos_emb, std=0.02)

    def _encode_emb(
        self, emb: torch.Tensor, pad_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        B, L, D = emb.shape
        if L > self.max_len:
            emb = emb[:, : self.max_len]
            if pad_mask is not None:
                pad_mask = pad_mask[:, : self.max_len]
            L = self.max_len
        emb = emb + self.pos_emb[:, :L, :]
        out = self.encoder(emb, src_key_padding_mask=pad_mask)
        out = self.norm(out)
        if pad_mask is None:
            mask = torch.ones(B, L, device=out.device, dtype=out.dtype)
        else:
            mask = (~pad_mask).float()
        pooled = (out * mask[:, :, None]).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        return pooled

    def forward_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        pad_mask = token_ids.eq(self.pad_id)
        emb = self.token_emb(token_ids.long())
        return self._encode_emb(emb, pad_mask)

    def forward_soft_probs(
        self, probs: torch.Tensor, pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        emb = probs @ self.token_emb.weight
        return self._encode_emb(emb, pad_mask)


class ReversePredictor(nn.Module):

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
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mol_encoder = MolTransformerEncoder(
            vocab_size=vocab_size,
            pad_id=pad_id,
            max_len=max_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.ctx_encoder = ContextEncoderNoCTP(
            n_cell=n_cell, n_time=n_time, d_model=d_model, dropout=dropout
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, y_dim),
        )

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        dose: torch.Tensor,
        cell: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        h_m = self.mol_encoder.forward_tokens(tokens)
        h_ctx = self.ctx_encoder(dose, cell, time)
        return self.head(torch.cat([h_m, h_ctx], dim=-1))

    def forward_soft_probs(
        self,
        probs: torch.Tensor,
        dose: torch.Tensor,
        cell: torch.Tensor,
        time: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h_m = self.mol_encoder.forward_soft_probs(probs, pad_mask=pad_mask)
        h_ctx = self.ctx_encoder(dose, cell, time)
        return self.head(torch.cat([h_m, h_ctx], dim=-1))


class ForwardGenerator(nn.Module):

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
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        prefix_len: int = 8,
        noise_dim: int = 64,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.max_len = max_len
        self.prefix_len = prefix_len
        self.noise_dim = noise_dim
        self.cond_encoder = ConditionEncoder(
            y_dim, n_cell, n_time, d_model=d_model, dropout=dropout
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
            dim_feedforward=dim_feedforward,
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
        B = y.size(0)
        h = self.cond_encoder(y, dose, cell, time)
        if z is None:
            z = torch.randn(B, self.noise_dim, device=y.device, dtype=y.dtype)
        h_z = self.noise_proj(z)
        prefix = self.prefix_proj(torch.cat([h, h_z], dim=-1))
        return prefix.view(B, self.prefix_len, -1)

    @staticmethod
    def _causal_prefix_mask(
        total_len: int, prefix_len: int, device: torch.device
    ) -> torch.Tensor:
        mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
        if total_len > prefix_len:
            mask[:prefix_len, prefix_len:] = True
            causal = torch.triu(
                torch.ones(
                    total_len - prefix_len,
                    total_len - prefix_len,
                    dtype=torch.bool,
                    device=device,
                ),
                diagonal=1,
            )
            mask[prefix_len:, prefix_len:] = causal
        return mask

    def forward(
        self,
        y: torch.Tensor,
        dose: torch.Tensor,
        cell: torch.Tensor,
        time: torch.Tensor,
        input_tokens: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = input_tokens.shape
        prefix = self._make_prefix(y, dose, cell, time, z=z)
        tok = self.token_emb(input_tokens.long())
        x = torch.cat([prefix, tok], dim=1)
        total_len = x.size(1)
        x = x + self.pos_emb[:, :total_len, :]
        mask = self._causal_prefix_mask(total_len, self.prefix_len, device=x.device)
        key_padding = None
        if self.pad_id is not None:
            pad = input_tokens.eq(self.pad_id)
            prefix_pad = torch.zeros(
                B, self.prefix_len, dtype=torch.bool, device=input_tokens.device
            )
            key_padding = torch.cat([prefix_pad, pad], dim=1)
        out = self.transformer(x, mask=mask, src_key_padding_mask=key_padding)
        out = self.norm(out[:, self.prefix_len :, :])
        logits = self.lm_head(out)
        return logits
