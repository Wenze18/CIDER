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
from cider.utils import (
    get_device,
    make_loader,
    safe_torch_load,
    set_seed,
    top_k_top_p_filtering,
)
from cider.training.forward_generator import (
    RetrievalIndex,
    StrongConditionEncoder,
    canonicalize,
    ids_to_smiles,
    maccs_fp,
    mol_from_smiles,
    mol_props,
    morgan_fp,
    scaffold_smiles,
    tanimoto,
)


class RetrievalCrossAttentionGenerator(nn.Module):

    def __init__(
        self,
        y_dim: int,
        vocab_size: int,
        pad_id: int,
        n_cell: int,
        n_time: int,
        max_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        noise_dim: int,
        cond_depth: int,
        retrieval_k: int,
        cond_tokens: int,
        retrieval_len: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.max_len = max_len
        self.noise_dim = noise_dim
        self.retrieval_k = retrieval_k
        self.cond_tokens = cond_tokens
        self.retrieval_len = retrieval_len
        self.cond_encoder = StrongConditionEncoder(
            y_dim, n_cell, n_time, d_model, dropout, depth=cond_depth
        )
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.cond_proj = nn.Linear(d_model * 2, cond_tokens * d_model)
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.dec_pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.retr_pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.rank_emb = nn.Parameter(torch.zeros(1, retrieval_k, 1, d_model))
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        nn.init.normal_(self.dec_pos, std=0.02)
        nn.init.normal_(self.retr_pos, std=0.02)
        nn.init.normal_(self.rank_emb, std=0.02)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        with torch.no_grad():
            self.token_emb.weight[pad_id].zero_()

    @staticmethod
    def causal_mask(length: int, device):
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def make_memory(self, ctp, dose, cell, time, retrieved_tokens, z=None):
        bsz = ctp.size(0)
        retrieved_tokens = retrieved_tokens[:, :, : self.retrieval_len]
        cond = self.cond_encoder(ctp, dose, cell, time)
        if z is None:
            z = torch.randn(bsz, self.noise_dim, device=ctp.device, dtype=ctp.dtype)
        cond_mem = self.cond_proj(torch.cat([cond, self.noise_proj(z)], dim=-1)).view(
            bsz, self.cond_tokens, -1
        )
        k = retrieved_tokens.size(1)
        retr = self.token_emb(retrieved_tokens.long())
        retr = (
            retr
            + self.rank_emb[:, :k]
            + self.retr_pos[:, : retrieved_tokens.size(2)].unsqueeze(1)
        )
        retr = retr.reshape(bsz, k * retrieved_tokens.size(2), -1)
        memory = torch.cat([cond_mem, retr], dim=1)
        cond_pad = torch.zeros(
            bsz, self.cond_tokens, dtype=torch.bool, device=ctp.device
        )
        retr_pad = retrieved_tokens.eq(self.pad_id).reshape(
            bsz, k * retrieved_tokens.size(2)
        )
        memory_pad = torch.cat([cond_pad, retr_pad], dim=1)
        return (memory, memory_pad)

    def forward(self, ctp, dose, cell, time, input_tokens, retrieved_tokens, z=None):
        length = input_tokens.size(1)
        memory, memory_pad = self.make_memory(
            ctp, dose, cell, time, retrieved_tokens, z=z
        )
        tgt = self.token_emb(input_tokens.long()) + self.dec_pos[:, :length]
        out = self.decoder(
            tgt,
            memory,
            tgt_mask=self.causal_mask(length, input_tokens.device),
            tgt_key_padding_mask=input_tokens.eq(self.pad_id),
            memory_key_padding_mask=memory_pad,
        )
        return self.lm_head(self.norm(out))


def masked_sequence_exact(pred: torch.Tensor, tgt: torch.Tensor, pad_id: int) -> int:
    mask = tgt.ne(pad_id)
    return int((pred.eq(tgt) | ~mask).all(dim=1).sum().item())


def run_epoch(
    model,
    loader,
    retriever,
    optimizer,
    device,
    pad_id: int,
    retrieval_k: int,
    train: bool,
    label_smoothing: float,
):
    model.train(train)
    total_loss = total_acc = total_seq = n_tok = n_seq = 0
    pbar = tqdm(loader, desc="train xattn" if train else "eval xattn", leave=False)
    for batch in pbar:
        ctp = batch["ctp"].to(device)
        dose = batch["dose"].to(device)
        cell = batch["cell"].to(device)
        time = batch["time"].to(device)
        tokens = batch["tokens"].to(device)
        row_index = batch["row_index"].to(device)
        retrieved = retriever.retrieve_tokens(ctp, row_index, retrieval_k)
        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(ctp, dose, cell, time, inp, retrieved)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=pad_id,
            label_smoothing=label_smoothing,
        )
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            mask = tgt.ne(pad_id)
            pred = logits.argmax(dim=-1)
            total_acc += int((pred.eq(tgt) & mask).sum().item())
            count = int(mask.sum().item())
            total_seq += masked_sequence_exact(pred, tgt, pad_id)
        total_loss += float(loss.item()) * max(count, 1)
        n_tok += count
        n_seq += tgt.size(0)
    return {
        "loss": total_loss / max(n_tok, 1),
        "token_acc": total_acc / max(n_tok, 1),
        "seq_exact": total_seq / max(n_seq, 1),
    }


@torch.no_grad()
def generate_ids(
    model, ctp, dose, cell, time, retrieved, cfg, temperature, top_k, top_p
):
    ids = torch.tensor([[cfg["bos_id"]]], dtype=torch.long, device=ctp.device)
    for _ in range(cfg["max_len"] - 1):
        logits = model(ctp, dose, cell, time, ids, retrieved)
        nxt_logits = logits[:, -1, :] / max(temperature, 1e-06)
        nxt_logits[:, cfg["pad_id"]] = -float("inf")
        nxt_logits[:, cfg["unk_id"]] = -float("inf")
        filt = top_k_top_p_filtering(nxt_logits, top_k=top_k, top_p=top_p)
        nxt = torch.multinomial(torch.softmax(filt, dim=-1), 1)
        ids = torch.cat([ids, nxt], dim=1)
        if int(nxt.item()) == cfg["eos_id"]:
            break
    return ids.squeeze(0).tolist()


@torch.no_grad()
def generation_eval(
    model,
    cache,
    retriever,
    split,
    device,
    rows,
    n_per_row,
    retrieval_k,
    temperature,
    top_k,
    top_p,
):
    cfg = cache["config"]
    indices = list(map(int, cache["splits"][split]))[:rows]
    train_set = set()
    for i in cache["splits"]["train"]:
        can, ok = canonicalize(cache["smiles"][int(i)])
        if ok:
            train_set.add(can)
    valid = total = unique_total = novel_total = eos_total = 0
    gt_max_morgan = []
    gt_mean_morgan = []
    gt_max_maccs = []
    gt_mean_maccs = []
    hit03 = hit05 = hit07 = scaffold_hit = 0
    prop_l1 = []
    lengths = []
    for idx in tqdm(indices, desc=f"gen eval {split}", leave=False):
        ctp = cache["X"][idx].to(device).view(1, -1)
        dose = cache["dose"][idx].to(device).view(1)
        cell = cache["cell"][idx].to(device).view(1)
        time = cache["time"][idx].to(device).view(1)
        row_index = torch.tensor([idx], dtype=torch.long, device=device)
        retrieved = retriever.retrieve_tokens(ctp, row_index, retrieval_k)
        gt_mol = mol_from_smiles(cache["smiles"][idx])
        gt_morgan = morgan_fp(gt_mol)
        gt_maccs = maccs_fp(gt_mol)
        gt_scaf = scaffold_smiles(gt_mol)
        gt_props = mol_props(gt_mol)
        seen = set()
        row_morgan = []
        row_maccs = []
        row_prop = []
        row_scaf = False
        for _ in range(n_per_row):
            ids = generate_ids(
                model, ctp, dose, cell, time, retrieved, cfg, temperature, top_k, top_p
            )
            total += 1
            eos_total += int(cfg["eos_id"] in ids)
            lengths.append(len(ids))
            smi = ids_to_smiles(
                ids, cache["vocab"]["itos"], cfg["bos_id"], cfg["eos_id"], cfg["pad_id"]
            )
            can, ok = canonicalize(smi)
            if not ok:
                continue
            valid += 1
            if can not in seen:
                unique_total += 1
                novel_total += int(can not in train_set)
                seen.add(can)
            mol = mol_from_smiles(can)
            row_morgan.append(tanimoto(gt_morgan, morgan_fp(mol)))
            row_maccs.append(tanimoto(gt_maccs, maccs_fp(mol)))
            gen_scaf = scaffold_smiles(mol)
            row_scaf = row_scaf or bool(gt_scaf and gen_scaf and (gt_scaf == gen_scaf))
            props = mol_props(mol)
            if gt_props is not None and props is not None:
                row_prop.append(float(np.mean(np.abs(gt_props - props))))
        if row_morgan:
            mmax = float(np.nanmax(row_morgan))
            gt_max_morgan.append(mmax)
            gt_mean_morgan.append(float(np.nanmean(row_morgan)))
            hit03 += int(mmax >= 0.3)
            hit05 += int(mmax >= 0.5)
            hit07 += int(mmax >= 0.7)
        if row_maccs:
            gt_max_maccs.append(float(np.nanmax(row_maccs)))
            gt_mean_maccs.append(float(np.nanmean(row_maccs)))
        scaffold_hit += int(row_scaf)
        if row_prop:
            prop_l1.append(float(np.min(row_prop)))
    eval_rows = len(gt_max_morgan)
    return {
        "split": split,
        "rows": len(indices),
        "n_per_row": n_per_row,
        "validity": valid / max(total, 1),
        "unique_valid_per_sample": unique_total / max(total, 1),
        "novel_unique_per_sample": novel_total / max(total, 1),
        "eos_rate": eos_total / max(total, 1),
        "mean_length": float(np.mean(lengths)) if lengths else float("nan"),
        "gt_eval_rows": eval_rows,
        "gt_max_morgan": (
            float(np.mean(gt_max_morgan)) if gt_max_morgan else float("nan")
        ),
        "gt_mean_morgan": (
            float(np.mean(gt_mean_morgan)) if gt_mean_morgan else float("nan")
        ),
        "gt_hit_morgan_0.3": hit03 / max(eval_rows, 1),
        "gt_hit_morgan_0.5": hit05 / max(eval_rows, 1),
        "gt_hit_morgan_0.7": hit07 / max(eval_rows, 1),
        "gt_max_maccs": float(np.mean(gt_max_maccs)) if gt_max_maccs else float("nan"),
        "gt_mean_maccs": (
            float(np.mean(gt_mean_maccs)) if gt_mean_maccs else float("nan")
        ),
        "gt_scaffold_hit": scaffold_hit / max(eval_rows, 1),
        "gt_min_property_l1": float(np.mean(prop_l1)) if prop_l1 else float("nan"),
    }


def main():
    p = argparse.ArgumentParser(
        description="Forward retrieval cross-attention decoder."
    )
    p.add_argument("--cache", default="data/processed/cache.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--metrics_jsonl", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--noise_dim", type=int, default=64)
    p.add_argument("--cond_depth", type=int, default=3)
    p.add_argument("--retrieval_k", type=int, default=4)
    p.add_argument("--cond_tokens", type=int, default=8)
    p.add_argument("--retrieval_len", type=int, default=64)
    p.add_argument("--label_smoothing", type=float, default=0.03)
    p.add_argument("--early_stop_patience", type=int, default=3)
    p.add_argument("--gen_eval_rows", type=int, default=128)
    p.add_argument("--gen_eval_n", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    cache = safe_torch_load(args.cache, map_location="cpu")
    cfg = cache["config"]
    retriever = RetrievalIndex(cache, device)
    model = RetrievalCrossAttentionGenerator(
        cfg["y_dim"],
        cfg["vocab_size"],
        cfg["pad_id"],
        cfg["n_cell"],
        cfg["n_time"],
        cfg["max_len"],
        args.d_model,
        args.heads,
        args.layers,
        args.dropout,
        args.noise_dim,
        args.cond_depth,
        args.retrieval_k,
        args.cond_tokens,
        args.retrieval_len,
    ).to(device)
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
    model_args = {
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
        "noise_dim": args.noise_dim,
        "cond_depth": args.cond_depth,
        "retrieval_k": args.retrieval_k,
        "cond_tokens": args.cond_tokens,
        "retrieval_len": args.retrieval_len,
    }
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            model,
            train_loader,
            retriever,
            optimizer,
            device,
            cfg["pad_id"],
            args.retrieval_k,
            True,
            args.label_smoothing,
        )
        va = run_epoch(
            model,
            val_loader,
            retriever,
            optimizer,
            device,
            cfg["pad_id"],
            args.retrieval_k,
            False,
            args.label_smoothing,
        )
        row = {"epoch": epoch, "train": tr, "val": va, "args": vars(args)}
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch:03d} | train CE {tr['loss']:.4f} acc {tr['token_acc']:.4f} | val CE {va['loss']:.4f} acc {va['token_acc']:.4f}"
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
                    "model_args": model_args,
                    "method": "retrieval_xattn",
                },
                out,
            )
            print(f"  saved best retrieval xattn model -> {out}")
        else:
            bad += 1
            if bad >= args.early_stop_patience:
                print(
                    f"  early stopping after epoch {epoch}: best val CE {best:.6f} at epoch {best_epoch}"
                )
                break
    ckpt = safe_torch_load(out, map_location=device)
    model.load_state_dict(ckpt["model"])
    ev = generation_eval(
        model,
        cache,
        retriever,
        "val",
        device,
        args.gen_eval_rows,
        args.gen_eval_n,
        args.retrieval_k,
        args.temperature,
        args.top_k,
        args.top_p,
    )
    payload = {"best_epoch": best_epoch, "best_val": ckpt["val"], "generation_val": ev}
    out.with_suffix(".eval.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
