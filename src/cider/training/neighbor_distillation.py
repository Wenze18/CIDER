from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from cider.utils import get_device, make_loader, safe_torch_load, set_seed
from cider.training.forward_generator import (
    RetrievalIndex,
    forward_model,
    generation_eval,
    make_model_from_checkpoint,
    masked_sequence_exact,
)


class CTPNeighborSampler:

    def __init__(self, cache: dict, device, k: int, sample_temp: float):
        self.device = device
        self.k = k
        self.sample_temp = sample_temp
        self.train_idx = torch.as_tensor(
            cache["splits"]["train"], dtype=torch.long, device=device
        )
        train_x = cache["X"][self.train_idx.cpu()].float().to(device)
        self.train_x = F.normalize(train_x, dim=1)
        self.tokens = cache["tokens"]

    @torch.no_grad()
    def sample(self, ctp: torch.Tensor, row_index: torch.Tensor):
        q = F.normalize(ctp.float(), dim=1)
        scores = q @ self.train_x.T
        same = row_index[:, None].eq(self.train_idx[None, :])
        scores = scores.masked_fill(same, -float("inf"))
        vals, pos = scores.topk(self.k, dim=1)
        if self.k == 1:
            pick = pos[:, 0]
        else:
            probs = torch.softmax(vals / max(self.sample_temp, 1e-06), dim=1)
            rel = torch.multinomial(probs, 1).squeeze(1)
            pick = pos.gather(1, rel[:, None]).squeeze(1)
        idx = self.train_idx[pick].detach().cpu()
        return self.tokens[idx].to(self.device)


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    train: bool,
    cfg: dict,
    neighbor_sampler: CTPNeighborSampler | None,
    neighbor_prob: float,
    label_smoothing: float,
    token_dropout: float,
    fixed_zero_noise: bool,
    retriever: RetrievalIndex | None,
    retrieval_k: int,
):
    model.train(train)
    total_loss = total_acc = total_seq = n_tok = n_seq = 0
    pbar = tqdm(
        loader,
        desc="train neighbor distill" if train else "eval original CE",
        leave=False,
    )
    for batch in pbar:
        ctp = batch["ctp"].to(device)
        dose = batch["dose"].to(device)
        cell = batch["cell"].to(device)
        time = batch["time"].to(device)
        row_index = batch["row_index"].to(device)
        tokens = batch["tokens"].to(device)
        if train and neighbor_sampler is not None and (neighbor_prob > 0):
            neighbor_tokens = neighbor_sampler.sample(ctp, row_index)
            use_neighbor = torch.rand(tokens.size(0), device=device) < neighbor_prob
            tokens = torch.where(use_neighbor[:, None], neighbor_tokens, tokens)
        inp = tokens[:, :-1].clone()
        tgt = tokens[:, 1:]
        if train and token_dropout > 0:
            drop = (torch.rand_like(inp.float()) < token_dropout) & inp.ne(
                cfg["pad_id"]
            )
            drop[:, 0] = False
            inp = inp.masked_fill(drop, cfg["pad_id"])
        if train:
            optimizer.zero_grad(set_to_none=True)
        retrieved_tokens = (
            retriever.retrieve_tokens(ctp, row_index, retrieval_k)
            if retriever is not None and retrieval_k > 0
            else None
        )
        logits = forward_model(
            model,
            ctp,
            dose,
            cell,
            time,
            inp,
            fixed_zero_noise,
            retrieved_tokens=retrieved_tokens,
        )
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=cfg["pad_id"],
            label_smoothing=label_smoothing,
        )
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            mask = tgt.ne(cfg["pad_id"])
            pred = logits.argmax(dim=-1)
            correct = (pred.eq(tgt) & mask).sum().item()
            count = mask.sum().item()
            seq_ok = masked_sequence_exact(pred, tgt, cfg["pad_id"])
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


def main():
    p = argparse.ArgumentParser(
        description="Reward/ranking-style neighbor distillation for forward generation."
    )
    p.add_argument("--cache", default="data/processed/cache.pt")
    p.add_argument("--base_ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--metrics_jsonl", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-05)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--neighbor_k", type=int, default=8)
    p.add_argument("--neighbor_prob", type=float, default=0.5)
    p.add_argument("--neighbor_temp", type=float, default=0.07)
    p.add_argument("--label_smoothing", type=float, default=0.02)
    p.add_argument("--token_dropout", type=float, default=0.0)
    p.add_argument("--fixed_zero_noise", action="store_true")
    p.add_argument("--gen_eval_rows", type=int, default=256)
    p.add_argument("--gen_eval_n", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--early_stop_patience", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    cache = safe_torch_load(args.cache, map_location="cpu")
    cfg = cache["config"]
    ckpt = safe_torch_load(args.base_ckpt, map_location=device)
    model = make_model_from_checkpoint(ckpt).to(device)
    model.load_state_dict(ckpt["model"])
    retriever = (
        RetrievalIndex(cache, device)
        if ckpt.get("method") == "retrieval_adaln"
        else None
    )
    retrieval_k = (
        int(ckpt.get("model_args", {}).get("retrieval_k", 0))
        if retriever is not None
        else 0
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_jsonl)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists():
        metrics_path.unlink()
    train_loader = make_loader(
        cache, "train", args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = make_loader(
        cache, "val", args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    neighbor_sampler = CTPNeighborSampler(
        cache, device, args.neighbor_k, args.neighbor_temp
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    best_epoch = -1
    bad = 0
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            True,
            cfg,
            neighbor_sampler,
            args.neighbor_prob,
            args.label_smoothing,
            args.token_dropout,
            args.fixed_zero_noise,
            retriever,
            retrieval_k,
        )
        with torch.no_grad():
            va = run_epoch(
                model,
                val_loader,
                None,
                device,
                False,
                cfg,
                None,
                0.0,
                0.0,
                0.0,
                args.fixed_zero_noise,
                retriever,
                retrieval_k,
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
                    "base_ckpt": args.base_ckpt,
                    "method": ckpt.get("method", "prefix"),
                    "model_args": ckpt["model_args"],
                },
                out,
            )
            print(f"  saved best neighbor-distilled model -> {out}")
        else:
            bad += 1
            if bad >= args.early_stop_patience:
                print(
                    f"  early stopping after epoch {epoch}: best val CE {best:.6f} at epoch {best_epoch}"
                )
                break
    best_ckpt = safe_torch_load(out, map_location=device)
    model.load_state_dict(best_ckpt["model"])
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
        retrieval_k=retrieval_k,
    )
    payload = {
        "best_epoch": best_epoch,
        "best_val": best_ckpt["val"],
        "generation_val": gen_val,
    }
    out.with_suffix(".eval.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
