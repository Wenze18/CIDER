from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from cider.utils import get_device, make_loader, safe_torch_load, set_seed
from cider.training.fingerprint import load_or_build_fps
from cider.training.forward_generator import (
    RetrievalIndex,
    forward_model,
    generation_eval,
    make_model_from_checkpoint,
    masked_sequence_exact,
)


class StructureAwarePositiveSampler:

    def __init__(
        self,
        cache: dict,
        fps: torch.Tensor,
        device: torch.device,
        ctp_k: int,
        min_sim: float,
        own_prob: float,
        sim_temp: float,
        mix_ctp_weight: float,
    ):
        self.device = device
        self.ctp_k = ctp_k
        self.min_sim = min_sim
        self.own_prob = own_prob
        self.sim_temp = sim_temp
        self.mix_ctp_weight = mix_ctp_weight
        self.train_idx = torch.as_tensor(
            cache["splits"]["train"], dtype=torch.long, device=device
        )
        train_x = cache["X"][self.train_idx.cpu()].float().to(device)
        self.train_x = F.normalize(train_x, dim=1)
        self.tokens = cache["tokens"]
        self.fps = fps.float().to(device)
        self.fp_sum = self.fps.sum(dim=1)

    def _fp_tanimoto(
        self, row_index: torch.Tensor, candidate_rows: torch.Tensor
    ) -> torch.Tensor:
        query_fp = self.fps[row_index].float()
        cand_fp = self.fps[candidate_rows].float()
        inter = (cand_fp * query_fp[:, None, :]).sum(dim=-1)
        denom = self.fp_sum[candidate_rows] + self.fp_sum[row_index][:, None] - inter
        return inter / denom.clamp_min(1e-06)

    @torch.no_grad()
    def sample(self, ctp: torch.Tensor, row_index: torch.Tensor):
        q = F.normalize(ctp.float(), dim=1)
        ctp_scores = q @ self.train_x.T
        same = row_index[:, None].eq(self.train_idx[None, :])
        ctp_scores = ctp_scores.masked_fill(same, -float("inf"))
        vals, pos = ctp_scores.topk(self.ctp_k, dim=1)
        candidate_rows = self.train_idx[pos]
        sim = self._fp_tanimoto(row_index, candidate_rows)
        ok = sim.ge(self.min_sim)
        if self.mix_ctp_weight != 0:
            logits = sim / max(self.sim_temp, 1e-06) + self.mix_ctp_weight * vals
        else:
            logits = sim / max(self.sim_temp, 1e-06)
        logits = logits.masked_fill(~ok, -float("inf"))
        no_pos = ~torch.isfinite(logits).any(dim=1)
        logits[no_pos] = sim[no_pos] / max(self.sim_temp, 1e-06)
        use_own = torch.rand(row_index.size(0), device=self.device).lt(self.own_prob)
        rel = torch.multinomial(torch.softmax(logits, dim=1), 1).squeeze(1)
        picked_rows = candidate_rows.gather(1, rel[:, None]).squeeze(1)
        picked_rows = torch.where(use_own, row_index, picked_rows)
        picked_sim = torch.where(
            use_own,
            torch.ones_like(row_index, dtype=torch.float32),
            sim.gather(1, rel[:, None]).squeeze(1),
        )
        return (self.tokens[picked_rows.detach().cpu()].to(self.device), picked_sim)


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    train: bool,
    cfg: dict,
    sampler: StructureAwarePositiveSampler | None,
    label_smoothing: float,
    token_dropout: float,
    fixed_zero_noise: bool,
    retriever: RetrievalIndex | None,
    retrieval_k: int,
):
    model.train(train)
    total_loss = total_acc = total_seq = n_tok = n_seq = n_rows = sim_sum = 0
    pbar = tqdm(
        loader, desc="train multipositive" if train else "eval original CE", leave=False
    )
    for batch in pbar:
        ctp = batch["ctp"].to(device)
        dose = batch["dose"].to(device)
        cell = batch["cell"].to(device)
        time = batch["time"].to(device)
        row_index = batch["row_index"].to(device)
        tokens = batch["tokens"].to(device)
        sampled_sim = torch.ones(tokens.size(0), device=device)
        if train and sampler is not None:
            tokens, sampled_sim = sampler.sample(ctp, row_index)
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
        n_rows += tokens.size(0)
        sim_sum += float(sampled_sim.sum().item())
        pbar.set_postfix(
            loss=total_loss / max(n_tok, 1),
            acc=total_acc / max(n_tok, 1),
            sim=sim_sum / max(n_rows, 1),
        )
    return {
        "loss": total_loss / max(n_tok, 1),
        "token_acc": total_acc / max(n_tok, 1),
        "seq_exact": total_seq / max(n_seq, 1),
        "sampled_target_sim": sim_sum / max(n_rows, 1),
    }


def save_checkpoint(path: Path, model, ckpt: dict, epoch: int, va: dict, args):
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "val": va,
            "args": vars(args),
            "base_ckpt": args.base_ckpt,
            "method": ckpt.get("method", "prefix"),
            "model_args": ckpt["model_args"],
            "training_objective": "structure_aware_multipositive_likelihood",
        },
        path,
    )


def main():
    p = argparse.ArgumentParser(
        description="Structure-aware multi-positive fine-tuning for forward proposal models."
    )
    p.add_argument("--cache", default="data/processed/cache.pt")
    p.add_argument("--fp_cache", default="data/features/fingerprints_2048.pt")
    p.add_argument("--base_ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--metrics_jsonl", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-05)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--ctp_k", type=int, default=64)
    p.add_argument("--min_sim", type=float, default=0.2)
    p.add_argument("--own_prob", type=float, default=0.5)
    p.add_argument("--sim_temp", type=float, default=0.08)
    p.add_argument("--mix_ctp_weight", type=float, default=0.0)
    p.add_argument("--fp_dim", type=int, default=2048)
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
    fps, _, _ = load_or_build_fps(cache, args.fp_dim, args.fp_cache, fp_workers=8)
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
    last_out = out.with_name(out.stem + "_last.pt")
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
    sampler = StructureAwarePositiveSampler(
        cache,
        fps,
        device,
        args.ctp_k,
        args.min_sim,
        args.own_prob,
        args.sim_temp,
        args.mix_ctp_weight,
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
            sampler,
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
                args.fixed_zero_noise,
                retriever,
                retrieval_k,
            )
        row = {"epoch": epoch, "train": tr, "val": va, "args": vars(args)}
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch:03d} | train CE {tr['loss']:.4f} sim {tr['sampled_target_sim']:.4f} | val CE {va['loss']:.4f} acc {va['token_acc']:.4f}"
        )
        save_checkpoint(last_out, model, ckpt, epoch, va, args)
        if va["loss"] < best:
            best = va["loss"]
            best_epoch = epoch
            bad = 0
            save_checkpoint(out, model, ckpt, epoch, va, args)
            print(f"  saved best multipositive model -> {out}")
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
        "last_ckpt": str(last_out),
    }
    out.with_suffix(".eval.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
