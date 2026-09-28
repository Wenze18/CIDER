from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys
from torch.optim import AdamW
from tqdm import tqdm
from cider.utils import get_device, safe_torch_load, set_seed
from cider.neighbors import ctp_nn_fps, hard_tanimoto_to_matrix, load_diff_model
from cider.sampling import (
    generate_ids_batch,
    load_fp_model,
    load_generator,
    soft_tanimoto,
)
from cider.training.fingerprint_diffusion import sample_fp as sample_diff_fp
from cider.training.fingerprint import load_or_build_fps, morgan_bits
from cider.training.forward_generator import (
    canonicalize,
    ids_to_smiles,
    mol_from_smiles,
    mol_props,
    scaffold_smiles,
    tanimoto,
)

RDLogger.DisableLog("rdApp.*")


def morgan_fp(mol):
    return (
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        if mol is not None
        else None
    )


def maccs_fp(mol):
    return MACCSkeys.GenMACCSKeys(mol) if mol is not None else None


class StructuralReranker(nn.Module):

    def __init__(self, in_dim: int, hidden: int, depth: int, dropout: float):
        super().__init__()
        layers = [
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(max(depth - 1, 0)):
            layers += [
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_set_canonical(cache):
    out = set()
    for i in cache["splits"]["train"]:
        can, ok = canonicalize(cache["smiles"][int(i)])
        if ok:
            out.add(can)
    return out


def parse_repeat_seeds(text: str, default_seed: int):
    vals = [int(s) for s in text.split(",") if s.strip()] if text else []
    return vals or [default_seed]


def make_feature(
    args,
    cache,
    cfg,
    ctp_cpu,
    dose,
    cell,
    time,
    pred_fp,
    diff_fp,
    nn_mat,
    bits,
    fp_score,
    diff_score,
    nn_score,
    gen_rank_score,
    seed_rank_score,
    is_novel,
    ids_len,
    props,
):
    parts = []
    cand = torch.as_tensor(bits, dtype=torch.float32)
    if args.use_bits:
        parts.append(cand)
    if args.use_pred_fp:
        pred = pred_fp.detach().float().cpu()
        parts += [pred, cand * pred, (cand - pred).abs()]
    if args.use_nn_mean:
        parts.append(nn_mat.detach().float().cpu().mean(dim=0))
    if args.use_ctp:
        parts.append(ctp_cpu.detach().float().cpu())
    prop_feats = props.tolist() if props is not None else [0.0] * 7
    scalar = torch.tensor(
        [
            fp_score,
            diff_score,
            nn_score,
            gen_rank_score,
            seed_rank_score,
            float(is_novel),
            ids_len / max(float(cfg["max_len"]), 1.0),
            float(dose),
            float(cell) / max(float(cache["config"]["n_cell"] - 1), 1.0),
            float(time) / max(float(cache["config"]["n_time"] - 1), 1.0),
        ]
        + prop_feats,
        dtype=torch.float32,
    )
    parts.append(scalar)
    return torch.cat(parts)


@torch.no_grad()
def make_row_candidates(
    cache,
    cfg,
    idx: int,
    generators,
    fp_model,
    diff_model,
    diff_dim,
    diff_target,
    fps,
    train_set,
    args,
    device,
    repeat_seeds,
):
    ctp = cache["X"][idx].to(device).view(1, -1)
    ctp_cpu = cache["X"][idx].float()
    dose = cache["dose"][idx].to(device).view(1)
    cell = cache["cell"][idx].to(device).view(1)
    time = cache["time"][idx].to(device).view(1)
    row_index = torch.tensor([idx], dtype=torch.long, device=device)
    pred_fp = fp_model(ctp, dose, cell, time).sigmoid().squeeze(0)
    diff_fp = None
    if diff_model is not None:
        diff_fp = sample_diff_fp(
            diff_model,
            ctp,
            dose,
            cell,
            time,
            diff_dim,
            args.diff_steps,
            args.diff_samples,
            diff_target,
        ).mean(dim=0)
    nn_mat = ctp_nn_fps(cache, fps, idx, device, args.nn_k)
    gt_mol = mol_from_smiles(cache["smiles"][idx])
    gt_morgan = morgan_fp(gt_mol)
    gt_maccs = maccs_fp(gt_mol)
    gt_scaf = scaffold_smiles(gt_mol)
    gt_props = mol_props(gt_mol)
    candidates = {}
    for seed_rank, repeat_seed in enumerate(repeat_seeds):
        for gen_rank, (gen_model, _, retriever, retrieval_k) in enumerate(generators):
            set_seed((repeat_seed + idx * 1000003 + gen_rank * 9176) % (2**32 - 1))
            retrieved_tokens = (
                retriever.retrieve_tokens(ctp, row_index, retrieval_k)
                if retriever is not None and retrieval_k > 0
                else None
            )
            generated = generate_ids_batch(
                gen_model,
                ctp,
                dose,
                cell,
                time,
                cfg,
                args.candidates_per_generator,
                args.temperature,
                args.top_k,
                args.top_p,
                args.fixed_zero_noise,
                retrieved_tokens=retrieved_tokens,
            )
            for ids in generated:
                smi = ids_to_smiles(
                    ids,
                    cache["vocab"]["itos"],
                    cfg["bos_id"],
                    cfg["eos_id"],
                    cfg["pad_id"],
                )
                can, ok = canonicalize(smi)
                if not ok or can in candidates:
                    continue
                mol = Chem.MolFromSmiles(can)
                if mol is None:
                    continue
                bits = morgan_bits(mol, args.fp_dim)
                fp_score = soft_tanimoto(pred_fp, bits)
                diff_score = (
                    soft_tanimoto(diff_fp, bits) if diff_fp is not None else 0.0
                )
                nn_score = hard_tanimoto_to_matrix(bits, nn_mat)
                gen_rank_score = (len(generators) - gen_rank) / max(len(generators), 1)
                seed_rank_score = (len(repeat_seeds) - seed_rank) / max(
                    len(repeat_seeds), 1
                )
                props = mol_props(mol)
                feature = make_feature(
                    args,
                    cache,
                    cfg,
                    ctp_cpu,
                    float(dose.item()),
                    int(cell.item()),
                    int(time.item()),
                    pred_fp,
                    diff_fp,
                    nn_mat,
                    bits,
                    fp_score,
                    diff_score,
                    nn_score,
                    gen_rank_score,
                    seed_rank_score,
                    can not in train_set,
                    len(ids),
                    props,
                )
                label = tanimoto(gt_morgan, morgan_fp(mol))
                if not np.isnan(label):
                    candidates[can] = {
                        "feature": feature,
                        "label": float(label),
                        "mol": mol,
                    }
    return (
        candidates,
        {"morgan": gt_morgan, "maccs": gt_maccs, "scaf": gt_scaf, "props": gt_props},
    )


def build_rows(
    cache,
    cfg,
    split,
    rows,
    generators,
    fp_model,
    diff_model,
    diff_dim,
    diff_target,
    fps,
    train_set,
    args,
    device,
    repeat_seeds,
):
    indices = list(map(int, cache["splits"][split]))
    if split == "train" and args.shuffle_train_rows:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(indices)
    indices = indices[:rows]
    out = []
    for idx in tqdm(indices, desc=f"build structural reranker {split}"):
        candidates, _ = make_row_candidates(
            cache,
            cfg,
            idx,
            generators,
            fp_model,
            diff_model,
            diff_dim,
            diff_target,
            fps,
            train_set,
            args,
            device,
            repeat_seeds,
        )
        if len(candidates) >= 2:
            feats = torch.stack([v["feature"] for v in candidates.values()])
            labels = torch.tensor(
                [v["label"] for v in candidates.values()], dtype=torch.float32
            )
            out.append((feats, labels))
    return out


def listwise_loss(scores, labels, target_temp: float):
    target = torch.softmax(labels / max(target_temp, 1e-06), dim=0)
    logp = torch.log_softmax(scores, dim=0)
    return -(target * logp).sum()


def top_weighted_regression_loss(scores, labels, top_frac: float):
    weights = torch.softmax(labels / 0.05, dim=0).detach()
    pred = torch.sigmoid(scores)
    return (weights * F.smooth_l1_loss(pred, labels, reduction="none")).sum()


def train_model(train_rows, args, device):
    in_dim = int(train_rows[0][0].size(1))
    model = StructuralReranker(in_dim, args.hidden, args.depth, args.dropout).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = len(train_rows)
    order = torch.randperm(n)
    cut = max(1, int(n * 0.9))
    tr_idx = order[:cut].tolist()
    va_idx = order[cut:].tolist()
    best_state = None
    best = float("inf")
    bad = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_order = tr_idx.copy()
        np.random.shuffle(epoch_order)
        losses = []
        opt.zero_grad(set_to_none=True)
        for step, row_id in enumerate(epoch_order, start=1):
            feats, labels = train_rows[row_id]
            feats = feats.to(device)
            labels = labels.to(device)
            scores = model(feats)
            loss = listwise_loss(scores, labels, args.target_temp)
            if args.reg_weight > 0:
                loss = loss + args.reg_weight * top_weighted_regression_loss(
                    scores, labels, args.top_frac
                )
            (loss / args.grad_accum).backward()
            if step % args.grad_accum == 0 or step == len(epoch_order):
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for row_id in va_idx:
                feats, labels = train_rows[row_id]
                feats = feats.to(device)
                labels = labels.to(device)
                scores = model(feats)
                val_losses.append(
                    float(listwise_loss(scores, labels, args.target_temp).item())
                )
        val = float(np.mean(val_losses)) if val_losses else float(np.mean(losses))
        tr = float(np.mean(losses))
        history.append({"epoch": epoch, "train_listwise": tr, "val_listwise": val})
        print(f"epoch {epoch:03d} | train {tr:.5f} | val {val:.5f}")
        if val < best:
            best = val
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.early_stop_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return (model, history, in_dim)


@torch.no_grad()
def eval_model(
    model,
    cache,
    cfg,
    split,
    rows,
    generators,
    fp_model,
    diff_model,
    diff_dim,
    diff_target,
    fps,
    train_set,
    args,
    device,
    repeat_seeds,
):
    model.eval()
    indices = list(map(int, cache["splits"][split]))[:rows]
    selected_total = selected_unique = selected_novel = rows_with_valid = 0
    gt_max_morgan = []
    gt_mean_morgan = []
    gt_max_maccs = []
    gt_mean_maccs = []
    gt_hit03 = gt_hit05 = gt_hit07 = gt_scaffold_hit = 0
    gt_prop_min_l1 = []
    for idx in tqdm(indices, desc=f"structural learned rerank eval {split}"):
        candidates, gt = make_row_candidates(
            cache,
            cfg,
            idx,
            generators,
            fp_model,
            diff_model,
            diff_dim,
            diff_target,
            fps,
            train_set,
            args,
            device,
            repeat_seeds,
        )
        if not candidates:
            continue
        cans = list(candidates.keys())
        feats = torch.stack([candidates[c]["feature"] for c in cans]).to(device)
        learned = model(feats).detach().cpu().numpy()
        manual = []
        for c in cans:
            f = candidates[c]["feature"]
            fp_score = float(f[-17].item())
            nn_score = float(f[-15].item())
            manual.append(args.w_fp * fp_score + args.w_nn * nn_score)
        scores = args.w_learned * learned + np.asarray(manual, dtype=np.float32)
        order = np.argsort(-scores)[: args.keep]
        selected_total += len(order)
        rows_with_valid += 1
        row_morgan = []
        row_maccs = []
        row_prop = []
        row_scaf = False
        seen = set()
        for pos in order:
            can = cans[int(pos)]
            mol = candidates[can]["mol"]
            selected_unique += int(can not in seen)
            seen.add(can)
            selected_novel += int(can not in train_set)
            row_morgan.append(tanimoto(gt["morgan"], morgan_fp(mol)))
            row_maccs.append(tanimoto(gt["maccs"], maccs_fp(mol)))
            gen_scaf = scaffold_smiles(mol)
            row_scaf = row_scaf or bool(
                gt["scaf"] and gen_scaf and (gen_scaf == gt["scaf"])
            )
            props = mol_props(mol)
            if gt["props"] is not None and props is not None:
                row_prop.append(float(np.mean(np.abs(props - gt["props"]))))
        if row_morgan:
            mmax = float(np.nanmax(row_morgan))
            gt_max_morgan.append(mmax)
            gt_mean_morgan.append(float(np.nanmean(row_morgan)))
            gt_hit03 += int(mmax >= 0.3)
            gt_hit05 += int(mmax >= 0.5)
            gt_hit07 += int(mmax >= 0.7)
        if row_maccs:
            gt_max_maccs.append(float(np.nanmax(row_maccs)))
            gt_mean_maccs.append(float(np.nanmean(row_maccs)))
        gt_scaffold_hit += int(row_scaf)
        if row_prop:
            gt_prop_min_l1.append(float(np.min(row_prop)))
    return {
        "split": split,
        "rows": len(indices),
        "gt_eval_rows": rows_with_valid,
        "selected_per_row": selected_total / max(len(indices), 1),
        "selected_unique_per_slot": selected_unique / max(len(indices) * args.keep, 1),
        "selected_novel_per_slot": selected_novel / max(len(indices) * args.keep, 1),
        "gt_max_morgan": (
            float(np.mean(gt_max_morgan)) if gt_max_morgan else float("nan")
        ),
        "gt_mean_morgan": (
            float(np.mean(gt_mean_morgan)) if gt_mean_morgan else float("nan")
        ),
        "gt_hit_morgan_0.3": gt_hit03 / max(rows_with_valid, 1),
        "gt_hit_morgan_0.5": gt_hit05 / max(rows_with_valid, 1),
        "gt_hit_morgan_0.7": gt_hit07 / max(rows_with_valid, 1),
        "gt_max_maccs": float(np.mean(gt_max_maccs)) if gt_max_maccs else float("nan"),
        "gt_mean_maccs": (
            float(np.mean(gt_mean_maccs)) if gt_mean_maccs else float("nan")
        ),
        "gt_scaffold_hit": gt_scaffold_hit / max(rows_with_valid, 1),
        "gt_min_property_l1": (
            float(np.mean(gt_prop_min_l1)) if gt_prop_min_l1 else float("nan")
        ),
    }


def main():
    p = argparse.ArgumentParser(
        description="Train a structural learned reranker on generated candidate pools."
    )
    p.add_argument("--cache", default="data/processed/cache.pt")
    p.add_argument("--fp_cache", default="data/features/fingerprints_2048.pt")
    p.add_argument("--generator_ckpts", required=True)
    p.add_argument("--fp_ckpt", required=True)
    p.add_argument("--diff_ckpt", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--model_out", default="")
    p.add_argument("--load_model", default="")
    p.add_argument("--train_rows", type=int, default=512)
    p.add_argument("--eval_rows", type=int, default=128)
    p.add_argument("--candidates_per_generator", type=int, default=8)
    p.add_argument("--keep", type=int, default=8)
    p.add_argument("--fp_dim", type=int, default=2048)
    p.add_argument("--nn_k", type=int, default=8)
    p.add_argument("--repeat_seeds", default="")
    p.add_argument("--diff_samples", type=int, default=2)
    p.add_argument("--diff_steps", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=80)
    p.add_argument("--top_p", type=float, default=0.97)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--early_stop_patience", type=int, default=3)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--clip_norm", type=float, default=1.0)
    p.add_argument("--target_temp", type=float, default=0.04)
    p.add_argument("--reg_weight", type=float, default=0.1)
    p.add_argument("--top_frac", type=float, default=0.25)
    p.add_argument("--w_fp", type=float, default=1.0)
    p.add_argument("--w_nn", type=float, default=1.5)
    p.add_argument("--w_learned", type=float, default=1.0)
    p.add_argument("--use_bits", action="store_true")
    p.add_argument("--use_pred_fp", action="store_true")
    p.add_argument("--use_nn_mean", action="store_true")
    p.add_argument("--use_ctp", action="store_true")
    p.add_argument("--shuffle_train_rows", action="store_true")
    p.add_argument("--fixed_zero_noise", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    cache = safe_torch_load(args.cache, map_location="cpu")
    cfg = cache["config"]
    loaded = None
    if args.load_model:
        loaded = safe_torch_load(args.load_model, map_location="cpu")
        saved_args = loaded.get("args", {})
        for name in [
            "use_bits",
            "use_pred_fp",
            "use_nn_mean",
            "use_ctp",
            "hidden",
            "depth",
            "dropout",
        ]:
            if name in saved_args:
                setattr(args, name, saved_args[name])
    gen_paths = [p for p in args.generator_ckpts.split(",") if p]
    repeat_seeds = parse_repeat_seeds(args.repeat_seeds, args.seed)
    generators = [load_generator(path, cache, device) for path in gen_paths]
    fp_model, _ = load_fp_model(args.fp_ckpt, cache, device)
    diff_model = diff_dim = diff_target = None
    if args.diff_ckpt:
        diff_model, diff_dim, diff_target = load_diff_model(
            args.diff_ckpt, cache, device
        )
    fps, _, _ = load_or_build_fps(cache, args.fp_dim, args.fp_cache, fp_workers=8)
    train_set = train_set_canonical(cache)
    if loaded is None:
        train_rows = build_rows(
            cache,
            cfg,
            "train",
            args.train_rows,
            generators,
            fp_model,
            diff_model,
            diff_dim,
            diff_target,
            fps,
            train_set,
            args,
            device,
            repeat_seeds,
        )
        model, history, in_dim = train_model(train_rows, args, device)
    else:
        train_rows = []
        history = loaded.get("history", [])
        in_dim = int(loaded["feature_dim"])
        model = StructuralReranker(in_dim, args.hidden, args.depth, args.dropout).to(
            device
        )
        model.load_state_dict(loaded["model"])
    ev = eval_model(
        model,
        cache,
        cfg,
        "val",
        args.eval_rows,
        generators,
        fp_model,
        diff_model,
        diff_dim,
        diff_target,
        fps,
        train_set,
        args,
        device,
        repeat_seeds,
    )
    payload = {
        "generator_ckpts": gen_paths,
        "repeat_seeds": repeat_seeds,
        "train_rows_requested": args.train_rows,
        "train_rows_built": len(train_rows),
        "load_model": args.load_model,
        "feature_dim": in_dim,
        "history": history,
        "eval_val": ev,
        "args": vars(args),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    if args.model_out:
        model_path = Path(args.model_out)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "feature_dim": in_dim,
                "args": vars(args),
                "history": history,
            },
            model_path,
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
