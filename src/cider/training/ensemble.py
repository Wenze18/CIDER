from __future__ import annotations
import argparse
from cider.paths import resolve_path
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from cider.models import ReversePredictor
from cider.utils import (
    batch_cosine,
    batch_pearson,
    get_device,
    make_loader,
    safe_torch_load,
)
from cider.training.tabular_ensemble import (
    FeatureDataset,
    TabularReversePredictor,
    build_or_load_feature_cache,
)
from cider.training.reverse_transformer import TransformerReversePredictor
from cider.training.reverse_perceiver import PerceiverReversePredictor


def parse_named_spec(spec: str) -> tuple[str, list[str]]:
    if "=" not in spec:
        raise ValueError(f"Expected name=path[,path...] but got: {spec}")
    name, paths = spec.split("=", 1)
    ckpts = [p for p in paths.split(",") if p]
    if not name or not ckpts:
        raise ValueError(f"Bad candidate spec: {spec}")
    return (name, ckpts)


def metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "mse": float(F.mse_loss(pred, target).item()),
        "pearson": batch_pearson(pred, target),
        "cosine": batch_cosine(pred, target),
    }


@torch.no_grad()
def predict_tabular_group(
    ckpt_paths: list[str],
    cache: dict,
    split: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    preds = []
    target = None
    for ckpt_path in ckpt_paths:
        ckpt = safe_torch_load(ckpt_path, map_location="cpu")
        args = ckpt["args"]
        features = build_or_load_feature_cache(
            cache,
            Path(resolve_path(f"data/features/rdkit_{int(args['morgan_bits'])}.pt")),
            int(args["morgan_bits"]),
        )
        loader = DataLoader(
            FeatureDataset(cache, features, split),
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.get("num_workers", 0),
            pin_memory=torch.cuda.is_available(),
        )
        model = TabularReversePredictor(**ckpt["model_args"]).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        pred_parts = []
        target_parts = []
        for batch in loader:
            pred = model(
                batch["features"].to(device, non_blocking=True),
                batch["dose"].to(device, non_blocking=True),
                batch["cell"].to(device, non_blocking=True),
                batch["time"].to(device, non_blocking=True),
            )
            pred_parts.append(pred.cpu())
            target_parts.append(batch["ctp"])
        preds.append(torch.cat(pred_parts))
        if target is None:
            target = torch.cat(target_parts)
    return (torch.stack(preds, dim=0).mean(dim=0), target)


def make_token_model(ckpt: dict):
    model_args = dict(ckpt["model_args"])
    if "arch" in model_args:
        return TransformerReversePredictor(**model_args)
    if "latent_len" in model_args:
        return PerceiverReversePredictor(**model_args)
    return ReversePredictor(**model_args)


@torch.no_grad()
def predict_token_ckpt(
    ckpt_path: str, cache: dict, split: str, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    loader = make_loader(
        cache,
        split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=ckpt.get("args", {}).get("num_workers", 0),
    )
    model = make_token_model(ckpt).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    preds = []
    targets = []
    for batch in loader:
        pred = model.forward_tokens(
            batch["tokens"].to(device, non_blocking=True),
            batch["dose"].to(device, non_blocking=True),
            batch["cell"].to(device, non_blocking=True),
            batch["time"].to(device, non_blocking=True),
        )
        preds.append(pred.cpu())
        targets.append(batch["ctp"])
    return (torch.cat(preds), torch.cat(targets))


def project_simplex(v: torch.Tensor) -> torch.Tensor:
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1
    ind = torch.arange(1, v.numel() + 1, dtype=v.dtype, device=v.device)
    cond = u - cssv / ind > 0
    rho = int(torch.nonzero(cond, as_tuple=False)[-1])
    theta = cssv[rho] / float(rho + 1)
    return torch.clamp(v - theta, min=0)


def fit_simplex_weights(
    preds: torch.Tensor, target: torch.Tensor, steps: int = 1500, lr: float = 0.2
) -> torch.Tensor:
    k = preds.size(0)
    flat_p = preds.reshape(k, -1).double()
    flat_y = target.reshape(-1).double()
    gram = flat_p @ flat_p.t() / flat_y.numel()
    rhs = flat_p @ flat_y / flat_y.numel()
    largest = float(torch.linalg.eigvalsh(gram).max().clamp_min(1e-08).item())
    step_size = min(lr, 1.0 / largest)
    w = torch.full((k,), 1.0 / k, dtype=torch.double)
    for _ in range(steps):
        grad = gram @ w - rhs
        w = project_simplex(w - step_size * grad)
    return w.float()


def blend_from_weights(preds: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (preds * weights[:, None, None]).sum(dim=0)


def main():
    p = argparse.ArgumentParser(
        description="Evaluate cross-family reverse-model blends."
    )
    p.add_argument("--cache", default="data/processed/cache.pt")
    p.add_argument("--manifest")
    p.add_argument("--out_manifest")
    p.add_argument("--fit_split", default="val")
    p.add_argument("--eval_split", default="test")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--out_json", default="outputs/training/blend/cross_family_blend.json"
    )
    p.add_argument(
        "--tabular",
        action="append",
        default=[],
        help="Candidate group as name=ckpt[,ckpt...]",
    )
    p.add_argument(
        "--token", action="append", default=[], help="Candidate as name=ckpt"
    )
    args = p.parse_args()
    manifest = None
    if args.manifest:
        if args.tabular or args.token:
            p.error(
                "Use --manifest or explicit --tabular/--token candidates, not both."
            )
        manifest = json.loads(resolve_path(args.manifest).read_text())
        args.tabular = ["tabular=" + ",".join(manifest["tabular_checkpoints"])]
        args.token = [
            f"{name}={path}" for name, path in manifest["token_checkpoints"].items()
        ]
    if args.out_manifest and manifest is None:
        p.error("--out_manifest requires --manifest.")
    device = get_device(args.device)
    cache = safe_torch_load(args.cache, map_location="cpu")
    candidates = [("tabular", *parse_named_spec(spec)) for spec in args.tabular]
    candidates += [("token", *parse_named_spec(spec)) for spec in args.token]
    if len(candidates) < 2:
        raise ValueError("Need at least two candidates to blend.")

    def predict_all(split: str):
        pred_list = []
        target = None
        per_candidate = []
        for kind, name, ckpts in candidates:
            if kind == "tabular":
                pred, y = predict_tabular_group(
                    ckpts, cache, split, args.batch_size, device
                )
            else:
                if len(ckpts) != 1:
                    raise ValueError(
                        f"Token candidate {name} must have exactly one checkpoint."
                    )
                pred, y = predict_token_ckpt(
                    ckpts[0], cache, split, args.batch_size, device
                )
            if target is None:
                target = y
            pred_list.append(pred.float())
            per_candidate.append(
                {
                    "name": name,
                    "kind": kind,
                    "checkpoints": ckpts,
                    **metrics(pred.float(), target),
                }
            )
            print(split, name, per_candidate[-1])
        return (torch.stack(pred_list, dim=0), target.float(), per_candidate)

    fit_preds, fit_target, fit_per_candidate = predict_all(args.fit_split)
    weights = fit_simplex_weights(fit_preds, fit_target)
    fit_blend = blend_from_weights(fit_preds, weights)
    eval_preds, eval_target, eval_per_candidate = predict_all(args.eval_split)
    eval_blend = blend_from_weights(eval_preds, weights)
    result = {
        "fit_split": args.fit_split,
        "eval_split": args.eval_split,
        "candidate_names": [name for _, name, _ in candidates],
        "weights": {name: float(w) for (_, name, _), w in zip(candidates, weights)},
        "fit_per_candidate": fit_per_candidate,
        "fit_blend": metrics(fit_blend, fit_target),
        "eval_per_candidate": eval_per_candidate,
        "eval_blend": metrics(eval_blend, eval_target),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    if args.out_manifest:
        manifest["weights"] = result["weights"]
        destination = resolve_path(args.out_manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
