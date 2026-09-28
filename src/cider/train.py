import argparse
import importlib
import json
import sys
from pathlib import Path

from cider.paths import resolve_path


REVERSE_RECIPES = [
    *[f"reverse_tabular_{index:02d}" for index in range(1, 10)],
    "reverse_transformer",
    "reverse_perceiver",
]
FORWARD_RECIPES = [
    "forward_prefix",
    "forward_wide_prefix",
    "forward_retrieval",
    "forward_exploratory_cvae",
    "forward_fingerprint",
    "forward_neighbor",
    "forward_multipositive_prefix_a",
    "forward_multipositive_prefix_b",
    "forward_multipositive_neighbor",
    "forward_structural_ranker",
]


def recipe_arguments(recipe, device=None, epochs=None):
    argv = list(recipe["argv"])
    for flag, value in [("--device", device), ("--epochs", epochs)]:
        if value is None:
            continue
        if flag in argv:
            argv[argv.index(flag) + 1] = str(value)
        else:
            argv.extend([flag, str(value)])
    return argv


def main():
    parser = argparse.ArgumentParser(
        description="Train a CIDER component from a saved recipe."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--recipe")
    selection.add_argument("--stage", choices=["reverse", "forward", "all"])
    parser.add_argument("--recipe-dir", default="configs/training")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.recipe:
        paths = [resolve_path(args.recipe)]
    else:
        names = {
            "reverse": REVERSE_RECIPES,
            "forward": FORWARD_RECIPES,
            "all": REVERSE_RECIPES + FORWARD_RECIPES,
        }[args.stage]
        paths = [resolve_path(args.recipe_dir) / f"{name}.json" for name in names]
    jobs = []
    for path in paths:
        recipe = json.loads(path.read_text())
        if not recipe["module"].startswith("cider.training."):
            raise ValueError("Recipe module must belong to cider.training.")
        jobs.append(
            {**recipe, "argv": recipe_arguments(recipe, args.device, args.epochs)}
        )
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    for job in jobs:
        destination = resolve_path(job["output_checkpoint"])
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Checkpoint already exists: {destination}")
    for job in jobs:
        sys.argv = [job["module"], *job["argv"]]
        importlib.import_module(job["module"]).main()
        if not resolve_path(job["output_checkpoint"]).is_file():
            raise FileNotFoundError(job["output_checkpoint"])


if __name__ == "__main__":
    main()
