import argparse
import importlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from cider.chemistry import canonicalize_smiles, mol_props, tanimoto_between
from cider.generate import PRESETS, apply_preset, parse_indices
from cider.paths import resolve_path
from cider.selection import rank01, reverse_metric_arrays, select_with_cycle
from cider.train import (
    FORWARD_RECIPES,
    REVERSE_RECIPES,
    main as train_main,
    recipe_arguments,
)
from cider.training.ensemble import fit_simplex_weights, main as ensemble_main


ROOT = Path(__file__).parent.parent


class ParserComplete(Exception):
    pass


class InferenceTests(unittest.TestCase):
    def test_training_dependency_order(self):
        available = set()
        for name in REVERSE_RECIPES + FORWARD_RECIPES:
            recipe = json.loads(
                (ROOT / "configs/training" / f"{name}.json").read_text()
            )
            for index, value in enumerate(recipe["argv"]):
                if index > 0 and recipe["argv"][index - 1] in ["--out", "--model_out"]:
                    continue
                for part in value.split(","):
                    if part.startswith("checkpoints/"):
                        self.assertIn(part, available, name)
            available.add(recipe["output_checkpoint"])
        forward = json.loads((ROOT / "configs/forward.json").read_text())
        reverse = json.loads((ROOT / "configs/reverse.json").read_text())
        required = (
            forward["conservative"]["generator_ckpts"]
            + [
                forward["conservative"]["fp_ckpt"],
                forward["conservative"]["structural_reranker_ckpt"],
                forward["exploratory"]["generator_ckpt"],
            ]
            + reverse["tabular_checkpoints"]
            + list(reverse["token_checkpoints"].values())
        )
        self.assertTrue(set(required).issubset(available))

    def test_training_stage_dry_run(self):
        stream = io.StringIO()
        with patch("sys.argv", ["cider-train", "--stage", "all", "--dry-run"]):
            with patch("sys.stdout", stream):
                train_main()
        jobs = json.loads(stream.getvalue())
        self.assertEqual(len(jobs), 21)

    def test_training_output_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory(dir=".") as folder:
            root = Path(folder)
            output = root / "model.pt"
            recipe = root / "recipe.json"
            recipe.write_text(
                json.dumps(
                    {
                        "module": "cider.training.forward_generator",
                        "argv": [],
                        "output_checkpoint": str(output),
                    }
                )
            )
            module = SimpleNamespace(main=lambda: output.write_bytes(b"checkpoint"))
            with patch("sys.argv", ["train", "--recipe", str(recipe)]), patch(
                "cider.train.importlib.import_module", return_value=module
            ) as imported:
                train_main()
                imported.assert_called_once()
            self.assertTrue(output.is_file())
            with patch("sys.argv", ["train", "--recipe", str(recipe)]), patch(
                "cider.train.importlib.import_module"
            ) as imported:
                with self.assertRaises(FileExistsError):
                    train_main()
                imported.assert_not_called()

    def test_configuration_paths_are_relative(self):
        for path in (ROOT / "configs").rglob("*.json"):
            values = [json.loads(path.read_text())]
            while values:
                value = values.pop()
                if isinstance(value, dict):
                    values.extend(value.values())
                elif isinstance(value, list):
                    values.extend(value)
                elif isinstance(value, str):
                    for part in value.split(","):
                        self.assertFalse(Path(part).is_absolute(), (path, part))

    def test_relative_paths_and_training_overrides(self):
        self.assertEqual(
            resolve_path("checkpoints/forward/prefix.pt"),
            Path("checkpoints/forward/prefix.pt"),
        )
        with self.assertRaises(ValueError):
            resolve_path(Path.cwd())
        recipe = {
            "argv": [
                "--generator_ckpts",
                "checkpoints/forward/prefix.pt,checkpoints/forward/neighbor.pt",
                "--epochs",
                "10",
            ]
        }
        argv = recipe_arguments(recipe, device="cpu", epochs=1)
        self.assertEqual(argv[1], recipe["argv"][1])
        self.assertEqual(argv[3], "1")
        self.assertEqual(argv[-2:], ["--device", "cpu"])
        self.assertEqual(recipe["argv"][3], "10")

    def test_ensemble_manifest_export(self):
        with tempfile.TemporaryDirectory(dir=".") as folder:
            root = Path(folder)
            target = torch.tensor([[0.0, 1.0, 2.0]])
            manifest = ROOT / "configs/reverse.json"
            argv = [
                "ensemble",
                "--manifest",
                str(manifest.relative_to(Path.cwd())),
                "--out_manifest",
                str(root / "ensemble.json"),
                "--out_json",
                str(root / "metrics.json"),
                "--device",
                "cpu",
            ]
            with patch("sys.argv", argv), patch("sys.stdout", io.StringIO()), patch(
                "cider.training.ensemble.safe_torch_load", return_value={}
            ), patch(
                "cider.training.ensemble.predict_tabular_group",
                return_value=(target, target),
            ), patch(
                "cider.training.ensemble.predict_token_ckpt",
                return_value=(target + 1, target),
            ):
                ensemble_main()
            result = json.loads((root / "ensemble.json").read_text())
            self.assertGreater(result["weights"]["tabular"], 0.99)
            self.assertEqual(
                result["token_checkpoints"],
                json.loads(manifest.read_text())["token_checkpoints"],
            )

    def test_training_recipes_parse(self):
        parse = argparse.ArgumentParser.parse_args
        for path in sorted((ROOT / "configs/training").glob("*.json")):
            recipe = json.loads(path.read_text())
            argv = recipe_arguments(recipe, device="cpu")

            def capture(parser, *args, **kwargs):
                parsed = parse(parser, argv)
                self.assertGreater(parsed.epochs, 0)
                self.assertLessEqual(parsed.epochs, 10)
                raise ParserComplete()

            with self.subTest(recipe=path.name):
                self.assertTrue(recipe["output_checkpoint"].startswith("checkpoints/"))
                module = importlib.import_module(recipe["module"])
                with patch.object(argparse.ArgumentParser, "parse_args", capture):
                    with self.assertRaises(ParserComplete):
                        module.main()

    def test_sample_order_and_explicit_rows(self):
        cache = {"splits": {"test": list(range(20))}}
        args = SimpleNamespace(
            row_indices="", split="test", sample_rows=True, row_seed=20260507, rows=5
        )
        expected = (
            np.random.default_rng(20260507)
            .choice(list(range(20)), size=5, replace=False)
            .tolist()
        )
        self.assertEqual(parse_indices(cache, args), expected)
        args.row_indices = "8,2,1"
        self.assertEqual(parse_indices(cache, args), [8, 2, 1])

    def test_presets_and_explicit_overrides(self):
        args = SimpleNamespace(version="conservative", keep=3, w_cycle_rank=None)
        apply_preset(args)
        self.assertEqual(args.keep, 3)
        self.assertEqual(args.w_cycle_rank, 0.75)
        self.assertEqual(PRESETS["exploratory"]["candidates"], 512)
        self.assertEqual(PRESETS["exploratory"]["base_rank_floor"], 0.9)

    def test_response_metrics(self):
        target = torch.tensor([[1.0, 2.0, 4.0]])
        result = reverse_metric_arrays(target, target, 0.2)
        self.assertAlmostEqual(float(result["reverse_pearson"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(result["reverse_cosine"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(result["reverse_mse"][0]), 0.0, places=6)
        self.assertAlmostEqual(float(result["cycle_score"][0]), 0.0, places=6)

    def test_ranking_and_selection(self):
        np.testing.assert_array_equal(rank01([3, 1, 2]), [1, 0, 0.5])
        np.testing.assert_array_equal(rank01([3, 1, 2], False), [0, 1, 0.5])
        args = SimpleNamespace(
            keep=1,
            base_rank_floor=0.9,
            w_base_rank=1,
            w_cycle_rank=1,
            w_pearson_rank=1,
            w_cosine_rank=0.25,
            w_mse_rank=0.25,
        )
        pool = [{"smiles": "C", "base_score": 0}, {"smiles": "CC", "base_score": 1}]
        metrics = {
            "reverse_pearson": np.array([1, 0]),
            "reverse_cosine": np.array([1, 0]),
            "reverse_mse": np.array([0, 1]),
            "cycle_score": np.array([0, 1]),
        }
        self.assertEqual(select_with_cycle(pool, metrics, args), [1])

    def test_simplex_fit(self):
        target = torch.tensor([[0.0, 1.0, 2.0]])
        predictions = torch.stack([target, target + 1])
        weights = fit_simplex_weights(predictions, target)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertTrue(bool((weights >= 0).all()))
        self.assertGreater(float(weights[0]), 0.99)

    def test_chemical_metrics(self):
        self.assertEqual(canonicalize_smiles("OCC"), ("CCO", True))
        self.assertAlmostEqual(tanimoto_between("CCO", "OCC"), 1.0)
        props = mol_props("CCO")
        self.assertAlmostEqual(props["mw"], 46.069, places=3)
        self.assertTrue(np.isfinite(props["sas"]))
        self.assertTrue(props["lipinski"])


if __name__ == "__main__":
    unittest.main()
