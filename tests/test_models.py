import unittest

import torch
import torch.nn.functional as F

from cider.models import ForwardGenerator, ReversePredictor
from cider.training.ensemble import make_token_model
from cider.training.fingerprint import FingerprintPredictor
from cider.training.forward_generator import (
    CVAEAdaLNForwardGenerator,
    RetrievalAdaLNForwardGenerator,
)
from cider.training.reverse_perceiver import PerceiverReversePredictor
from cider.training.reverse_tabular import TabularReversePredictor
from cider.training.reverse_transformer import TransformerReversePredictor
from cider.training.structural_ranker import StructuralReranker, listwise_loss


class ModelTests(unittest.TestCase):
    def test_checkpoint_architecture_loading(self):
        configurations = [
            (ReversePredictor, {**self.common, "num_layers": 1}),
            (
                TransformerReversePredictor,
                {**self.common, "num_layers": 1, "arch": "mean", "head_blocks": 1},
            ),
            (
                PerceiverReversePredictor,
                {
                    **self.common,
                    "latent_len": 4,
                    "latent_layers": 1,
                    "decoder_layers": 1,
                    "ensemble_size": 2,
                },
            ),
        ]
        for model_class, arguments in configurations:
            with self.subTest(model=model_class.__name__):
                original = model_class(**arguments)
                restored = make_token_model({"model_args": arguments})
                restored.load_state_dict(original.state_dict(), strict=True)
                self.assertIsInstance(restored, model_class)

    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.tokens = torch.tensor([[1, 4, 5, 2, 0], [1, 6, 4, 2, 0]])
        self.response = torch.randn(2, 8)
        self.dose = torch.tensor([0.1, -0.2])
        self.cell = torch.tensor([0, 1])
        self.time = torch.tensor([0, 0])
        self.common = dict(
            y_dim=8,
            vocab_size=12,
            pad_id=0,
            n_cell=2,
            n_time=1,
            max_len=8,
            d_model=16,
            nhead=2,
            dropout=0.0,
        )

    def check_step(self, model, loss):
        self.assertTrue(bool(torch.isfinite(loss)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
        optimizer.step()

    def test_reverse_transformer(self):
        model = TransformerReversePredictor(
            **self.common, num_layers=1, dim_feedforward=32, arch="mean", head_blocks=1
        )
        pred = model.forward_tokens(self.tokens, self.dose, self.cell, self.time)
        self.assertEqual(pred.shape, (2, 8))
        self.check_step(model, F.mse_loss(pred, self.response))

    def test_reverse_perceiver(self):
        model = PerceiverReversePredictor(
            **self.common,
            latent_len=4,
            latent_layers=1,
            decoder_layers=1,
            ffn_mult=2,
            ensemble_size=2
        )
        pred = model.forward_tokens(self.tokens, self.dose, self.cell, self.time)
        self.assertEqual(pred.shape, (2, 8))
        self.check_step(model, F.mse_loss(pred, self.response))

    def test_reverse_tabular(self):
        model = TabularReversePredictor(
            in_dim=10,
            y_dim=8,
            n_cell=2,
            n_time=1,
            d_model=16,
            blocks=1,
            hidden_mult=2,
            dropout=0.0,
            ensemble_size=2,
        )
        pred = model(torch.randn(2, 10), self.dose, self.cell, self.time)
        self.assertEqual(pred.shape, (2, 8))
        self.check_step(model, F.mse_loss(pred, self.response))

    def test_forward_prefix(self):
        model = ForwardGenerator(
            **self.common, num_layers=1, dim_feedforward=32, prefix_len=2, noise_dim=4
        )
        logits = model(
            self.response, self.dose, self.cell, self.time, self.tokens[:, :-1]
        )
        self.assertEqual(logits.shape, (2, 4, 12))
        self.check_step(
            model,
            F.cross_entropy(
                logits.reshape(-1, 12), self.tokens[:, 1:].reshape(-1), ignore_index=0
            ),
        )

    def test_forward_cvae(self):
        model = CVAEAdaLNForwardGenerator(
            **self.common, num_layers=1, noise_dim=4, cond_depth=1
        )
        logits, kl = model(
            self.response,
            self.dose,
            self.cell,
            self.time,
            self.tokens[:, :-1],
            posterior_tokens=self.tokens,
        )
        loss = (
            F.cross_entropy(
                logits.reshape(-1, 12), self.tokens[:, 1:].reshape(-1), ignore_index=0
            )
            + 0.01 * kl
        )
        self.check_step(model, loss)

    def test_forward_retrieval(self):
        model = RetrievalAdaLNForwardGenerator(
            **self.common, num_layers=1, noise_dim=4, cond_depth=1, retrieval_k=1
        )
        logits = model(
            self.response,
            self.dose,
            self.cell,
            self.time,
            self.tokens[:, :-1],
            retrieved_tokens=self.tokens[:, None, :],
        )
        self.assertEqual(logits.shape, (2, 4, 12))
        self.check_step(
            model,
            F.cross_entropy(
                logits.reshape(-1, 12), self.tokens[:, 1:].reshape(-1), ignore_index=0
            ),
        )

    def test_fingerprint_predictor(self):
        model = FingerprintPredictor(8, 2, 1, 16, 32, 0.0, 1)
        logits = model(self.response, self.dose, self.cell, self.time)
        self.assertEqual(logits.shape, (2, 32))
        self.check_step(
            model,
            F.binary_cross_entropy_with_logits(
                logits, torch.randint(0, 2, (2, 32)).float()
            ),
        )

    def test_structural_ranker(self):
        model = StructuralReranker(10, 16, 2, 0.0)
        scores = model(torch.randn(5, 10))
        self.check_step(model, listwise_loss(scores, torch.linspace(0, 1, 5), 0.04))


if __name__ == "__main__":
    unittest.main()
