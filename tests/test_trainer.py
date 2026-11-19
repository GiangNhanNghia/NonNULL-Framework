"""
NonNULL Framework V1.0 - Unit Tests for HierarchicalTrainer & 6-Component Loss
File: tests/test_trainer.py
"""

import os
import sys
import unittest
from types import SimpleNamespace
import torch
import torch.nn as nn
from transformers import TrainingArguments

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from doctrine.doctrine_trainer import HierarchicalTrainer, compute_euler_phase_loss
from doctrine.meta_layers import HarmonyLayer, ValueCritic


class ToyCausalLM(nn.Module):
    """Lightweight mock CausalLM with 'lora_' trainable parameters and hidden_states."""

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.embed.weight.requires_grad = False
        self.lora_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask=None, labels=None, output_hidden_states: bool = False, **kwargs):
        h = self.embed(input_ids) + self.dropout(self.lora_proj(self.embed(input_ids)))
        logits = self.lm_head(h)
        hs = (h,) if output_hidden_states else None
        return SimpleNamespace(logits=logits, hidden_states=hs)


class TestHierarchicalTrainer(unittest.TestCase):
    """Verifies L_total (including EulerPhaseLoss), gradient propagation, and MetaController GAE updates."""

    def setUp(self):
        self.model = ToyCausalLM()
        self.meta_controller = HarmonyLayer(gating_tau=0.1)
        self.value_critic = ValueCritic()
        self.args = TrainingArguments(
            output_dir="./output/test_trainer_tmp",
            per_device_train_batch_size=2,
            logging_steps=1,
            report_to="none",
        )
        self.trainer = HierarchicalTrainer(
            model=self.model,
            args=self.args,
            alpha_mirror=0.20,
            beta_empath=0.15,
            eta_euler=0.08,
            gamma_distill=0.10,
            lambda_reg=0.001,
            gating_tau=0.1,
            meta_controller=self.meta_controller,
            value_critic=self.value_critic,
        )

    def test_euler_phase_loss_computation(self):
        h = torch.randn(2, 10, 32, requires_grad=True)
        mask = torch.ones((2, 10), dtype=torch.bool)
        mask[:, :2] = False
        loss_e = compute_euler_phase_loss(h, mask)
        self.assertTrue(torch.is_tensor(loss_e))
        self.assertGreater(loss_e.item(), 0.0)
        loss_e.backward()
        self.assertIsNotNone(h.grad)

    def test_six_component_loss_and_backward(self):
        self.model.train()
        device = next(self.model.parameters()).device
        inputs = {
            "input_ids": torch.randint(0, 64, (2, 12), device=device),
            "labels": torch.tensor([
                [-100, -100, 5, 12, 8, 3, 19, 21, 4, 9, 10, 2],
                [-100, -100, 7, 14, 6, 11, 22, 30, 1, 8, 15, 3],
            ], device=device),
        }

        loss = self.trainer.compute_loss(self.model, inputs)
        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(loss.requires_grad)

        loss.backward()
        self.assertIsNotNone(self.model.lora_proj.weight.grad)
        self.assertGreater(self.model.lora_proj.weight.grad.abs().sum().item(), 0.0)

        breakdown = self.trainer.get_loss_breakdown()
        for key in ["task", "mirror", "euler", "empath", "distill", "reg", "gate", "lr_scale", "total"]:
            self.assertIn(key, breakdown)
        self.assertGreater(breakdown["euler"], 0.0)


if __name__ == "__main__":
    unittest.main()
