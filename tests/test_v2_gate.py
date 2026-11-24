"""
NonNULL Framework V2.0 - Unit Tests for HiddenStateGate & SplitLayerGateController (1-Pass)
File: tests/test_v2_gate.py
"""

import os
import sys
import tempfile
import unittest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from doctrine.hidden_gate import HiddenStateGate, SplitLayerGateController


class ToyLoRALinear(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.base = nn.Linear(dim, dim, bias=False)
        self.lora = nn.Linear(dim, dim, bias=False)
        self.scaling = {"default": 1.0}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling["default"] * self.lora(x)


class ToyLayer(nn.Module):
    def __init__(self, dim: int, has_lora: bool):
        super().__init__()
        self.proj = ToyLoRALinear(dim) if has_lora else nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ToySplitModel(nn.Module):
    """6-layer mock Transformer with frozen lower layers (0..2) and LoRA upper layers (3..5)."""

    def __init__(self, dim: int = 16, split_layer: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer(dim, has_lora=(i >= split_layer)) for i in range(6)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


class TestV2HiddenGate(unittest.TestCase):
    def test_gate_forward_routing_and_save_load(self):
        gate = HiddenStateGate(
            hidden_size=16,
            probe_layers=[0, 1, 2],
            gate_layer=3,
            mlp_hidden_1=12,
            mlp_hidden_2=6,
            tau_zero=0.30,
            tau_high=0.65,
        )
        vecs = torch.randn(2, 3, 16)
        probs = gate(vecs)
        self.assertEqual(probs.shape, (2,))
        self.assertTrue(torch.all((probs >= 0.0) & (probs <= 1.0)))

        # Verify 3 routing regimes
        a_zp, reg_zp, is_zp = gate.route_alpha(0.10)
        self.assertEqual(a_zp, 0.0)
        self.assertEqual(reg_zp, "ZERO_POINT")
        self.assertTrue(is_zp)

        a_bd, reg_bd, is_bd = gate.route_alpha(0.475)
        self.assertTrue(0.15 <= a_bd <= 1.0)
        self.assertEqual(reg_bd, "BORDERLINE")
        self.assertFalse(is_bd)

        a_hi, reg_hi, is_hi = gate.route_alpha(0.90)
        self.assertEqual(a_hi, 1.0)
        self.assertEqual(reg_hi, "ACTIVE_VAN")
        self.assertFalse(is_hi)

        # Save & reload roundtrip
        with tempfile.TemporaryDirectory() as tmpdir:
            gpath = os.path.join(tmpdir, "gate.pt")
            gate.save(gpath, extra_meta={"val_roc_auc": 0.99})
            loaded = HiddenStateGate.load(gpath)
            self.assertEqual(loaded.gate_layer, 3)
            self.assertEqual(loaded.probe_layers, [0, 1, 2])
            gate.eval()
            self.assertTrue(torch.allclose(gate(vecs), loaded(vecs), atol=1e-6))

    def test_split_layer_1pass_controller_latches_and_restores(self):
        model = ToySplitModel(dim=16, split_layer=3)
        gate = HiddenStateGate(hidden_size=16, probe_layers=[0, 1, 2], gate_layer=3, mlp_hidden_1=12, mlp_hidden_2=6)
        ctrl = SplitLayerGateController(model, gate)

        self.assertFalse(ctrl.has_lower_layer_lora())

        x = torch.randn(1, 5, 16)
        with ctrl.activate():
            _ = model(x)
            self.assertIn(ctrl.last_routing["regime"], ("ZERO_POINT", "BORDERLINE", "ACTIVE_VAN"))
            self.assertTrue(ctrl.last_routing["true_1pass_split"])
            expected_alpha = ctrl.last_routing["alpha"]
            # Upper layers must have latched scaling = expected_alpha during active context
            for idx in range(3, 6):
                self.assertAlmostEqual(model.layers[idx].proj.scaling["default"], expected_alpha, places=5)

        # After exiting context, all LoRA scalings must be restored to 1.0
        for idx in range(3, 6):
            self.assertAlmostEqual(model.layers[idx].proj.scaling["default"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
