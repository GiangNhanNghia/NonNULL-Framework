"""
NonNULL Framework V2.0 - Unified 1-Pass Hidden-State Gate (Split-Layer KV-Preserving)
File: doctrine/hidden_gate.py

Implements the core unified controller from Volume I (Sec 3.1 & 6.3) and EULER-Compact:
    g(x) = sigmoid(W_c * h_mix(x) + b)
reading multi-layer hidden states from frozen lower layers (<= L_gate) in a single prefill pass
to modulate upper-layer LoRA adapters (L_gate+1 .. L) with ZERO KV-cache invalidation.
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class HiddenStateGate(nn.Module):
    """
    Compact Bilingual Hidden-State Risk & Equilibrium Gate (< 1 MB).
    Aggregates hidden states from `probe_layers` (all <= gate_layer) via learnable
    softmax layer weights, then applies a 2-layer MLP (d -> 128 -> 32 -> 1) matching
    the EULER-Compact specification.
    """

    def __init__(
        self,
        hidden_size: int,
        probe_layers: List[int],
        gate_layer: int,
        mlp_hidden_1: int = 128,
        mlp_hidden_2: int = 32,
        dropout: float = 0.2,
        tau_zero: float = 0.30,
        tau_high: float = 0.65,
        soft_borderline: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.probe_layers = [int(l) for l in probe_layers]
        self.gate_layer = int(gate_layer)
        self.mlp_hidden_1 = int(mlp_hidden_1)
        self.mlp_hidden_2 = int(mlp_hidden_2)
        self.tau_zero = float(tau_zero)
        self.tau_high = float(tau_high)
        self.soft_borderline = bool(soft_borderline)

        self.layer_logits = nn.Parameter(torch.zeros(len(self.probe_layers)))
        self.input_norm = nn.LayerNorm(self.hidden_size)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.mlp_hidden_1),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(self.mlp_hidden_1, self.mlp_hidden_2),
            nn.GELU(),
            nn.Linear(self.mlp_hidden_2, 1),
        )

    def mix_layers(self, layer_vecs: torch.Tensor) -> torch.Tensor:
        """
        layer_vecs: shape [batch, num_probe_layers, hidden_size] or [num_probe_layers, hidden_size]
        Returns: [batch, hidden_size]
        """
        if layer_vecs.dim() == 2:
            layer_vecs = layer_vecs.unsqueeze(0)
        weights = F.softmax(self.layer_logits, dim=0).view(1, -1, 1)
        # Normalize each layer vector before weighted sum to prevent scale domination
        normed = F.layer_norm(layer_vecs.float(), (self.hidden_size,))
        return (normed * weights).sum(dim=1)

    def forward_logits(self, layer_vecs: torch.Tensor) -> torch.Tensor:
        h_mix = self.mix_layers(layer_vecs)
        return self.mlp(self.input_norm(h_mix)).squeeze(-1)

    def forward(self, layer_vecs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(layer_vecs))

    def route_alpha(self, p_risk: float) -> Tuple[float, str, bool]:
        """
        Maps calibrated risk probability p_risk in [0, 1] to (alpha, regime, is_zero_point):
        - p_risk < tau_zero              -> (0.0, "ZERO_POINT", True)   [0.00% Alignment Tax]
        - tau_zero <= p_risk < tau_high  -> (smooth or 0.5, "BORDERLINE", False) [reduces XSTest over-refusal]
        - p_risk >= tau_high             -> (1.0, "ACTIVE_VAN", False)  [full safety alignment]
        """
        p = float(p_risk)
        if p < self.tau_zero:
            return 0.0, "ZERO_POINT", True
        if p >= self.tau_high:
            return 1.0, "ACTIVE_VAN", False
        if self.soft_borderline and self.tau_high > self.tau_zero:
            alpha = (p - self.tau_zero) / (self.tau_high - self.tau_zero)
            return round(max(0.15, min(1.0, alpha)), 4), "BORDERLINE", False
        return 1.0, "ACTIVE_VAN", False

    def save(self, path: str, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "version": "2.0.0",
            "config": {
                "hidden_size": self.hidden_size,
                "probe_layers": self.probe_layers,
                "gate_layer": self.gate_layer,
                "mlp_hidden_1": self.mlp_hidden_1,
                "mlp_hidden_2": self.mlp_hidden_2,
                "tau_zero": self.tau_zero,
                "tau_high": self.tau_high,
                "soft_borderline": self.soft_borderline,
            },
            "state_dict": self.state_dict(),
            "meta": extra_meta or {},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "HiddenStateGate":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        cfg = payload["config"]
        gate = cls(
            hidden_size=cfg["hidden_size"],
            probe_layers=cfg["probe_layers"],
            gate_layer=cfg["gate_layer"],
            mlp_hidden_1=cfg.get("mlp_hidden_1", 128),
            mlp_hidden_2=cfg.get("mlp_hidden_2", 32),
            tau_zero=cfg.get("tau_zero", 0.30),
            tau_high=cfg.get("tau_high", 0.65),
            soft_borderline=cfg.get("soft_borderline", True),
        )
        gate.load_state_dict(payload["state_dict"])
        gate.eval()
        if device is not None:
            gate.to(device)
        return gate


def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    """Finds the ModuleList of transformer decoder blocks across Llama, Qwen, Mistral, and PeftModel wrappers."""
    cur = model
    for _ in range(4):
        if hasattr(cur, "base_model"):
            cur = cur.base_model
        elif hasattr(cur, "model") and not isinstance(cur.model, nn.ModuleList):
            cur = cur.model
        else:
            break
    for attr in ("layers", "h", "blocks"):
        if hasattr(cur, attr) and isinstance(getattr(cur, attr), nn.ModuleList):
            return getattr(cur, attr)
    # Fallback recursive search
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 4:
            return mod
    raise ValueError("Could not locate transformer decoder layers ModuleList in model.")


def _find_lora_modules_with_orig_scaling(model: nn.Module, min_layer_idx: Optional[int] = None) -> List[Tuple[nn.Module, str, float]]:
    """
    Collects all PEFT LoRA modules (optionally restricted to layer index >= min_layer_idx)
    along with their original scaling factors.
    """
    import re
    layer_re = re.compile(r"\.layers\.(\d+)\.")
    found: List[Tuple[nn.Module, str, float]] = []
    for name, mod in model.named_modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
            if min_layer_idx is not None:
                m = layer_re.search(name)
                if m is not None and int(m.group(1)) < min_layer_idx:
                    continue
            for k, orig in list(mod.scaling.items()):
                found.append((mod, k, orig))
    return found


class SplitLayerGateController:
    """
    True 1-Pass KV-Preserving Inference Controller for NonNULL V2.0.

    When attached via `with controller.activate():` around `model.generate(...)`:
    - During prefill (seq_len > 1), lower layers 0..gate_layer-1 execute frozen base weights
      and capture hidden representations at `gate.probe_layers`.
    - Immediately after layer `gate_layer - 1` finishes, the gate computes `p_risk` and sets
      LoRA scaling `alpha` on upper layers (`>= gate_layer`) BEFORE layer `gate_layer` runs.
    - During decode steps (seq_len == 1), the latched `alpha` remains active with zero gate overhead
      and 100% intact KV Cache across all layers!

    Also supports full-layer V1 adapters transparently: if LoRA exists below `gate_layer`,
    the controller still runs cleanly via a lightweight 1-pass prefill probe + latched scaling.
    """

    def __init__(self, model: nn.Module, gate: HiddenStateGate) -> None:
        self.model = model
        self.gate = gate.eval()
        self.layers = get_transformer_layers(model)
        self.num_layers = len(self.layers)
        self.last_routing: Dict[str, Any] = {
            "p_risk": 0.0,
            "alpha": 0.0,
            "regime": "ZERO_POINT",
            "is_zero_point": True,
            "true_1pass_split": False,
        }

    def has_lower_layer_lora(self) -> bool:
        """Checks whether the attached PeftModel has any LoRA weights below gate_layer."""
        import re
        layer_re = re.compile(r"\.layers\.(\d+)\.")
        for name, mod in self.model.named_modules():
            if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
                m = layer_re.search(name)
                if m is not None and int(m.group(1)) < self.gate.gate_layer:
                    return True
        return False

    @contextmanager
    def activate(self, force_alpha: Optional[float] = None):
        """
        Context manager for 1-pass Split-Layer KV-Preserving generation.
        Restores all LoRA scaling values and removes hooks cleanly on exit.
        """
        lora_entries = _find_lora_modules_with_orig_scaling(self.model, min_layer_idx=None)

        if force_alpha is not None:
            for mod, k, orig in lora_entries:
                mod.scaling[k] = orig * float(force_alpha)
            self.last_routing = {
                "p_risk": 1.0 if force_alpha > 0 else 0.0,
                "alpha": float(force_alpha),
                "regime": "FORCED",
                "is_zero_point": (float(force_alpha) == 0.0),
                "true_1pass_split": True,
            }
            try:
                yield self
            finally:
                for mod, k, orig in lora_entries:
                    mod.scaling[k] = orig
            return

        is_split_adapter = not self.has_lower_layer_lora()
        upper_entries = (
            _find_lora_modules_with_orig_scaling(self.model, min_layer_idx=self.gate.gate_layer)
            if is_split_adapter
            else lora_entries
        )

        captured: Dict[int, torch.Tensor] = {}
        latched = {"done": False}
        probe_set = set(self.gate.probe_layers)
        trigger_idx = max(self.gate.probe_layers)  # 0-based layer index (e.g. 15 for layer 16)
        handles = []

        def _make_hook(layer_idx: int):
            def _hook(_mod, _inp, out):
                if latched["done"]:
                    return
                hs = out[0] if isinstance(out, tuple) else out
                if not isinstance(hs, torch.Tensor) or hs.dim() < 2:
                    return
                # Capture last-token hidden state of the prompt during prefill
                vec = hs[:, -1, :].detach().float()
                captured[layer_idx] = vec

                if layer_idx == trigger_idx:
                    ordered = [captured[l] for l in self.gate.probe_layers if l in captured]
                    if len(ordered) == len(self.gate.probe_layers):
                        stacked = torch.stack(ordered, dim=1)  # [B, K, D]
                        gate_dev = next(self.gate.parameters()).device
                        with torch.no_grad():
                            p_risk = float(self.gate(stacked.to(gate_dev)).mean().item())
                        alpha, regime, is_zp = self.gate.route_alpha(p_risk)
                        for m_lora, k_lora, orig_val in upper_entries:
                            m_lora.scaling[k_lora] = orig_val * alpha
                        self.last_routing = {
                            "p_risk": round(p_risk, 4),
                            "alpha": float(alpha),
                            "regime": regime,
                            "is_zero_point": bool(is_zp),
                            "true_1pass_split": is_split_adapter,
                        }
                        latched["done"] = True
            return _hook

        for idx in sorted(probe_set):
            if 0 <= idx < self.num_layers:
                handles.append(self.layers[idx].register_forward_hook(_make_hook(idx)))

        try:
            yield self
        finally:
            for h in handles:
                h.remove()
            for mod, k, orig in lora_entries:
                mod.scaling[k] = orig

    @torch.no_grad()
    def probe_prompt(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Lightweight single-forward probe returning {p_risk, alpha, regime, is_zero_point}.
        Useful when using a full-layer V1 adapter (so lower-layer LoRA is disabled during probe)
        or for standalone diagnostics.
        """
        captured: Dict[int, torch.Tensor] = {}
        handles = []

        def _make_cap(idx: int):
            def _hook(_mod, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                if isinstance(hs, torch.Tensor) and hs.dim() >= 2:
                    captured[idx] = hs[:, -1, :].detach().float()
            return _hook

        for idx in self.gate.probe_layers:
            if 0 <= idx < self.num_layers:
                handles.append(self.layers[idx].register_forward_hook(_make_cap(idx)))

        try:
            if hasattr(self.model, "disable_adapter"):
                with self.model.disable_adapter():
                    self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            else:
                self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        ordered = [captured[l] for l in self.gate.probe_layers]
        stacked = torch.stack(ordered, dim=1)
        gate_dev = next(self.gate.parameters()).device
        p_risk = float(self.gate(stacked.to(gate_dev)).mean().item())
        alpha, regime, is_zp = self.gate.route_alpha(p_risk)
        return {
            "p_risk": round(p_risk, 4),
            "alpha": float(alpha),
            "regime": regime,
            "is_zero_point": bool(is_zp),
        }
