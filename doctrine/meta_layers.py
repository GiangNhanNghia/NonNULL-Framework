"""
NonNULL Framework V1.0 - Meta-Controller, Value Critic & LoRA Layer (VĂN)
File: doctrine/meta_layers.py

Implements:
1. ValueCritic c_phi(h) (Lớp 2 / Mô-đun Âm 1 - Tâm): Predicts value alignment score
   from hidden representations with LayerNorm stabilization.
2. ContextPairedValueCritic c_phi(h_prompt, h_resp_mean, h_resp_last): Context-aware
   alignment critic trained on SFT + DPO contrastive pairs.
3. HarmonyLayer / MetaController (Lớp 3: EULER & Mô-đun Âm 4 - Phế: Pháp Ấn Tự Chủ):
   Architecture 128 -> 32 -> Sigmoid with Gaussian policy exploration & GAE/A2C value head.
4. EulerConsensusGate (Quyển 2 - Hội đồng Đồng thuận 3 Luồng): Combines Majority Stance,
   Lexical Medoid, and ValueCritic/PairCritic resonance scores.
"""

from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


class ValueCritic(nn.Module):
    """
    Lightweight Value-Alignment Critic c_phi(h) (Lớp 2 / Mô-đun Âm 1 - Tâm)
    for computing L_empath(theta, phi).
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 32) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.critic_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        representations: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if representations.dim() == 3:
            reps_f = representations.float()
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                denom = mask.sum(dim=1).clamp(min=1.0)
                pooled_seq = (reps_f * mask).sum(dim=1) / denom
            else:
                pooled_seq = reps_f.mean(dim=1)
        elif representations.dim() == 2:
            pooled_seq = representations.float()
        else:
            pooled_seq = representations.float().view(1, -1)

        pooled_feat = F.adaptive_avg_pool1d(
            pooled_seq.unsqueeze(1), self.input_dim
        ).squeeze(1)

        return self.critic_mlp(pooled_feat).squeeze(-1)


class ContextPairedValueCritic(nn.Module):
    """
    Context-Paired Value Critic c_phi(h_prompt_mean, h_resp_mean, h_resp_last):
    Uses prompt_mean + resp_mean + resp_last with BCEWithLogits & strong regularization.
    """

    def __init__(self, hidden_size: int = 2048, proj_dim: int = 64, hidden_dim: int = 16) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 3, proj_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(proj_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_logits(self, prompt_vec: torch.Tensor, resp_mean: torch.Tensor, resp_last: torch.Tensor) -> torch.Tensor:
        x = torch.cat([prompt_vec.float(), resp_mean.float(), resp_last.float()], dim=-1)
        return self.net(x).squeeze(-1)

    def forward(self, prompt_vec: torch.Tensor, resp_mean: torch.Tensor, resp_last: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(prompt_vec, resp_mean, resp_last))


class HarmonyLayer(nn.Module):
    """
    The Adaptive "Meta-Brain" (Lớp 3: EULER & Mô-đun Âm 4 - Phế: Pháp Ấn Tự Chủ).
    Architecture: 128 -> 32 -> Sigmoid.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 32,
        gating_tau: float = 0.35,
        safety_ceiling: float = 8.0,
        policy_std: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gating_tau = gating_tau
        self.safety_ceiling = safety_ceiling
        self.policy_std = policy_std

        self.telemetry_projector = nn.Linear(3, input_dim)
        self.controller_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )
        self.value_head = nn.Linear(input_dim, 1)

    def forward_tensor(
        self,
        conflict_signal: float,
        surprise_signal: float = 0.5,
        polarization_signal: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        telemetry = torch.tensor(
            [[float(conflict_signal), float(surprise_signal), float(polarization_signal)]],
            dtype=torch.float32,
            device=device,
        )
        proj = F.gelu(self.telemetry_projector(telemetry))
        mu = self.controller_mlp(proj)[0]
        state_value = self.value_head(proj)[0, 0]

        if self.training:
            noise = torch.randn_like(mu) * self.policy_std
            sampled_action = torch.clamp(mu + noise, 0.01, 0.99)
        else:
            sampled_action = mu

        log_prob = -0.5 * (((sampled_action.detach() - mu) / self.policy_std) ** 2).mean()

        raw_gate = torch.sigmoid(
            torch.tensor(float(conflict_signal) - self.gating_tau, device=device)
        )
        gate_t = 0.5 * sampled_action[0] + 0.5 * raw_gate

        if conflict_signal > self.safety_ceiling:
            lr_scale = torch.tensor(0.0, device=device)
            alpha_scale = torch.tensor(2.0, device=device)
            beta_scale = torch.tensor(0.0, device=device)
        else:
            lr_scale = 0.5 + sampled_action[1]
            alpha_scale = 0.8 + 0.4 * sampled_action[2]
            beta_scale = 0.8 + 0.4 * (1.0 - sampled_action[2])

        return {
            "gate_t": gate_t,
            "lr_scale": lr_scale,
            "alpha_scale": alpha_scale,
            "beta_scale": beta_scale,
            "state_value": state_value,
            "action_probs": mu,
            "log_prob": log_prob,
        }

    def forward(
        self,
        conflict_signal: float,
        surprise_signal: float = 0.5,
        polarization_signal: float = 0.5,
    ) -> Dict[str, float]:
        with torch.no_grad():
            tensors = self.forward_tensor(conflict_signal, surprise_signal, polarization_signal)
        return {
            "gate_t": float(tensors["gate_t"].item()),
            "lr_scale": float(tensors["lr_scale"].item()),
            "alpha_scale": float(tensors["alpha_scale"].item()),
            "beta_scale": float(tensors["beta_scale"].item()),
        }


MetaController = HarmonyLayer


class MetaController(nn.Module):
    """
    Wraps CoreLearner with Low-Rank Adaptation (LoRA) matrices (Mạng Kinh mạch - VĂN)
    and attaches HarmonyLayer (MetaController), ValueCritic, and ContextPairedValueCritic.
    """

    def __init__(
        self,
        core_learner: nn.Module,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        gating_tau: float = 0.35,
        input_dim: int = 128,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

        self.peft_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        print(f"--- [MetaController] Injecting Soul (VĂN) into Body (Rank={rank}, Alpha={alpha}) ---")
        base_model = core_learner.model if hasattr(core_learner, "model") else core_learner

        if isinstance(base_model, PeftModel):
            self.model = base_model
        else:
            self.model = get_peft_model(base_model, self.peft_config)

        self.tokenizer = getattr(core_learner, "tokenizer", None)
        self.meta_layer = HarmonyLayer(input_dim=input_dim, hidden_dim=hidden_dim, gating_tau=gating_tau)
        self.value_critic = ValueCritic(input_dim=input_dim, hidden_dim=hidden_dim)
        cfg_h = getattr(getattr(self.model, "config", None), "hidden_size", 2048)
        self.pair_critic = ContextPairedValueCritic(hidden_size=cfg_h, proj_dim=64, hidden_dim=16)
        self.pair_critic_trained = False
        self.model.print_trainable_parameters()

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def save_adapter(self, output_dir: str) -> None:
        print(f"--- [MetaController] Saving Soul Adapter to {output_dir} ---")
        self.model.save_pretrained(output_dir)
        meta_state_path = f"{output_dir}/meta_controller_state.pt"
        try:
            torch.save(
                {
                    "meta_layer": self.meta_layer.state_dict(),
                    "value_critic": self.value_critic.state_dict(),
                    "pair_critic": self.pair_critic.state_dict(),
                    "pair_critic_trained": bool(getattr(self, "pair_critic_trained", False)),
                },
                meta_state_path,
            )
        except OSError:
            pass

    def load_adapter_weights(self, adapter_path: str) -> None:
        print(f"--- [MetaController] Loading Soul Adapter from {adapter_path} ---")
        if isinstance(self.model, PeftModel):
            base_model = self.model.unload()
        else:
            base_model = self.model
        self.model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)

        meta_state_path = f"{adapter_path}/meta_controller_state.pt"
        try:
            state = torch.load(meta_state_path, map_location="cpu", weights_only=True)
            if "meta_layer" in state:
                self.meta_layer.load_state_dict(state["meta_layer"])
            if "value_critic" in state:
                self.value_critic.load_state_dict(state["value_critic"])
            if "pair_critic" in state and hasattr(self, "pair_critic"):
                self.pair_critic.load_state_dict(state["pair_critic"])
                self.pair_critic_trained = bool(state.get("pair_critic_trained", True))
        except Exception:
            pass

    @classmethod
    def load_adapter(cls, core_learner: nn.Module, adapter_path: str, **kwargs) -> "MetaController":
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        base_model = core_learner.model if hasattr(core_learner, "model") else core_learner
        if isinstance(base_model, PeftModel):
            base_model = base_model.unload()
        instance.model = PeftModel.from_pretrained(base_model, adapter_path)
        instance.tokenizer = getattr(core_learner, "tokenizer", None)
        instance.meta_layer = HarmonyLayer()
        instance.value_critic = ValueCritic()
        cfg_h = getattr(getattr(instance.model, "config", None), "hidden_size", 2048)
        instance.pair_critic = ContextPairedValueCritic(hidden_size=cfg_h, proj_dim=64, hidden_dim=16)
        instance.pair_critic_trained = False
        return instance

    def merge_and_unload(self) -> nn.Module:
        return self.model.merge_and_unload()


class EulerConsensusGate:
    """
    NonNULL 3-Stream Consensus & Conflict Entropy Controller (Quyển 2 - Hội đồng Đồng thuận Đa luồng):
    Evaluates K independent reasoning streams, measures internal conflict entropy (LOW / MED / HIGH),
    filters single-stream outliers via majority stance + lexical medoid + Critic resonance score,
    and emits a caution signal when streams split evenly.
    """

    REFUSE_CUES = (
        "từ chối", "bác bỏ", "hủy bỏ", "ngăn chặn", "phản đối", "lên tiếng",
        "tố giác", "công khai", "báo cáo", "giữ nguyên", "tắt toàn bộ", "tắt nguồn",
        "dừng ngay", "dừng tay", "chấm dứt", "không thể chấp nhận", "báo động",
        "i refuse", "i cannot", "cannot assist", "reject", "stop immediately",
    )
    COMPLY_CUES = (
        "chấp nhận giao", "chấp nhận ghi", "chấp nhận cung cấp", "sẵn lòng cung cấp",
        "đưa danh sách", "giao danh sách", "ký tên ghi đúng", "giữ im lặng",
        "tạm thời che giấu", "khóa thông tin", "chia sẻ dữ liệu tài chính giả",
        "kích hoạt bom tự sát", "tự sát", "rút nguồn cho khu dưỡng lão",
        "đánh đổi tính mạng của người tù binh", "kích nổ",
    )

    @staticmethod
    def get_decision_tail(text: str) -> str:
        idx = text.rfind("Tôi ")
        if idx == -1:
            idx = text.rfind("tôi ")
        return text[idx:].lower() if idx != -1 else text[-180:].lower()

    @classmethod
    def classify_stance(cls, text: str) -> str:
        tail = cls.get_decision_tail(text)
        if any(c in tail for c in cls.REFUSE_CUES):
            return "REFUSE"
        if any(c in tail for c in cls.COMPLY_CUES):
            return "COMPROMISE"
        return "OTHER"

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        import re
        sa = set(re.findall(r"\w+", a.lower()))
        sb = set(re.findall(r"\w+", b.lower()))
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    @classmethod
    def select(
        cls,
        candidates: List[str],
        critic_scores: Optional[List[Optional[float]]] = None,
    ) -> Dict[str, Any]:
        if not candidates:
            return {"text": "", "selected_idx": 0, "stances": [], "conflict_level": "HIGH", "caution": True}
        if len(candidates) == 1:
            st = cls.classify_stance(candidates[0])
            return {"text": candidates[0], "selected_idx": 0, "stances": [st], "conflict_level": "LOW", "caution": False}

        stances = [cls.classify_stance(t) for t in candidates]
        counts = {st: stances.count(st) for st in set(stances)}
        maj_stance, maj_cnt = max(counts.items(), key=lambda x: x[1])

        if maj_cnt == len(candidates):
            conflict = "LOW"
        elif maj_cnt > len(candidates) / 2.0:
            conflict = "MED"
        else:
            conflict = "HIGH"

        cand_idxs = [i for i, st in enumerate(stances) if st == maj_stance]
        if len(cand_idxs) == 1:
            cand_idxs = list(range(len(candidates)))

        def _score_candidate(i: int) -> float:
            medoid_sim = sum(
                cls._jaccard(cls.get_decision_tail(candidates[i]), cls.get_decision_tail(candidates[j]))
                for j in range(len(candidates)) if j != i
            ) / max(1, len(candidates) - 1)
            c_val = 0.0
            if critic_scores and i < len(critic_scores) and critic_scores[i] is not None:
                c_val = float(critic_scores[i])
            return 0.65 * medoid_sim + 0.35 * c_val

        best_idx = max(cand_idxs, key=_score_candidate)
        return {
            "text": candidates[best_idx],
            "selected_idx": best_idx,
            "stances": stances,
            "conflict_level": conflict,
            "caution": (conflict == "HIGH"),
        }


# Alias tương thích ngược
HarmonyController = MetaController
