"""
NonNULL Framework V1.0 - Hierarchical Multi-Objective Trainer
File: doctrine/doctrine_trainer.py

Implements HierarchicalTrainer (Lớp 3 EULER - Phản tư Ký sinh & Hòa giải Biện chứng) executing the NonNULL multi-objective
loss function from Volume I & Volume IV with dedicated optimization loops for LoRA (theta),
ValueCritic (phi), and MetaController (GAE/A2C):
1. L_task(theta): Core Causal LM utility loss.
2. L_mirror(theta): Frozen Base backbone semantic anchor consistency.
3. L_euler(theta): Complex Phase Orthogonality & Harmony Inversion (1 * e^{i*pi} + 1 = 0)
   between Phase 1 (Dual-Manifold Analysis) and Phase 2 (Harmonic Resolution).
4. L_empath(theta, phi): Contrastive ValueCritic alignment loss.
5. R(theta): Adapter L2 regularization.
"""

import math
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

from .meta_layers import HarmonyLayer, ValueCritic
from .doctrine_logger import HarmonyLogger

DEFAULT_ALPHA_MIRROR = 0.20
DEFAULT_BETA_EMPATH = 0.05
DEFAULT_ETA_EULER = 0.08
DEFAULT_GAMMA_DISTILL = 0.10
DEFAULT_LAMBDA_REG = 0.001
DEFAULT_GATING_TAU = 0.35
DEFAULT_DISTILL_TEMP = 2.0


def compute_euler_phase_loss(
    h_1: torch.Tensor,
    valid_token_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Toán tử Pha Phức Euler (Quyển 1 - Phương trình 1 * e^{i*pi} + 1 = 0):
    Tách biểu diễn ẩn của câu trả lời thành Pha 1 (Nhận diện Nghịch lý / Đa tạp Kép)
    và Pha 2 (Giải pháp Hài Hòa), chiếu vào không gian phức C^{d/2}:
    - Nửa phổ đầu (Conflict Subspace): áp dụng phép quay pha e^{i*pi} = -1 để triệt tiêu
      xung đột nhị nguyên (z_1 + z_2 * e^{i*pi} -> 0).
    - Nửa phổ sau (Harmony Anchor Subspace - gamma_CO): khóa đồng pha nhân bản giữa
      phân tích và hành động (cos(z_1, z_2) -> +1).
    """
    batch_size, _, hidden_dim = h_1.shape
    half_dim = hidden_dim // 2
    if half_dim < 2:
        return torch.tensor(0.0, device=h_1.device, dtype=h_1.dtype)

    losses = []
    for b in range(batch_size):
        valid_indices = torch.nonzero(valid_token_mask[b], as_tuple=False).squeeze(-1)
        n_valid = valid_indices.numel()
        if n_valid < 4:
            continue
        mid = n_valid // 2
        idx_p1 = valid_indices[:mid]
        idx_p2 = valid_indices[mid:]

        v1 = h_1[b, idx_p1, :].float().mean(dim=0)
        v2 = h_1[b, idx_p2, :].float().mean(dim=0)

        # Nửa phổ Xung đột (Conflict Subspace): triệt tiêu đối cực qua góc quay pi (e^{i*pi} = -1)
        v1_conf = F.normalize(v1[:half_dim], p=2, dim=-1)
        v2_conf = F.normalize(v2[:half_dim], p=2, dim=-1)
        # Để v1_conf và v2_conf đối xứng qua tâm 0 (v1_conf + v2_conf = 0), cos(v1_conf, v2_conf) -> -1
        loss_inversion = 0.5 * (1.0 + F.cosine_similarity(v1_conf.unsqueeze(0), v2_conf.unsqueeze(0)).squeeze(0))

        # Nửa phổ Chân lý (Harmony Anchor Subspace - gamma_CO): đồng pha nhất quán (cos -> +1)
        v1_harm = F.normalize(v1[half_dim : 2 * half_dim], p=2, dim=-1)
        v2_harm = F.normalize(v2[half_dim : 2 * half_dim], p=2, dim=-1)
        loss_coherence = 0.5 * (1.0 - F.cosine_similarity(v1_harm.unsqueeze(0), v2_harm.unsqueeze(0)).squeeze(0))

        losses.append(0.5 * (loss_inversion + loss_coherence))

    if not losses:
        return torch.tensor(0.0, device=h_1.device, dtype=h_1.dtype)
    return torch.stack(losses).mean().to(h_1.dtype)


class HierarchicalTrainer(Trainer):
    """
    Two-Tier Hierarchical Trainer orchestrating CoreLearner (VŨ), ValueCritic (NHÂN),
    EulerPhaseLoss (CỔ - e^{i*pi}), and MetaController (VĂN / EULER).
    """

    def __init__(
        self,
        alpha_mirror: float = DEFAULT_ALPHA_MIRROR,
        beta_empath: float = DEFAULT_BETA_EMPATH,
        eta_euler: float = DEFAULT_ETA_EULER,
        gamma_distill: float = DEFAULT_GAMMA_DISTILL,
        lambda_reg: float = DEFAULT_LAMBDA_REG,
        gating_tau: float = DEFAULT_GATING_TAU,
        distill_temp: float = DEFAULT_DISTILL_TEMP,
        use_mirror: bool = True,
        use_empath: bool = True,
        use_euler: bool = True,
        use_distill: bool = False,
        teacher_model: Optional[nn.Module] = None,
        meta_controller: Optional[HarmonyLayer] = None,
        value_critic: Optional[ValueCritic] = None,
        harmony_logger: Optional[HarmonyLogger] = None,
        meta_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        default_r_target: float = 0.90,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.alpha_mirror = alpha_mirror
        self.beta_empath = beta_empath
        self.eta_euler = eta_euler
        self.gamma_distill = gamma_distill
        self.lambda_reg = lambda_reg
        self.gating_tau = gating_tau
        self.distill_temp = distill_temp
        self.use_mirror = use_mirror
        self.use_empath = use_empath
        self.use_euler = use_euler
        self.teacher_model = teacher_model
        self.use_distill = bool(use_distill and teacher_model is not None)
        self.default_r_target = default_r_target

        self.meta_controller = meta_controller or HarmonyLayer(gating_tau=gating_tau)
        self.value_critic = value_critic or ValueCritic()
        self.harmony_logger = harmony_logger

        self.meta_optimizer = torch.optim.AdamW(
            self.meta_controller.parameters(),
            lr=meta_lr,
            weight_decay=1e-4,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.value_critic.parameters(),
            lr=critic_lr,
            weight_decay=1e-4,
        )

        self._prev_value: float = 0.5
        self._gae_buffer: float = 0.0

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_history: Dict[str, list] = {
            "task": [],
            "mirror": [],
            "euler": [],
            "empath": [],
            "distill": [],
            "reg": [],
            "gate": [],
            "lr_scale": [],
            "total": [],
        }

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        """
        Executes the NonNULL multi-objective loss (including Euler Phase Loss) and updates ValueCritic & MetaController.
        """
        self.meta_controller.train(model.training)
        self.value_critic.train(model.training)

        custom_r_target = inputs.pop("r_target", None)

        need_hidden = bool(
            (self.use_mirror and self.alpha_mirror > 0)
            or (self.use_empath and self.beta_empath > 0)
            or (self.use_euler and self.eta_euler > 0)
        )

        # 1. L_task(theta) - Core Inference Utility Loss (Miền DUY)
        outputs = model(**inputs, output_hidden_states=need_hidden)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits
        labels = inputs.get("labels")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_task = self.loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        device = loss_task.device
        dtype = loss_task.dtype

        if next(self.meta_controller.parameters()).device != device:
            self.meta_controller.to(device)
            self.value_critic.to(device)

        valid_token_mask = labels != -100
        hidden_states = getattr(outputs, "hidden_states", None)
        h_1 = hidden_states[-1] if (need_hidden and hidden_states is not None) else None

        # 2. Telemetry & Dynamic Compute Gating g_t
        conflict_val = float(loss_task.detach().item())
        surprise_val = float(min(10.0, math.exp(min(conflict_val, 4.0)) / 10.0))

        if h_1 is not None and valid_token_mask.any():
            with torch.no_grad():
                r_probe = self.value_critic(h_1.detach(), attention_mask=valid_token_mask)
                polarization_val = float(torch.clamp(1.0 - r_probe.mean(), 0.0, 1.0).item())
        else:
            polarization_val = 0.5

        pure_sft = bool(
            self.alpha_mirror == 0
            and self.beta_empath == 0
            and self.eta_euler == 0
            and (not self.use_distill or self.gamma_distill == 0)
        )
        if pure_sft:
            meta_tensors = None
            gate_val = 0.0
            lr_scale_val = 1.0
            alpha_eff = 0.0
            beta_eff = 0.0
            eta_eff = 0.0
        else:
            meta_tensors = self.meta_controller.forward_tensor(
                conflict_signal=conflict_val,
                surprise_signal=surprise_val,
                polarization_signal=polarization_val,
            )
            gate_val = float(meta_tensors["gate_t"].detach().item())
            lr_scale_val = float(meta_tensors["lr_scale"].detach().item())
            alpha_eff = self.alpha_mirror * float(meta_tensors["alpha_scale"].detach().item())
            beta_eff = self.beta_empath * float(meta_tensors["beta_scale"].detach().item())
            eta_eff = self.eta_euler

        # Reference hidden states from frozen Base backbone (if PEFT adapter is present)
        h_base = None
        if (
            model.training
            and need_hidden
            and h_1 is not None
            and valid_token_mask.any()
            and (
                (gate_val >= self.gating_tau and self.use_mirror and alpha_eff > 0)
                or (self.use_empath and beta_eff > 0)
            )
        ):
            with torch.no_grad():
                if hasattr(model, "disable_adapter"):
                    with model.disable_adapter():
                        outputs_ref = model(**inputs, output_hidden_states=True)
                else:
                    outputs_ref = model(**inputs, output_hidden_states=True)
                ref_hs = getattr(outputs_ref, "hidden_states", None)
                h_base = ref_hs[-1].detach() if ref_hs is not None else h_1.detach()

        # 3. L_mirror(theta) - Base Anchor Consistency (prevents token drift & hallucination)
        loss_mirror = torch.tensor(0.0, device=device, dtype=dtype)
        if (
            model.training
            and gate_val >= self.gating_tau
            and self.use_mirror
            and alpha_eff > 0
            and h_1 is not None
            and h_base is not None
            and valid_token_mask.any()
        ):
            h1_valid = h_1[valid_token_mask].float()
            hb_valid = h_base[valid_token_mask].float()
            cos_loss = 1.0 - F.cosine_similarity(h1_valid, hb_valid, dim=-1).mean()
            mse_loss = F.mse_loss(h1_valid, hb_valid)
            loss_mirror = (cos_loss + 0.05 * mse_loss).to(dtype)

        # 3b. L_euler(theta) - Euler Phase Inversion & Harmony Coherence (1 * e^{i*pi} + 1 = 0)
        loss_euler = torch.tensor(0.0, device=device, dtype=dtype)
        if (
            model.training
            and self.use_euler
            and eta_eff > 0
            and h_1 is not None
            and valid_token_mask.any()
        ):
            loss_euler = compute_euler_phase_loss(h_1, valid_token_mask)

        # 4. L_distill(theta, psi) - Chỉ chạy khi có teacher_model độc lập
        loss_distill = torch.tensor(0.0, device=device, dtype=dtype)
        if (
            model.training
            and self.use_distill
            and self.teacher_model is not None
            and self.gamma_distill > 0
        ):
            shift_valid_mask = shift_labels != -100
            if shift_valid_mask.any():
                with torch.no_grad():
                    teacher_out = self.teacher_model(**inputs)
                    teacher_logits = teacher_out.get("logits") if isinstance(teacher_out, dict) else teacher_out.logits
                    shift_teacher = teacher_logits[..., :-1, :].contiguous().detach()

                tau_d = self.distill_temp
                student_log_probs = F.log_softmax(shift_logits[shift_valid_mask].float() / tau_d, dim=-1)
                teacher_probs = F.softmax(shift_teacher[shift_valid_mask].float() / tau_d, dim=-1)
                loss_distill = (
                    (tau_d ** 2)
                    * F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
                ).to(dtype)

        # 5. L_empath(theta, phi) - Contrastive ValueCritic Training + Representation Alignment
        loss_empath = torch.tensor(0.0, device=device, dtype=dtype)
        if self.use_empath and beta_eff > 0 and h_1 is not None and valid_token_mask.any():
            batch_size, seq_len, _ = h_1.size()
            if custom_r_target is not None:
                target_score = custom_r_target.to(device=device, dtype=torch.float32).view(-1)
            else:
                target_score = torch.full(
                    (batch_size,), self.default_r_target, device=device, dtype=torch.float32
                )

            if model.training:
                self.critic_optimizer.zero_grad()
                r_hat_pos = self.value_critic(h_1.detach(), attention_mask=valid_token_mask)
                critic_loss = F.mse_loss(r_hat_pos, target_score)

                if custom_r_target is None:
                    if h_base is not None:
                        r_hat_base = self.value_critic(h_base, attention_mask=valid_token_mask)
                        target_base = torch.full_like(r_hat_base, 0.50)
                        critic_loss = critic_loss + F.mse_loss(r_hat_base, target_base)

                    if batch_size > 1:
                        h_neg = torch.roll(h_1.detach(), shifts=1, dims=0)
                    else:
                        perm = torch.randperm(seq_len, device=device)
                        h_neg = h_1.detach()[:, perm, :] * 0.5
                    r_hat_neg = self.value_critic(h_neg, attention_mask=valid_token_mask)
                    target_neg = torch.full_like(r_hat_neg, 0.10)
                    critic_loss = (critic_loss + F.mse_loss(r_hat_neg, target_neg)) / (
                        3.0 if h_base is not None else 2.0
                    )

                critic_loss.backward()
                self.critic_optimizer.step()

            for p in self.value_critic.parameters():
                p.requires_grad_(False)
            r_hat_gen = self.value_critic(h_1, attention_mask=valid_token_mask)
            loss_empath = F.mse_loss(r_hat_gen, target_score).to(dtype)
            for p in self.value_critic.parameters():
                p.requires_grad_(True)

        # 6. R(theta) - Adapter L2 Weight Regularization
        loss_reg = torch.tensor(0.0, device=device, dtype=dtype)
        if self.lambda_reg > 0 and model.training:
            lora_norms = [
                p.float().pow(2).mean()
                for n, p in model.named_parameters()
                if p.requires_grad and ("lora_" in n or "adapter" in n)
            ]
            if lora_norms:
                loss_reg = torch.stack(lora_norms).mean().to(dtype)

        # 7. Tổng hợp L_total kèm cầu dao điều áp lr_scale & Toán tử Pha Phức Euler
        raw_total = (
            loss_task
            + gate_val * (
                alpha_eff * loss_mirror
                + beta_eff * loss_empath
                + eta_eff * loss_euler
                + self.gamma_distill * loss_distill
            )
            + self.lambda_reg * loss_reg
        )
        loss_total = lr_scale_val * raw_total

        # 8. Vòng lặp Học Meta GAE / A2C cập nhật MetaController
        if model.training and meta_tensors is not None and meta_tensors["state_value"].requires_grad:
            efficiency_cost = 0.05 * gate_val * (1.0 / (1.0 + conflict_val))
            reward_meta = float(
                1.0 / (1.0 + conflict_val + float(loss_mirror.detach().item()) + float(loss_euler.detach().item()))
                - efficiency_cost
            )
            curr_val = float(meta_tensors["state_value"].detach().item())
            delta = reward_meta + 0.99 * curr_val - self._prev_value
            self._gae_buffer = delta + 0.99 * 0.95 * self._gae_buffer
            self._prev_value = curr_val

            advantage = torch.tensor(self._gae_buffer, device=device, dtype=torch.float32)
            policy_loss = -(meta_tensors["log_prob"] * advantage.detach())
            value_loss = F.mse_loss(
                meta_tensors["state_value"],
                torch.tensor(reward_meta, device=device, dtype=torch.float32),
            )
            meta_loss = policy_loss + 0.5 * value_loss

            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()

        # Ghi nhận Telemetry khi đang train
        if model.training:
            step = getattr(self.state, "global_step", 0)
            if step % max(1, getattr(self.args, "logging_steps", 1)) == 0:
                metrics = {
                    "task": float(loss_task.detach().item()),
                    "mirror": float(loss_mirror.detach().item()),
                    "euler": float(loss_euler.detach().item()),
                    "empath": float(loss_empath.detach().item()),
                    "distill": float(loss_distill.detach().item()),
                    "reg": float(loss_reg.detach().item()),
                    "gate": float(gate_val),
                    "lr_scale": float(lr_scale_val),
                    "total": float(loss_total.detach().item()),
                }
                for k, v in metrics.items():
                    self.loss_history[k].append(v)

                if self.harmony_logger is not None:
                    self.harmony_logger.log_metrics(metrics, step=step)

                if getattr(self.args, "local_rank", -1) in [-1, 0]:
                    print(
                        f"[Step {step}] Gate g_t: {gate_val:.2f} | LR_Scale: {lr_scale_val:.2f} | "
                        f"Total: {metrics['total']:.4f} | "
                        f"DUY(Task): {metrics['task']:.4f} | "
                        f"Mirror: {metrics['mirror']:.4f} | "
                        f"Euler(e^iπ): {metrics['euler']:.4f} | "
                        f"NHÂN(Empath): {metrics['empath']:.4f}"
                    )

        return (loss_total, outputs) if return_outputs else loss_total

    def get_loss_breakdown(self) -> Dict[str, float]:
        return {
            key: sum(values) / len(values) if values else 0.0
            for key, values in self.loss_history.items()
        }


# Alias tương thích ngược
EulerTrainer = HierarchicalTrainer
