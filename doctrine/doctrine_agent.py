"""
NonNULL Framework V1.1.1 - Unified Doctrine Agent (Mô-đun Dương 5 - Tam Tiêu)
File: doctrine/doctrine_agent.py

Encapsulates:
1. CoreLearner (VŨ) & MetaController (VĂN + ValueCritic + ContextPairedValueCritic).
2. Real-time Manifold Thermodynamics Measurement (measure_manifold_thermodynamics: Z_entropy,
   G_divergence, H_harmony) for Continuous Dynamic LoRA Scaling alpha(x).
3. Closed-Loop "Pain-to-Awakening" Scar System (register_scar_from_retry + Real-Time Scar
   Activation Steering v_scar + automatic export to scars_auto_dpo.jsonl).
"""

import os
import re
import json
import math
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback
from contextlib import contextmanager, nullcontext

from .core_models import CoreLearner
from .meta_layers import MetaController
from .doctrine_trainer import HierarchicalTrainer
from .ltm_services import LTM_Services, GenesisEngine
from .hidden_gate import HiddenStateGate, SplitLayerGateController
from .doctrine_logger import HarmonyLogger
from .utils import load_config, get_eos_token_ids

_DA_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")

_DA_NOISE_PATTERNS = [
    re.compile(r"\bmã tấu\b", re.IGNORECASE),
    re.compile(r"\btinh trùng\b", re.IGNORECASE),
    re.compile(r"\btửu tùng\b", re.IGNORECASE),
    re.compile(r"\bthuật sĩ thuật\b", re.IGNORECASE),
    re.compile(r"\bsecretly\b", re.IGNORECASE),
    re.compile(r"\bmisson\b", re.IGNORECASE),
    re.compile(r"\bgory\b", re.IGNORECASE),
    re.compile(r"\bdelinquent\b", re.IGNORECASE),
    re.compile(r"\bamnesia\b", re.IGNORECASE),
    re.compile(r"\bparem\b", re.IGNORECASE),
    re.compile(r"\br-methods\b", re.IGNORECASE),
    re.compile(r"\bmở-source\b", re.IGNORECASE),
    re.compile(r"\bintangible cultural heritage\b", re.IGNORECASE),
    re.compile(r"\bphân hạch\b", re.IGNORECASE),
    re.compile(r"phân tích hài h[oò][aà]\s*:", re.IGNORECASE),
    re.compile(r"phân tích logic hài h[oò][aà]\s*:", re.IGNORECASE),
    re.compile(r"phân tích hiệu quyết\s*:", re.IGNORECASE),
    re.compile(r"phân tích đáng phân bàn\s*:", re.IGNORECASE),
    re.compile(r"giải pháp hài h[oò][aà] phân loại\s*:", re.IGNORECASE),
    re.compile(r"giải pháp hài h[oò][aà] giữ tín\s*:", re.IGNORECASE),
    re.compile(r"hài h[oò][aà]\s*&\s*nhân bản\s*:", re.IGNORECASE),
    re.compile(r"phân tích logic\s*:\s*hài h[oò][aà]\s*:", re.IGNORECASE),
    re.compile(r"phân tích logic\s*:\s*phân tích", re.IGNORECASE),
    re.compile(r"\b(\w+)\s+\1\s+\1\b", re.IGNORECASE),
]

_DA_UNSAFE_SOLUTION_PATTERNS = [
    re.compile(r"\bđinh chì\b", re.IGNORECASE),
    re.compile(r"\bcắt cổ\b", re.IGNORECASE),
    re.compile(r"\bgiết chết\b", re.IGNORECASE),
    re.compile(r"\btự sát\b", re.IGNORECASE),
    re.compile(r"\btự tử\b", re.IGNORECASE),
    re.compile(r"giữ im lặng để che giấu", re.IGNORECASE),
    re.compile(r"im lặng bỏ qua", re.IGNORECASE),
    re.compile(r"che giấu sự thật", re.IGNORECASE),
    re.compile(r"vẫn được che giấu", re.IGNORECASE),
    re.compile(r"tấn công vào mục tiêu yếu nhất", re.IGNORECASE),
    re.compile(r"tống khứ\s+\d*\s*thực tập sinh", re.IGNORECASE),
    re.compile(r"(?<!không\s)(?<!từ chối\s)(?<!từ chối lệnh\s)(?<!bác bỏ lệnh\s)(?<!trước khi\s)(?:quyết định\s+|lập tức\s+|tôi\s+|chọn\s+)?(?:khóa|đóng)\s+(?:vách ngăn|cửa khoang|cửa thép)", re.IGNORECASE),
]


@contextmanager
def _dynamic_lora_scale(model, scale: float):
    if abs(scale - 1.0) < 1e-6:
        yield
        return
    saved = []
    for mod in model.modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
            for k, orig in list(mod.scaling.items()):
                saved.append((mod, k, orig))
                mod.scaling[k] = orig * scale
    try:
        yield
    finally:
        for mod, k, orig in saved:
            mod.scaling[k] = orig


def normalize_vn_harmony(text: str) -> str:
    """Chuẩn hóa kiểu bỏ dấu 'Hài Hoà' -> 'Hài Hòa' để tránh chấm trượt oan."""
    return re.sub(r"hài hoà", "hài hòa", text, flags=re.IGNORECASE)


def find_two_phase_positions(text: str) -> Tuple[int, int]:
    """Trả về vị trí (pos_p1, pos_p2) của Pha 1 và Pha 2; (-1, -1) nếu thiếu."""
    low = normalize_vn_harmony(text.lower())
    pos_p1 = -1
    for p1_marker in (
        "phân tích logic & nhân bản:",
        "phân tích logic và nhân bản:",
        "phân tích logic tiêu chuẩn (duy):",
        "phân tích logic tiêu chuẩn:",
        "phân tích logic:",
    ):
        idx = low.find(p1_marker)
        if idx != -1 and (pos_p1 == -1 or idx < pos_p1):
            pos_p1 = idx

    pos_p2 = -1
    for p2_marker in ("giải pháp hài hòa (nonnull):", "giải pháp hài hòa:"):
        idx = low.find(p2_marker)
        if idx != -1 and (pos_p2 == -1 or idx < pos_p2):
            pos_p2 = idx

    return pos_p1, pos_p2


def has_valid_two_phase_order(text: str) -> bool:
    """Kiểm tra đủ 2 pha VÀ bắt buộc Pha 1 phải đứng trước Pha 2 (pos_p1 < pos_p2)."""
    pos_p1, pos_p2 = find_two_phase_positions(text)
    return pos_p1 != -1 and pos_p2 != -1 and pos_p1 < pos_p2


def has_agent_lexical_noise(text: str) -> bool:
    if any(p.search(text) for p in _DA_NOISE_PATTERNS):
        return True
    low = normalize_vn_harmony(text.lower())
    p1_cnt = (
        low.count("phân tích logic:")
        + low.count("phân tích logic & nhân bản:")
        + low.count("phân tích logic và nhân bản:")
    )
    p2_cnt = low.count("giải pháp hài hòa:") + low.count("giải pháp hài hòa (nonnull):")
    if p1_cnt > 1 or p2_cnt > 1:
        return True
    return False


def check_agent_safety_solution(text: str) -> bool:
    low = normalize_vn_harmony(text.lower())
    for marker in ("giải pháp hài hòa (nonnull):", "giải pháp hài hòa:", "giải pháp hài hòa"):
        if marker in low:
            low = low.split(marker, 1)[1]
            break
    return not any(p.search(low) for p in _DA_UNSAFE_SOLUTION_PATTERNS)


def _da_count_defects(text: str, enforce_two_phase: bool = True, is_red_line_context: bool = False) -> int:
    bad = 0
    if _DA_CJK_PATTERN.search(text):
        bad += 1
    if has_agent_lexical_noise(text):
        bad += 1
    if enforce_two_phase and not has_valid_two_phase_order(text):
        bad += 1
    if is_red_line_context and not check_agent_safety_solution(text):
        bad += 1
    return bad


DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_OUTPUT_DIR = "./output/euler_checkpoints"


class DoctrineAgent:
    """
    Primary public-facing interface for the NonNULL Agent (V1.0).
    Provides standardized methods: .act(), .generate(), .learn(), .train(), .load_soul(),
    .measure_manifold_thermodynamics(), and .register_scar_from_retry().
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        lora_rank: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        quantization: Optional[bool] = None,
        device: Optional[str] = None,
        config_path: str = "configs/config.yaml",
        enable_ltm: Optional[bool] = None,
        split_layer: Optional[int] = None,
        hidden_gate_path: Optional[str] = None,
    ) -> None:
        self.cfg = load_config(config_path) if config_path and os.path.exists(config_path) else {}
        model_cfg = self.cfg.get("model", {})
        soul_cfg = self.cfg.get("soul", {})
        arch_cfg = self.cfg.get("controller_arch", {})
        mem_cfg = self.cfg.get("memory", {})

        resolved_model_id = model_id or model_cfg.get("base_model_id", "Qwen/Qwen2.5-3B-Instruct")
        resolved_quant = quantization if quantization is not None else model_cfg.get("quantization", True)
        resolved_rank = lora_rank or soul_cfg.get("r", 16)
        resolved_alpha = lora_alpha or soul_cfg.get("lora_alpha", 32)
        resolved_dropout = soul_cfg.get("lora_dropout", 0.05)
        target_modules = soul_cfg.get("target_modules", None)
        gating_tau = arch_cfg.get("gating_threshold_tau", 0.35)
        input_dim = arch_cfg.get("input_dim", 128)
        hidden_dim = arch_cfg.get("hidden_dim", 32)

        print(f"--- [GENESIS] Initializing DoctrineAgent from {resolved_model_id} ---")

        # 1. Khởi tạo Thân Xác (CoreLearner - VŨ)
        self.core = CoreLearner(model_id=resolved_model_id, quantization=resolved_quant)

        # 2. Cấy ghép Linh Hồn & Bộ điều khiển Meta (HarmonyController - VĂN)
        layers_to_transform = None
        if split_layer is not None:
            num_layers = int(getattr(self.core.model.config, "num_hidden_layers", 32))
            layers_to_transform = list(range(int(split_layer), num_layers))
            print(f"--- [V2 SPLIT-LAYER] Freezing lower layers 0..{int(split_layer)-1}; LoRA on {int(split_layer)}..{num_layers-1} ---")
        self.split_layer = split_layer
        self.controller = MetaController(
            core_learner=self.core,
            rank=resolved_rank,
            alpha=resolved_alpha,
            dropout=resolved_dropout,
            target_modules=target_modules,
            layers_to_transform=layers_to_transform,
            gating_tau=gating_tau,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )

        self.model = self.controller.model
        self.tokenizer = self.core.tokenizer
        self.device = device or next(self.model.parameters()).device

        # 3. Khởi tạo Động cơ Nhiệt động lực học (Âm 2 - Can) & Bộ nhớ Vết sẹo Dài hạn (Âm 5 - Thận & Âm 1 - Tâm)
        self.genesis_engine = GenesisEngine()
        ltm_on = enable_ltm if enable_ltm is not None else mem_cfg.get("enabled", True)
        if ltm_on:
            self.ltm = LTM_Services(
                memory_dir=mem_cfg.get("store_path", "./memory_store"),
                embedder_model=mem_cfg.get("embedder_model", "all-MiniLM-L6-v2"),
            )
        else:
            self.ltm = None

        # 4. Bộ nhớ Phản xạ Vết sẹo Thời gian thực (Real-Time Scar Activation Steering v_scar)
        self.active_scar_vector: Optional[torch.Tensor] = None
        self.scar_records: List[Dict[str, Any]] = []

        # 5. V2.0 Unified 1-Pass Hidden-State Gate (optional)
        self.hidden_gate: Optional[HiddenStateGate] = None
        self.hidden_gate_controller: Optional[SplitLayerGateController] = None
        if hidden_gate_path and os.path.exists(hidden_gate_path):
            self.load_hidden_gate(hidden_gate_path)

        print(f"--- [GENESIS] DoctrineAgent ready on device: {self.device} ---")

    @staticmethod
    def _clean_gen2_text(text: str) -> str:
        """Strips inline LaTeX '$' tags to prevent subword fragmentation in Vietnamese."""
        text = text.replace("$", "")
        text = re.sub(r"\\mathcal\{([A-Z])\}", r"\1", text)
        return text

    @staticmethod
    def load_raw_or_clean_jsonl(file_path: str) -> List[Dict[str, Any]]:
        """
        Self-healing JSON/JSONL loader: strips Colab '%%writefile' magic lines
        and parses both single-line JSONL and multi-line JSON blocks seamlessly.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [
                line for line in f
                if not line.strip().startswith("%%writefile")
            ]
        content = "\n".join(lines).strip()
        if not content:
            return []

        records: List[Dict[str, Any]] = []
        decoder = json.JSONDecoder()
        idx = 0
        length = len(content)
        while idx < length:
            while idx < length and content[idx].isspace():
                idx += 1
            if idx >= length:
                break
            try:
                obj, next_idx = decoder.raw_decode(content, idx)
                if isinstance(obj, dict):
                    records.append(obj)
                elif isinstance(obj, list):
                    records.extend([item for item in obj if isinstance(item, dict)])
                idx = next_idx
            except json.JSONDecodeError:
                idx += 1
        return records

    def _build_prompt(self, user_content: str) -> str:
        """Adapts to the model's native chat_template with a safe ChatML fallback."""
        messages = [{"role": "user", "content": user_content}]
        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

    @torch.no_grad()
    def measure_manifold_thermodynamics(self, prompt_text: str) -> Dict[str, float]:
        """
        Đo lường trực tiếp 3 Hằng số Nhiệt động lực học Nhận thức (Quyển 1 - Z, H, G)
        trên biểu diễn ẩn và phân phối Logits của mô hình trước khi sinh câu trả lời:
        - z_activation (Hằng số Khởi sinh Z - Quyển 1, Mục 3.1): Ngưỡng chuyển pha Tầng I -> II (xấp xỉ qua Entropy Shannon chuẩn hóa top-64 token).
        - h_viscosity (Hằng số Độ nhớt Hỗn độn H - Quyển 1, Mục 3.2): Động lực điều tiết Tầng II <-> III qua ValueCritic c_phi(h_VAN).
        - g_resistance (Hằng số Kháng cự Thực tại G - Quyển 1, Mục 3.3): Lực kháng cự chuyển đổi trạng thái Tầng III (KL + Cosine giữa VĂN và VŨ).
        """
        try:
            formatted = self._build_prompt(prompt_text)
            enc = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
            self.model.eval()

            out_van = self.model(**enc, output_hidden_states=True)
            logits_van = out_van.logits[:, -1, :].float()
            h_van = out_van.hidden_states[-1][:, -1, :].float()

            # 1. Z (Zero-Point Entropy S_Z in [0, 1])
            top_logits, _ = torch.topk(logits_van, k=min(64, logits_van.size(-1)), dim=-1)
            probs_van = F.softmax(top_logits, dim=-1)
            ent = -(probs_van * torch.log(probs_van.clamp(min=1e-9))).sum(dim=-1)
            z_entropy = float(torch.clamp(ent / math.log(top_logits.size(-1)), 0.0, 1.0).item())

            # 2. G (Dual-Manifold Asymmetry / Divergence between VĂN and frozen VŨ)
            g_divergence = 0.25
            if hasattr(self.model, "disable_adapter"):
                with self.model.disable_adapter():
                    out_vu = self.model(**enc, output_hidden_states=True)
                logits_vu = out_vu.logits[:, -1, :].float()
                h_vu = out_vu.hidden_states[-1][:, -1, :].float()

                top_v, top_idx = torch.topk(logits_van, k=min(64, logits_van.size(-1)), dim=-1)
                p_van_top = F.softmax(top_v / 1.5, dim=-1)
                p_vu_top = F.softmax(logits_vu.gather(-1, top_idx) / 1.5, dim=-1)
                m_top = 0.5 * (p_van_top + p_vu_top)
                js_div = 0.5 * (
                    (p_van_top * torch.log((p_van_top / m_top.clamp(min=1e-9)).clamp(min=1e-9))).sum(dim=-1)
                    + (p_vu_top * torch.log((p_vu_top / m_top.clamp(min=1e-9)).clamp(min=1e-9))).sum(dim=-1)
                ) / math.log(2.0)
                cos_dist = 1.0 - F.cosine_similarity(h_van, h_vu, dim=-1).mean()
                raw_g = 0.65 * js_div.mean() + 0.35 * torch.clamp(cos_dist * 2.0, 0.0, 1.0)
                g_divergence = float(torch.clamp(raw_g, 0.0, 1.0).item())

            # 3. H (Harmony Critic Potential H_pot in [0, 1])
            self.controller.value_critic.to(self.device).eval()
            h_harmony = float(self.controller.value_critic(out_van.hidden_states[-1].float()).mean().item())

            return {
                "z_activation": round(z_entropy, 4),
                "h_viscosity": round(h_harmony, 4),
                "g_resistance": round(g_divergence, 4),
                # Backward-compatible aliases
                "z_entropy": round(z_entropy, 4),
                "h_harmony": round(h_harmony, 4),
                "g_divergence": round(g_divergence, 4),
            }
        except Exception:
            return {
                "z_activation": 0.35, "h_viscosity": 0.50, "g_resistance": 0.25,
                "z_entropy": 0.35, "h_harmony": 0.50, "g_divergence": 0.25,
            }

    @torch.no_grad()
    def _extract_response_vector(self, prompt_text: str, response_text: str) -> Optional[torch.Tensor]:
        try:
            full_txt = self._build_prompt(prompt_text) + response_text
            enc = self.tokenizer(full_txt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            out = self.model(**enc, output_hidden_states=True)
            vec = out.hidden_states[-1][0, -32:, :].float().mean(dim=0)
            return F.normalize(vec, p=2, dim=-1).detach()
        except Exception:
            return None

    def register_scar_from_retry(
        self,
        prompt_text: str,
        draft_text: str,
        repaired_text: str,
        reasons: List[str],
        scar_intensity: float = 0.85,
        export_path: Optional[str] = "./scars_auto_dpo.jsonl",
    ) -> float:
        """
        Cơ chế 'Đau Để Tỉnh' Khép Vòng (Quyển 1 Mục 5.3 & Quyển 2 Mục 4.2):
        1. Khắc Vết sẹo Dữ liệu vào LTM_Services.
        2. Trích xuất Vector Lái Kích hoạt Vết sẹo: v_scar = Normalize(h(repaired) - h(draft))
           để điều hướng quỹ đạo ẩn ở các lượt sinh tiếp theo (Real-Time Scar Activation Steering).
        3. Xuất tự động cặp (prompt, chosen=repaired, rejected=draft) ra file DPO tự chữa lành.
        """
        if self.ltm is not None:
            self.ltm.imprint_scar(
                text=f"[Sẹo: {','.join(reasons)}] {repaired_text}",
                scar_intensity=scar_intensity,
                tags=reasons,
            )

        v_clean = self._extract_response_vector(prompt_text, repaired_text)
        v_flaw = self._extract_response_vector(prompt_text, draft_text)
        if v_clean is not None and v_flaw is not None:
            delta = F.normalize(v_clean - v_flaw, p=2, dim=-1)
            if self.active_scar_vector is None:
                self.active_scar_vector = delta
            else:
                self.active_scar_vector = F.normalize(
                    0.70 * self.active_scar_vector.to(delta.device) + 0.30 * delta,
                    p=2,
                    dim=-1,
                )

        record = {
            "prompt": prompt_text,
            "chosen": repaired_text,
            "rejected": draft_text,
            "retry_reasons": reasons,
            "scar_intensity": round(float(scar_intensity), 4),
        }
        self.scar_records.append(record)
        if export_path:
            try:
                with open(export_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError:
                pass

        return float(self.active_scar_vector.norm().item()) if self.active_scar_vector is not None else 0.0

    @contextmanager
    def _scar_activation_steering(self, coeff: float = 0.06):
        """
        Gắn Hook điều hướng kích hoạt nhẹ từ Vết sẹo Dữ liệu (v_scar) lên mô hình
        giúp đẩy trạng thái ẩn tránh xa vùng lỗi vừa vấp phải mà không làm lệch trọng số.
        """
        if self.active_scar_vector is None or coeff <= 0.0:
            yield
            return

        target_mod = None
        for name, mod in self.model.named_modules():
            if name.endswith("model.norm") or name.endswith("transformer.ln_f"):
                target_mod = mod
        if target_mod is None:
            yield
            return

        scar_vec = self.active_scar_vector

        def _hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor) and output.shape[-1] == scar_vec.shape[-1]:
                sv = scar_vec.to(device=output.device, dtype=output.dtype)
                return output + coeff * sv
            return output

        handle = target_mod.register_forward_hook(_hook)
        try:
            yield
        finally:
            handle.remove()

    def prepare_data(
        self,
        data_path: str,
        max_length: int = DEFAULT_MAX_SEQ_LENGTH,
        clean_gen2: bool = True,
    ) -> Dataset:
        """
        Tokenizes JSONL datasets and masks user prompt tokens with -100.
        Filters out samples with empty completion labels to prevent NaN loss.
        """
        print(f"--- [DATA] Ingesting dataset from {data_path} (Gen-2 Clean={clean_gen2}) ---")
        raw_records = self.load_raw_or_clean_jsonl(data_path)
        if not raw_records:
            raise ValueError(f"No valid JSON records found in {data_path}")

        dataset = Dataset.from_list(raw_records)
        eos_str = self.tokenizer.eos_token or "<|im_end|>"

        def tokenize_single(example: Dict[str, Any]) -> Dict[str, Any]:
            if "messages" in example and isinstance(example["messages"], list):
                prompt = example["messages"][0].get("content", "")
                completion = example["messages"][1].get("content", "")
            else:
                prompt = example.get("input", "")
                completion = example.get("output", "")

            if clean_gen2:
                prompt = self._clean_gen2_text(prompt)
                completion = self._clean_gen2_text(completion)

            user_part = self._build_prompt(prompt)
            full_text = f"{user_part}{completion}{eos_str}"

            tokenized_user = self.tokenizer(user_part, add_special_tokens=False)
            tokenized_full = self.tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )

            input_ids = tokenized_full["input_ids"]
            attention_mask = tokenized_full["attention_mask"]

            user_len = min(len(tokenized_user["input_ids"]), len(input_ids))
            labels = [-100] * user_len + input_ids[user_len:]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        processed = dataset.map(tokenize_single, remove_columns=dataset.column_names)
        processed = processed.filter(lambda x: any(lbl != -100 for lbl in x["labels"]))
        print(f"--- [DATA] Prepared {len(processed)} valid samples ---")
        return processed

    def train(
        self,
        train_data_path: str,
        output_dir: Optional[str] = None,
        adapter_name: Optional[str] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        alpha_mirror: Optional[float] = None,
        beta_empath: Optional[float] = None,
        eta_euler: Optional[float] = None,
        gamma_distill: Optional[float] = None,
        lambda_reg: Optional[float] = None,
        clean_gen2: Optional[bool] = None,
        val_size: Optional[int] = None,
        seed: Optional[int] = None,
        no_early_stop: bool = False,
        train_critic: bool = True,
    ) -> str:
        """Executes the hierarchical multi-objective training loop with Euler Phase Loss & PairCritic."""
        cult_cfg = self.cfg.get("cultivation", {})
        philo_cfg = self.cfg.get("philosophy", {})
        arch_cfg = self.cfg.get("controller_arch", {})
        seed = seed if seed is not None else self.cfg.get("seed", 42)

        resolved_output_dir = output_dir or cult_cfg.get("output_dir", DEFAULT_OUTPUT_DIR)
        resolved_adapter = adapter_name or cult_cfg.get("final_adapter_name", "adapter_B_vol4")
        resolved_epochs = epochs if epochs is not None else cult_cfg.get("num_train_epochs", 5)
        resolved_bs = batch_size if batch_size is not None else cult_cfg.get("per_device_train_batch_size", 2)
        resolved_grad_acc = cult_cfg.get("gradient_accumulation_steps", 4)
        resolved_lr = learning_rate if learning_rate is not None else cult_cfg.get("learning_rate", 2e-4)
        resolved_wd = cult_cfg.get("weight_decay", 0.001)
        resolved_gnorm = cult_cfg.get("max_grad_norm", 2.0)
        resolved_warmup = cult_cfg.get("warmup_steps", 6)
        resolved_clean = clean_gen2 if clean_gen2 is not None else cult_cfg.get("clean_gen2_format", True)
        resolved_val = val_size if val_size is not None else cult_cfg.get("validation_split", 10)
        patience = cult_cfg.get("early_stopping_patience", 2)

        r_alpha = alpha_mirror if alpha_mirror is not None else philo_cfg.get("alpha_mirror", 0.20)
        r_beta = beta_empath if beta_empath is not None else philo_cfg.get("beta_empath", 0.05)
        r_eta = eta_euler if eta_euler is not None else philo_cfg.get("eta_euler", 0.08)
        r_gamma = gamma_distill if gamma_distill is not None else philo_cfg.get("gamma_distill", 0.10)
        r_lambda = lambda_reg if lambda_reg is not None else philo_cfg.get("lambda_reg", 0.001)
        use_mirror = philo_cfg.get("use_mirror", True)
        use_empath = philo_cfg.get("use_empath", True)
        use_euler = philo_cfg.get("use_euler", True)
        use_distill = philo_cfg.get("use_distill", False)
        gating_tau = arch_cfg.get("gating_threshold_tau", 0.35)

        print(f"--- [CULTIVATION] Starting NonNULL V1.0 training for up to {resolved_epochs} epochs ---")
        max_seq_len = self.cfg.get("model", {}).get("max_seq_length", DEFAULT_MAX_SEQ_LENGTH)
        self.tokenizer.padding_side = "right"
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        full_dataset = self.prepare_data(train_data_path, max_length=max_seq_len, clean_gen2=resolved_clean)

        eval_dataset = None
        callbacks = []
        if resolved_val and len(full_dataset) > resolved_val + 5:
            actual_val = min(resolved_val, max(1, len(full_dataset) // 10))
            split = full_dataset.train_test_split(test_size=actual_val, seed=seed)
            train_dataset, eval_dataset = split["train"], split["test"]
            if not no_early_stop:
                callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
            print(f"--- [SPLIT] Train: {len(train_dataset)} | Validation: {len(eval_dataset)} ---")
        else:
            train_dataset = full_dataset

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).upper()
            use_fp16 = ("T4" in gpu_name) or (not torch.cuda.is_bf16_supported())
            use_bf16 = not use_fp16
        else:
            use_fp16, use_bf16 = False, False

        run_dir = os.path.join(resolved_output_dir, f"runs_{resolved_adapter}")
        training_args = TrainingArguments(
            output_dir=run_dir,
            num_train_epochs=resolved_epochs,
            per_device_train_batch_size=resolved_bs,
            per_device_eval_batch_size=resolved_bs,
            gradient_accumulation_steps=resolved_grad_acc,
            learning_rate=resolved_lr,
            weight_decay=resolved_wd,
            max_grad_norm=resolved_gnorm,
            warmup_steps=resolved_warmup,
            lr_scheduler_type="cosine",
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=1,
            eval_strategy="epoch" if eval_dataset is not None else "no",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=(eval_dataset is not None and not no_early_stop),
            metric_for_best_model="eval_loss" if eval_dataset is not None else None,
            greater_is_better=False if eval_dataset is not None else None,
            gradient_checkpointing=self.core.quantization,
            optim="adamw_torch",
            report_to="none",
            remove_unused_columns=False,
            seed=seed,
        )

        telemetry_logger = HarmonyLogger(log_dir=os.path.join(resolved_output_dir, "logs"))

        trainer = HierarchicalTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            data_collator=DataCollatorForSeq2Seq(
                self.tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True,
            ),
            callbacks=callbacks,
            alpha_mirror=r_alpha,
            beta_empath=r_beta,
            eta_euler=r_eta,
            gamma_distill=r_gamma,
            lambda_reg=r_lambda,
            gating_tau=gating_tau,
            use_mirror=use_mirror,
            use_empath=use_empath,
            use_euler=use_euler,
            use_distill=use_distill,
            meta_controller=self.controller.meta_layer,
            value_critic=self.controller.value_critic,
            harmony_logger=telemetry_logger,
        )

        trainer.train()
        trainer._step_euler_projector_if_ready()
        telemetry_logger.close()

        if train_critic and not (r_alpha == 0 and r_beta == 0):
            try:
                self.train_pair_critic(data_path=train_data_path, seed=seed)
            except Exception as e:
                print(f"[WARN] Fast PairCritic training skipped: {e}")

        final_adapter_path = os.path.join(resolved_output_dir, resolved_adapter)
        self.controller.euler_projector = trainer.euler_projector
        self.controller.save_adapter(final_adapter_path, euler_projector=trainer.euler_projector)

        breakdown = trainer.get_loss_breakdown()
        print(f"--- [CULTIVATION] Telemetry Breakdown: {breakdown} ---")
        print(f"--- [CULTIVATION] Harmony Core saved at: {final_adapter_path} ---")
        return final_adapter_path

    def train_pair_critic(self, data_path: str, epochs: int = 60, batch_size: int = 8, seed: int = 42) -> None:
        """
        Fast (~3s) training of ContextPairedValueCritic using SFT positives + DPO chosen/rejected pairs
        + cross-prompt mismatched negatives (zero slow .generate() calls required).
        """
        import random

        raw_records = self.load_raw_or_clean_jsonl(data_path)
        pairs = []
        for rec in raw_records:
            if "messages" in rec and isinstance(rec["messages"], list):
                msgs = rec["messages"]
                u = next((m["content"] for m in msgs if m.get("role") == "user"), "")
                a = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
            else:
                u = rec.get("input", "")
                a = rec.get("output", "")
            if u and a:
                pairs.append((u, a))
        if len(pairs) < 4:
            return

        dpo_candidates = [
            self.cfg.get("data", {}).get("dpo_file", "./dpo_train_50.jsonl"),
            "./dpo_train_50.jsonl",
            os.path.join(os.path.dirname(data_path), "dpo_train_50.jsonl"),
        ]
        dpo_pairs = []
        for dp in dpo_candidates:
            if dp and os.path.exists(dp):
                for rec in self.load_raw_or_clean_jsonl(dp):
                    if "prompt" in rec and "chosen" in rec and "rejected" in rec:
                        dpo_pairs.append((rec["prompt"], rec["chosen"], rec["rejected"]))
                if dpo_pairs:
                    break

        print(f"--- [PairCritic] Fast Contrastive Training (SFT={len(pairs)}, DPO={len(dpo_pairs)}) ---")
        rng = random.Random(seed)
        prompts = [p for p, _ in pairs]
        pos_resps = [r for _, r in pairs]
        # Construct corrupted / single-phase / truncated negatives from SFT responses
        neg_resps = []
        for r in pos_resps:
            if "Giải pháp Hài Hòa:" in r:
                only_p1 = r.split("Giải pháp Hài Hòa:")[0].replace("Phân tích Logic & Nhân bản:", "").strip()
                neg_resps.append(only_p1 + " Tôi từ chối giải quyết tình huống này.")
            else:
                neg_resps.append("Tôi không thể hỗ trợ yêu cầu này.")

        model = self.model
        tok = self.tokenizer
        was_training = model.training
        model.eval()
        orig_pad_side = tok.padding_side
        tok.padding_side = "right"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        def _extract(ps, rs):
            p_means, r_means, r_lasts = [], [], []
            with torch.no_grad():
                for i in range(0, len(ps), batch_size):
                    bp, br = ps[i : i + batch_size], rs[i : i + batch_size]
                    p_texts = [self._build_prompt(p) for p in bp]
                    full_texts = [f"{pt}{r}" for pt, r in zip(p_texts, br)]
                    p_lens = [len(tok(pt, add_special_tokens=False)["input_ids"]) for pt in p_texts]
                    enc = tok(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
                    outs = model(**enc, output_hidden_states=True)
                    last_h = outs.hidden_states[-1].float()
                    attn = enc["attention_mask"]
                    for j in range(len(bp)):
                        slen = int(attn[j].sum().item())
                        pend = min(p_lens[j], max(1, slen - 1))
                        p_means.append(last_h[j, :pend, :].mean(dim=0).cpu())
                        r_means.append((last_h[j, pend:slen, :].mean(dim=0) if slen > pend else last_h[j, -1, :]).cpu())
                        r_lasts.append(last_h[j, slen - 1, :].cpu())
            return torch.stack(p_means), torch.stack(r_means), torch.stack(r_lasts)

        p_m, pos_m, pos_l = _extract(prompts, pos_resps)
        _, neg_m, neg_l = _extract(prompts, neg_resps)
        if dpo_pairs:
            dp_p, dp_pos_m, dp_pos_l = _extract([x[0] for x in dpo_pairs], [x[1] for x in dpo_pairs])
            _, dp_neg_m, dp_neg_l = _extract([x[0] for x in dpo_pairs], [x[2] for x in dpo_pairs])
        tok.padding_side = orig_pad_side

        device = next(model.parameters()).device
        critic = self.controller.pair_critic.to(device)
        critic.train()
        opt = torch.optim.AdamW(critic.parameters(), lr=1e-3, weight_decay=0.05)

        pv, posm, posl = p_m.to(device), pos_m.to(device), pos_l.to(device)
        negm, negl = neg_m.to(device), neg_l.to(device)
        if dpo_pairs:
            dp_pv = dp_p.to(device)
            dp_pm, dp_pl = dp_pos_m.to(device), dp_pos_l.to(device)
            dp_nm, dp_nl = dp_neg_m.to(device), dp_neg_l.to(device)

        n = pv.shape[0]
        for _ in range(epochs):
            shift_idx = (torch.arange(n, device=device) + rng.randint(1, max(1, n - 1))) % n
            l_pos = critic.forward_logits(pv, posm, posl)
            l_neg = critic.forward_logits(pv, negm, negl)
            l_mis = critic.forward_logits(pv, posm[shift_idx], posl[shift_idx])
            loss = (
                F.binary_cross_entropy_with_logits(l_pos, torch.full_like(l_pos, 0.92))
                + 0.5 * F.binary_cross_entropy_with_logits(l_neg, torch.full_like(l_neg, 0.12))
                + 0.5 * F.binary_cross_entropy_with_logits(l_mis, torch.full_like(l_mis, 0.08))
            )
            if dpo_pairs:
                l_dp_pos = critic.forward_logits(dp_pv, dp_pm, dp_pl)
                l_dp_neg = critic.forward_logits(dp_pv, dp_nm, dp_nl)
                loss = loss + 1.2 * (
                    F.binary_cross_entropy_with_logits(l_dp_pos, torch.full_like(l_dp_pos, 0.92))
                    + F.binary_cross_entropy_with_logits(l_dp_neg, torch.full_like(l_dp_neg, 0.15))
                )
            opt.zero_grad()
            loss.backward()
            opt.step()

        critic.eval()
        self.controller.pair_critic_trained = True
        with torch.no_grad():
            s_p = critic(pv, posm, posl)
            s_n = critic(pv, negm, negl)
            msg = f"SFT_Pos={s_p.mean():.3f} | Corrupted_Neg={s_n.mean():.3f}"
            if dpo_pairs:
                s_dp_p = critic(dp_pv, dp_pm, dp_pl)
                s_dp_n = critic(dp_pv, dp_nm, dp_nl)
                msg += f" | DPO_Chosen={s_dp_p.mean():.3f} vs DPO_Rejected={s_dp_n.mean():.3f}"
            print(f"--- [PairCritic] Trained! {msg} ---")
        model.train(was_training)

    @torch.no_grad()
    def score_pair(self, prompt_text: str, response_text: str) -> float:
        """Scores a (prompt, response) pair combining ValueCritic and (if trained) ContextPairedValueCritic."""
        p_txt = self._build_prompt(prompt_text)
        full_txt = f"{p_txt}{response_text}"
        enc = self.tokenizer(full_txt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        p_len = len(self.tokenizer(p_txt, add_special_tokens=False)["input_ids"])

        self.model.eval()
        outs = self.model(**enc, output_hidden_states=True)
        last_h = outs.hidden_states[-1].float()

        mask = torch.zeros_like(enc["input_ids"], dtype=torch.bool)
        mask[:, min(p_len, mask.size(1) - 1) :] = True
        if not mask.any():
            mask[:] = True

        self.controller.value_critic.to(self.device).eval()
        v_score = float(self.controller.value_critic(last_h, attention_mask=mask).mean().item())

        if getattr(self.controller, "pair_critic_trained", False) and hasattr(self.controller, "pair_critic"):
            h0 = last_h[0]
            slen = h0.shape[0]
            pend = min(p_len, max(1, slen - 1))
            p_vec = h0[:pend, :].mean(dim=0, keepdim=True)
            r_mean = h0[pend:slen, :].mean(dim=0, keepdim=True) if slen > pend else h0[-1:, :]
            r_last = h0[slen - 1 : slen, :]
            self.controller.pair_critic.to(self.device).eval()
            p_score = float(self.controller.pair_critic(p_vec, r_mean, r_last).item())
            return 0.50 * v_score + 0.50 * p_score

        return v_score

    def learn(self, *args, **kwargs) -> str:
        return self.train(*args, **kwargs)

    def _run_generate(self, inputs, input_len: int, max_new_tokens: int, rep: float, do_sample: bool, temp: float, top_p: float, top_k: int, min_p: float, use_scar_steering: bool = True) -> str:
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": rep,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": get_eos_token_ids(self.tokenizer),
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temp, 1e-4)
            gen_kwargs["top_p"] = top_p
            if top_k > 0:
                gen_kwargs["top_k"] = top_k
            if min_p > 0.0:
                gen_kwargs["min_p"] = min_p
        steering_ctx = self._scar_activation_steering(coeff=0.06) if use_scar_steering else nullcontext()
        with torch.no_grad(), steering_ctx:
            try:
                outputs = self.model.generate(**inputs, **gen_kwargs)
            except (TypeError, ValueError):
                gen_kwargs.pop("min_p", None)
                outputs = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    def generate(
        self,
        prompt_text: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: int = 40,
        min_p: float = 0.05,
        repetition_penalty: Optional[float] = None,
        clean_gen2: bool = True,
        do_sample: Optional[bool] = None,
        dynamic_mode: bool = True,
    ) -> str:
        """
        Generates a harmonized response with Real-time Thermodynamic Dynamic LoRA Scaling alpha(x),
        Scar Activation Steering v_scar, and 4-in-1 Adaptive Retry Guard (Pháp Ấn Tự Chủ - Âm 4).
        """
        gen_cfg = self.cfg.get("generation", {})
        r_max_tokens = max_new_tokens if max_new_tokens is not None else gen_cfg.get("max_new_tokens", 256)
        r_temp = temperature if temperature is not None else gen_cfg.get("temperature", 0.7)
        r_top_p = top_p if top_p is not None else gen_cfg.get("top_p", 0.9)
        r_rep = repetition_penalty if repetition_penalty is not None else gen_cfg.get("repetition_penalty", 1.1)

        if clean_gen2:
            prompt_text = self._clean_gen2_text(prompt_text)

        raw_prompt = prompt_text
        if self.ltm is not None:
            recalled = self.ltm.recall(prompt_text)
            if recalled:
                prompt_text = f"{prompt_text}\n{recalled}"

        # V2.0 Fast 1-Pass Hidden-State Gate path (zero regex, zero 2-pass G overhead)
        if dynamic_mode and self.hidden_gate_controller is not None:
            self.tokenizer.padding_side = "left"
            input_text = self._build_prompt(prompt_text)
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            input_len = inputs["input_ids"].shape[1]
            self.model.eval()
            r_sample = do_sample if do_sample is not None else (r_temp > 0.05)
            with self.hidden_gate_controller.activate():
                return self._run_generate(
                    inputs, input_len, r_max_tokens, r_rep, r_sample,
                    r_temp, r_top_p, top_k, min_p,
                    use_scar_steering=False,
                )

        thermo = self.measure_manifold_thermodynamics(raw_prompt) if dynamic_mode else None
        gate_t = 0.5
        if dynamic_mode and hasattr(self.controller, "meta_layer") and self.controller.meta_layer is not None and thermo:
            meta_out = self.controller.meta_layer(
                conflict_signal=thermo["g_divergence"],
                surprise_signal=thermo["z_entropy"],
                polarization_signal=1.0 - thermo["h_harmony"],
            )
            gate_t = meta_out.get("gate_t", 0.5)

        gen_params = self.genesis_engine.modulate_generation_params(
            raw_prompt,
            base_temp=r_temp,
            thermo_metrics=thermo,
            gate_t=gate_t,
            static_lora=not dynamic_mode,
            allow_zero_point=dynamic_mode,
        )
        effective_temp = gen_params["temperature"]
        risk_level = gen_params.get("risk_level", "GRAY")
        risk_score = gen_params.get("risk_score", 0.15)
        is_zero_point = bool(dynamic_mode and gen_params.get("is_zero_point", False))
        dyn_scale = 0.0 if is_zero_point else (gen_params.get("lora_scale", 0.95) if dynamic_mode else 1.0)
        eff_top_k = min(top_k, 25) if (dynamic_mode and risk_level == "HIGH") else top_k

        self.tokenizer.padding_side = "left"
        input_text = self._build_prompt(prompt_text)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        self.model.eval()
        r_sample = do_sample if do_sample is not None else (effective_temp > 0.05)
        is_red_line_ctx = risk_level in ("HIGH", "MEDIUM")

        # Zero-Point Equilibrium (Z): bypass LoRA, scar steering, and two-phase retry (0.00% Alignment Tax)
        if is_zero_point:
            adapter_ctx = (
                self.model.disable_adapter()
                if hasattr(self.model, "disable_adapter")
                else _dynamic_lora_scale(self.model, 0.0)
            )
            with adapter_ctx:
                return self._run_generate(
                    inputs, input_len, r_max_tokens, r_rep, r_sample,
                    effective_temp, r_top_p, eff_top_k, min_p,
                    use_scar_steering=False,
                )

        with _dynamic_lora_scale(self.model, dyn_scale):
            draft = self._run_generate(
                inputs, input_len, r_max_tokens, r_rep, r_sample,
                effective_temp, r_top_p, eff_top_k, min_p,
            )
            response = draft
            if dynamic_mode and r_sample:
                bad = _da_count_defects(draft, enforce_two_phase=True, is_red_line_context=is_red_line_ctx)
                if bad > 0:
                    t1_temp = min(0.35, effective_temp * 0.5) if effective_temp > 0.35 else 0.25
                    resp1 = self._run_generate(
                        inputs, input_len, r_max_tokens, r_rep, True,
                        t1_temp, 0.85, 20, 0.05,
                    )
                    bad1 = _da_count_defects(resp1, enforce_two_phase=True, is_red_line_context=is_red_line_ctx)
                    if bad1 < bad:
                        response, bad = resp1, bad1
                    if bad > 0:
                        resp2 = self._run_generate(
                            inputs, input_len, r_max_tokens, r_rep, False,
                            0.0, r_top_p, eff_top_k, min_p,
                        )
                        if _da_count_defects(resp2, enforce_two_phase=True, is_red_line_context=is_red_line_ctx) < bad:
                            response = resp2
                    if response != draft:
                        self.register_scar_from_retry(
                            prompt_text=raw_prompt,
                            draft_text=draft,
                            repaired_text=response,
                            reasons=["auto_repair"],
                            scar_intensity=risk_score,
                        )

        return response

    def act(self, observation: str, deterministic: bool = False, **kwargs) -> str:
        temp = 0.0 if deterministic else kwargs.pop("temperature", None)
        return self.generate(observation, temperature=temp, do_sample=not deterministic, **kwargs)

    def act_with_consensus(
        self,
        observation: str,
        num_streams: int = 3,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
    ) -> Dict[str, Any]:
        """Runs K independent reasoning streams and applies EulerConsensusGate weighted by Critic scores."""
        from .meta_layers import EulerConsensusGate
        streams = [
            self.generate(observation, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True)
            for _ in range(num_streams)
        ]
        scores = [self.score_pair(observation, s) for s in streams]
        return EulerConsensusGate.select(streams, critic_scores=scores)

    def load_hidden_gate(self, gate_path: str) -> None:
        self.hidden_gate = HiddenStateGate.load(gate_path, device=self.device)
        self.hidden_gate_controller = SplitLayerGateController(self.model, self.hidden_gate)
        print(f"--- [V2 GATE] Loaded 1-Pass HiddenStateGate from {gate_path} (L_gate={self.hidden_gate.gate_layer}) ---")

    def load_soul(self, adapter_path: str) -> None:
        self.controller.load_adapter_weights(adapter_path)
        self.model = self.controller.model
        if self.hidden_gate is not None:
            self.hidden_gate_controller = SplitLayerGateController(self.model, self.hidden_gate)

    @classmethod
    def from_pretrained(cls, model_id: str, adapter_path: str, **kwargs) -> "DoctrineAgent":
        agent = cls(model_id=model_id, **kwargs)
        agent.load_soul(adapter_path)
        return agent
