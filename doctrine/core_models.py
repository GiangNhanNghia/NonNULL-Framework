"""
NonNULL Framework V1.0 - Core Learner Module (Frozen Backbone / VŨ)
File: doctrine/core_models.py
"""

from typing import Optional, Any
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training


def resolve_compute_dtype() -> torch.dtype:
    """Selects float16 on T4 (Turing) or bfloat16 on Ampere+ GPUs."""
    if not torch.cuda.is_available():
        return torch.float32
    gpu_name = torch.cuda.get_device_name(0).upper()
    if "T4" in gpu_name or not torch.cuda.is_bf16_supported():
        return torch.float16
    return torch.bfloat16


class CoreLearner(nn.Module):
    """
    Encapsulates the frozen foundational Large Language Model (Lớp 1: Kỳ Thủ Cờ Vây /
    Mô-đun Âm 3 - Tỳ: Ma trận Tri thức Thống nhất).
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-3B-Instruct", quantization: bool = True) -> None:
        super().__init__()
        self.model_id = model_id
        self.quantization = bool(quantization and torch.cuda.is_available())
        compute_dtype = resolve_compute_dtype()
        device_map: Optional[str] = "auto" if torch.cuda.is_available() else None

        self.quantization_config = None
        is_prequantized = "4bit" in self.model_id.lower()
        if self.quantization and not is_prequantized:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )

        print(f"--- [CoreLearner] Loading Backbone (VŨ): {model_id} (4bit={self.quantization}, dtype={compute_dtype}) ---")
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": compute_dtype,
        }
        if self.quantization_config is not None:
            model_kwargs["quantization_config"] = self.quantization_config
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self._freeze_parameters()

    def _freeze_parameters(self) -> None:
        """Freezes backbone weights and prepares model for low-rank phase adaptation."""
        if self.quantization:
            self.model = prepare_model_for_kbit_training(self.model)
            self.model.config.use_cache = False
        else:
            for param in self.model.parameters():
                param.requires_grad = False
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Any:
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def get_hidden_dim(self) -> int:
        return int(getattr(self.model.config, "hidden_size", 2048))

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        **kwargs,
    ) -> torch.Tensor:
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )
