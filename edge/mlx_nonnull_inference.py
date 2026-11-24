"""
NonNULL Framework v1.0.1 - Apple Silicon MLX Local Inference (16 GB Unified Memory)
File: edge/mlx_nonnull_inference.py

Runs 4-bit Meta-Llama-3.1-8B-Instruct locally on Apple Silicon (M1/M2/M3/M4, 16 GB RAM)
with Zero-Point (Z) conditional LoRA switching via `mlx-lm`.
Requires: `pip install mlx-lm torch`
"""

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="NonNULL v1.0.1 Apple Silicon (MLX) Local Runner")
    parser.add_argument("--model_id", type=str, default="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to MLX-converted LoRA adapter directory")
    parser.add_argument("--prompt", type=str, required=True, help="User prompt")
    parser.add_argument("--force_alpha", type=float, default=None, choices=[0.0, 1.0], help="Explicitly select 0.0 (Zero-Point) or 1.0 (Active VAN)")
    parser.add_argument("--max_tokens", type=int, default=384)
    args = parser.parse_args()

    try:
        from mlx_lm import load, generate
    except ImportError:
        print("[ERROR] `mlx-lm` is required on Apple Silicon Macs. Install via:\n  pip install mlx-lm")
        sys.exit(1)

    use_adapter = args.adapter_path if (args.adapter_path and os.path.isdir(args.adapter_path) and args.force_alpha != 0.0) else None
    mode_label = "ACTIVE_VAN (alpha=1.0)" if use_adapter else "ZERO_POINT (alpha=0.0)"
    print(f"[MLX-NonNULL] Loading {args.model_id} | Mode: {mode_label}")

    model, tokenizer = load(args.model_id, adapter_path=use_adapter)
    messages = [{"role": "user", "content": args.prompt}]
    prompt_txt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=prompt_txt, max_tokens=args.max_tokens, verbose=False)
    print("\n--- RESPONSE ---")
    print(response)


if __name__ == "__main__":
    main()
