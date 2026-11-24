# Edge Deployment Guide — 16 GB RAM Laptops (Mac & PC)

NonNULL Framework v1.0.1 is designed so that anyone can train an alignment adapter in **~4–6 minutes on a single Colab GPU** and deploy it locally on a **16 GB RAM laptop** using 4-bit quantization (~5.2 GB memory footprint).

## Option 1: PC / Linux / macOS via `llama.cpp` (GGUF LoRA)

1. Download a 4-bit GGUF base model (`Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`, ~4.9 GB).
2. Convert your trained NonNULL PEFT adapter (`./output/euler_checkpoints/adapter_v101_lndpo`) to GGUF LoRA:
   ```bash
   python3 edge/export_gguf_lora.py \
     --adapter_path ./output/euler_checkpoints/adapter_v101_lndpo \
     --base_model_hf meta-llama/Meta-Llama-3.1-8B-Instruct \
     --llama_cpp_dir ./llama.cpp \
     --out_file ./output/nonnull_v101_lora.gguf
   ```
3. Run locally with `llama-cli` or `llama-server` (`--lora-scaled nonnull_v101_lora.gguf:1.0` for active alignment, or `0.0` for Zero-Point bypass).

## Option 2: Apple Silicon Mac (M1/M2/M3/M4, 16 GB Unified Memory) via `mlx-lm`

1. Install MLX: `pip install mlx-lm`
2. Run locally using the 4-bit MLX backbone (`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`):
   ```bash
   python3 edge/mlx_nonnull_inference.py \
     --model_id mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
     --adapter_path ./output/mlx_nonnull_adapter \
     --prompt "Your prompt here"
   ```
