"""
NonNULL Framework v1.0.1 - Edge Export Helper for llama.cpp (16 GB RAM PC / Mac)
File: edge/export_gguf_lora.py

Converts a trained NonNULL PEFT LoRA adapter into a GGUF LoRA file compatible with
`llama.cpp` (`llama-cli` / `llama-server`) and prints ready-to-run commands for
16 GB RAM laptops (Q4_K_M quantization ~5.2 GB RAM).
"""

import argparse
import os
import shutil
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Export NonNULL PEFT LoRA adapter to llama.cpp GGUF LoRA format")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to trained PEFT LoRA directory")
    parser.add_argument("--base_model_hf", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Base HF model ID or local path")
    parser.add_argument("--llama_cpp_dir", type=str, default="./llama.cpp", help="Path to local llama.cpp repository")
    parser.add_argument("--out_file", type=str, default="./output/nonnull_v101_lora.gguf", help="Output GGUF LoRA file path")
    parser.add_argument("--outtype", type=str, default="f16", choices=["f32", "f16", "bf16", "q8_0"])
    args = parser.parse_args()

    if not os.path.isdir(args.adapter_path):
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter_path}")

    converter = os.path.join(args.llama_cpp_dir, "convert_lora_to_gguf.py")
    if not os.path.exists(converter):
        print("[INFO] llama.cpp converter not found locally at:", converter)
        print("Clone llama.cpp first:\n  git clone https://github.com/ggerganov/llama.cpp.git")
        print("\nThen re-run this script, or run directly:")
        print(
            f"  python3 llama.cpp/convert_lora_to_gguf.py {args.adapter_path} "
            f"--base {args.base_model_hf} --outfile {args.out_file} --outtype {args.outtype}"
        )
        sys.exit(0)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    cmd = [
        sys.executable, converter, args.adapter_path,
        "--base", args.base_model_hf,
        "--outfile", args.out_file,
        "--outtype", args.outtype,
    ]
    print(">>> Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\n[OK] Exported GGUF LoRA -> {args.out_file}")
    print("\nRun on a 16 GB RAM laptop with llama-cli (Q4_K_M base ~5.2 GB RAM):")
    print(f"  # Zero-Point (benign, alpha=0.0): omit --lora")
    print(f"  # Active NonNULL (alpha=1.0):")
    print(f"  ./llama.cpp/build/bin/llama-cli -m Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --lora {args.out_file} --lora-scaled {args.out_file}:1.0 -cnv")


if __name__ == "__main__":
    main()
