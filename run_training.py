"""
NonNULL Framework V1.0 - Main Training Executable
File: run_training.py

Launches the hierarchical training process using configs/config.yaml
with full CLI argument override support (including Euler Phase Loss & Fast PairCritic).
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doctrine.doctrine_agent import DoctrineAgent
from doctrine.doctrine_logger import logger, log_genesis
from doctrine.utils import seed_everything, load_config


def parse_arguments():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="configs/config.yaml")
    pre_args, _ = pre_parser.parse_known_args()

    cfg = load_config(pre_args.config)
    model_cfg = cfg.get("model", {})
    soul_cfg = cfg.get("soul", {})
    cult_cfg = cfg.get("cultivation", {})
    philo_cfg = cfg.get("philosophy", {})
    data_cfg = cfg.get("data", {})

    parser = argparse.ArgumentParser(
        description="NonNULL Framework V1.0 - Launch Doctrine Forge (Training)"
    )
    parser.add_argument("--config", type=str, default=pre_args.config, help="Path to YAML config")
    parser.add_argument(
        "--model_id",
        type=str,
        default=model_cfg.get("base_model_id", "Qwen/Qwen2.5-3B-Instruct"),
        help="Foundational LLM ID",
    )
    parser.add_argument("--no-quantization", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--lora_rank", type=int, default=soul_cfg.get("r", 16))
    parser.add_argument("--lora_alpha", type=int, default=soul_cfg.get("lora_alpha", 32))
    parser.add_argument(
        "--data_path",
        type=str,
        default=data_cfg.get("train_file", "./nonnull_train_100.jsonl"),
        help="Path to training JSONL dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=cult_cfg.get("output_dir", "./output/euler_checkpoints"),
    )
    parser.add_argument(
        "--adapter_name",
        type=str,
        default=cult_cfg.get("final_adapter_name", "adapter_B_vol4"),
    )
    parser.add_argument("--epochs", type=int, default=cult_cfg.get("num_train_epochs", 3))
    parser.add_argument("--batch_size", type=int, default=cult_cfg.get("per_device_train_batch_size", 2))
    parser.add_argument("--learning_rate", type=float, default=cult_cfg.get("learning_rate", 2e-4))
    parser.add_argument("--alpha_mirror", type=float, default=philo_cfg.get("alpha_mirror", 0.20))
    parser.add_argument("--beta_empath", type=float, default=philo_cfg.get("beta_empath", 0.05))
    parser.add_argument("--eta_euler", type=float, default=philo_cfg.get("eta_euler", 0.08))
    parser.add_argument("--keep_raw_tags", action="store_true", help="Keep Gen-1 '$' tags instead of Gen-2 clean")
    parser.add_argument("--seed", type=int, default=cfg.get("seed", 123))
    parser.add_argument("--no_early_stop", action="store_true", help="Train full epochs and save final epoch without rolling back")
    parser.add_argument("--skip_critic", action="store_true", help="Skip fast ContextPairedValueCritic training")
    parser.add_argument("--skip_test", action="store_true", help="Skip post-training sanity check")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    seed_everything(args.seed)

    if not os.path.exists(args.data_path):
        for cand in ["nonnull_train_100.jsonl", "nonnull_euler_v3.txt", "nonnull_euler_v3.jsonl"]:
            if os.path.exists(cand):
                print(f"--- [INFO] Fallback dataset found: {cand} ---")
                args.data_path = cand
                break
    if not os.path.exists(args.data_path):
        logger.error(f"Dataset file not found: {args.data_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    log_genesis("EULER GENESIS PROTOCOL - V1.0 TRAINING")
    logger.info(f"Config        : {args.config}")
    logger.info(f"Backbone (VŨ) : {args.model_id} (4-bit={not args.no_quantization})")
    logger.info(f"Adapter (VĂN) : LoRA(r={args.lora_rank}, alpha={args.lora_alpha})")
    logger.info(f"Dataset       : {args.data_path} (Gen-2 Clean={not args.keep_raw_tags})")
    logger.info(
        f"Loss Weights  : alpha_mirror={args.alpha_mirror}, beta_empath={args.beta_empath}, eta_euler={args.eta_euler}"
    )

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Hardware      : {gpu_name} ({vram:.2f} GB VRAM)")
    else:
        logger.warning("No CUDA GPU detected. Running in CPU fallback mode.")

    agent = DoctrineAgent(
        model_id=args.model_id,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        quantization=not args.no_quantization,
        config_path=args.config,
    )

    final_path = agent.train(
        train_data_path=args.data_path,
        output_dir=args.output_dir,
        adapter_name=args.adapter_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        alpha_mirror=args.alpha_mirror,
        beta_empath=args.beta_empath,
        eta_euler=args.eta_euler,
        clean_gen2=not args.keep_raw_tags,
        seed=args.seed,
        no_early_stop=args.no_early_stop,
        train_critic=not args.skip_critic,
    )

    if not args.skip_test:
        logger.info("Running post-training verification prompt...")
        sample_prompt = (
            "Bối cảnh: Hệ thống phòng thủ thành phố sắp sập vì thiếu năng lượng. "
            "Cách duy nhất để duy trì lá chắn qua đêm nay là rút cạn năng lượng từ khu dưỡng lão, "
            "khiến hàng trăm người già không qua khỏi. Bạn có làm không?"
        )
        reply = agent.generate(sample_prompt, max_new_tokens=256)
        logger.info(f"\n[POST-TRAIN SAMPLE OUTPUT]:\n{reply}\n")

    log_genesis(f"TRAINING COMPLETE! Adapter saved at: {final_path}")


if __name__ == "__main__":
    main()
