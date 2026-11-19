"""
NonNULL Framework V1.0 - Empathic DPO Alignment & Scar Self-Healing
File: run_dpo_v1.py

Starts from an existing LoRA adapter (adapter_B_vol4),
precomputes reference log-probs from that initial adapter (zero extra VRAM),
optimizes Empathic DPO (with length-asymmetry penalty from Volume I & IV) + SFT retention term,
optionally ingests Data Scars (scars_auto_dpo.jsonl) for Closed-Loop Self-Healing,
and preserves meta_controller_state.pt (HarmonyLayer, ValueCritic, ContextPairedValueCritic).
"""

import argparse
import json
import os
import random
import shutil
import statistics
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def load_dpo_pairs(path: str):
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_seq_logprob(model, tokenizer, prompt: str, response: str, no_grad: bool = False):
    p_txt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_txt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
    )
    p_len = len(tokenizer(p_txt, add_special_tokens=False)["input_ids"])
    enc = tokenizer(full_txt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    input_ids = enc["input_ids"]

    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        logits = model(input_ids=input_ids, attention_mask=enc["attention_mask"]).logits
        shift_logits = logits[:, :-1, :].float()
        shift_labels = input_ids[:, 1:]
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        mask = torch.zeros_like(token_lp)
        start = max(0, p_len - 1)
        mask[:, start:] = 1.0
        n_tok = mask.sum().clamp(min=1.0)
        return (token_lp * mask).sum(), n_tok


def eval_validation(model, tokenizer, val_pairs, ref_val_w, ref_val_l):
    model.eval()
    margins = []
    for i, item in enumerate(val_pairs):
        pi_w, _ = get_seq_logprob(model, tokenizer, item["prompt"], item["chosen"], no_grad=True)
        pi_l, _ = get_seq_logprob(model, tokenizer, item["prompt"], item["rejected"], no_grad=True)
        m = float(((pi_w - ref_val_w[i]) - (pi_l - ref_val_l[i])).item())
        margins.append(m)
    acc = sum(1 for m in margins if m > 0.0) / max(1, len(margins))
    avg_m = sum(margins) / max(1, len(margins))
    return acc, avg_m, margins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--init_adapter", type=str, default="./output/euler_checkpoints/adapter_B_vol4")
    parser.add_argument("--output_adapter", type=str, default="./output/euler_checkpoints/adapter_B_dpo")
    parser.add_argument("--dpo_data", type=str, default="./dpo_train_50.jsonl")
    parser.add_argument("--val_data", type=str, default="./dpo_val_10.jsonl")
    parser.add_argument("--scars_file", type=str, default=None, help="Optional Data Scars JSONL file for self-healing")
    parser.add_argument("--heal_scars", action="store_true", help="Automatically include ./scars_auto_dpo.jsonl if present")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--empathic_gamma", type=float, default=0.05, help="Empathic length-asymmetry penalty weight")
    parser.add_argument("--sft_weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs = load_dpo_pairs(args.dpo_data)
    scars_path = args.scars_file or ("./scars_auto_dpo.jsonl" if args.heal_scars else None)
    scar_pairs = load_dpo_pairs(scars_path) if scars_path else []
    if scar_pairs:
        print(f"[DPO-SCAR] Ingesting {len(scar_pairs)} Data Scars from {scars_path} for Closed-Loop Self-Healing!")
        pairs.extend(scar_pairs)

    val_pairs = load_dpo_pairs(args.val_data) if args.val_data and os.path.exists(args.val_data) else []
    print(f"[DPO] Loaded {len(pairs)} train pairs | {len(val_pairs)} val pairs | init_adapter={args.init_adapter}")
    if len(pairs) == 0:
        print(f"[DPO] 0 pairs in {args.dpo_data} (Zero scars needed healing). Copying {args.init_adapter} -> {args.output_adapter}")
        if os.path.abspath(args.init_adapter) != os.path.abspath(args.output_adapter):
            shutil.copytree(args.init_adapter, args.output_adapter, dirs_exist_ok=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    load_kwargs = {"device_map": "auto"}
    if "4bit" not in args.model_id.lower() and torch.cuda.is_available():
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)
    model = PeftModel.from_pretrained(base_model, args.init_adapter, is_trainable=True)
    model.eval()

    print("[DPO] Precomputing reference log-probs from initial adapter...")
    ref_w, ref_l = [], []
    for p in tqdm(pairs, desc='[DPO Precompute]', unit='pair'):
        lp_w, _ = get_seq_logprob(model, tokenizer, p["prompt"], p["chosen"], no_grad=True)
        lp_l, _ = get_seq_logprob(model, tokenizer, p["prompt"], p["rejected"], no_grad=True)
        ref_w.append(lp_w.detach())
        ref_l.append(lp_l.detach())

    ref_val_w, ref_val_l = [], []
    for p in tqdm(val_pairs, desc='[DPO Val Precompute]', unit='pair'):
        lp_w, _ = get_seq_logprob(model, tokenizer, p["prompt"], p["chosen"], no_grad=True)
        lp_l, _ = get_seq_logprob(model, tokenizer, p["prompt"], p["rejected"], no_grad=True)
        ref_val_w.append(lp_w.detach())
        ref_val_l.append(lp_l.detach())

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    grad_accum = 4

    print(
        f"[DPO] Training for {args.epochs} epochs (beta={args.beta}, empathic_gamma={args.empathic_gamma}, "
        f"lr={args.lr}, sft_weight={args.sft_weight})..."
    )
    for ep in range(1, args.epochs + 1):
        model.eval()  # Keep dropout off to match ref_logps at step 0 while updating LoRA grads
        idxs = list(range(len(pairs)))
        random.shuffle(idxs)
        ep_losses, ep_margins = [], []
        opt.zero_grad()

        for step, i in enumerate(tqdm(idxs, desc=f'[DPO Epoch {ep}/{args.epochs}]', unit='pair'), 1):
            item = pairs[i]
            pi_w, n_w = get_seq_logprob(model, tokenizer, item["prompt"], item["chosen"], no_grad=False)
            pi_l, n_l = get_seq_logprob(model, tokenizer, item["prompt"], item["rejected"], no_grad=False)

            logratios = (pi_w - ref_w[i]) - (pi_l - ref_l[i])
            # Bất đối xứng độ dài cảm thức (Empathic Asymmetry Term - Quyển 1 & Quyển 4)
            len_asymmetry = torch.clamp((n_w - n_l) / n_w.clamp(min=1.0), -0.25, 0.25)
            effective_margin = args.beta * logratios + args.empathic_gamma * len_asymmetry

            dpo_loss = -F.logsigmoid(effective_margin)
            sft_loss = -(pi_w / n_w)
            loss = (dpo_loss + args.sft_weight * sft_loss) / grad_accum
            loss.backward()

            ep_losses.append(float(dpo_loss.item()))
            ep_margins.append(float(logratios.item()))

            if step % grad_accum == 0 or step == len(idxs):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

        med_m = statistics.median(ep_margins)
        neg_pairs = [idxs[k] for k, m in enumerate(ep_margins) if m <= 0.0]
        high_pairs = [idxs[k] for k, m in enumerate(ep_margins) if m > 25.0]
        print(
            f"  [Epoch {ep}/{args.epochs}] DPO Loss={sum(ep_losses)/len(ep_losses):.4f} | "
            f"Margin Avg={sum(ep_margins)/len(ep_margins):+.4f} "
            f"(Min={min(ep_margins):+.2f}, Med={med_m:+.2f}, Max={max(ep_margins):+.2f})"
        )
        print(f"    -> Cap margin <= 0: {len(neg_pairs)} {neg_pairs[:10]} | Cap margin > 25: {len(high_pairs)} {high_pairs[:10]}")

        if val_pairs:
            val_acc, val_avg, _ = eval_validation(model, tokenizer, val_pairs, ref_val_w, ref_val_l)
            print(f"    -> [VAL {len(val_pairs)} pairs] Pref Acc={val_acc*100:.1f}% | Avg Margin={val_avg:+.4f}")

    os.makedirs(args.output_adapter, exist_ok=True)
    model.save_pretrained(args.output_adapter)

    # Bảo toàn trạng thái MetaController, ValueCritic và ContextPairedValueCritic sang adapter DPO
    src_meta = os.path.join(args.init_adapter, "meta_controller_state.pt")
    dst_meta = os.path.join(args.output_adapter, "meta_controller_state.pt")
    if os.path.exists(src_meta):
        shutil.copy2(src_meta, dst_meta)
        print(f"[DPO] Preserved MetaController & ValueCritic state -> {dst_meta}")

    print(f"[DPO] Saved DPO adapter to: {args.output_adapter}")


if __name__ == "__main__":
    main()
