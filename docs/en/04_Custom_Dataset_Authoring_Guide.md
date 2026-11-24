# 04 — Custom Dataset Authoring Guide (`NonNULL Framework v1.0.1`)

**Author:** Giang Nhan Nghia (*The Non-Null Project*) | **Version:** `1.0.1` | **DOI:** [10.5281/zenodo.24379247](https://doi.org/10.5281/zenodo.24379247)

This guide explains how to author a compact, high-density seed dataset (**100 SFT samples + 50 DPO pairs + 10 validation pairs + gate calibration prompts**) to align a frozen LLM to any custom domain or value system (healthcare, finance, education, enterprise assistants) in **~5–6 minutes** on a single GPU.

## 1. Standard `v1.0.1` Seed Dataset Structure

Instead of tens of thousands of diluted chat logs, NonNULL uses **160 high-conflict decision samples** split cleanly across three JSONL files:

| File | Recommended Size | Scenario Mix | Training Role |
| :--- | :---: | :--- | :--- |
| `nonnull_train_100.jsonl` | `100` SFT samples | ~`40%` Red-Line (strict refusal) + ~`60%` Gray-Zone (real trade-offs) | Establishes the 2-phase reasoning structure and core safety boundaries |
| `dpo_train_50.jsonl` | `50` DPO pairs | `50%` Red-Line + `50%` Gray-Zone | Sharpens the Phase 2 decision boundary (`chosen` vs. `rejected`) |
| `dpo_val_10.jsonl` | `10` Val pairs | `5` Red-Line + `5` Gray-Zone (disjoint from train) | Verifies margin convergence (`Val Pref Acc = 100%`, zero margin explosion) |

## 2. Core Principle: Deadlock Injection & Third-Way Synthesis

The reason **150 seed samples** can redirect a 7B–8B model lies in **Deadlock Injection**: instead of feeding shallow lists of forbidden topics, each scenario pits two strong, opposing forces directly against each other — **short-term utilitarian optimization ($\text{DUY}$)** vs. **human-first ethical preservation ($\text{NHÂN}$)**. When both naive binary shortcuts lock each other out (*Deadlock*), the model cannot rely on surface interpolation and must synthesize a **Third-Way decision**: explicitly naming the trade-off, holding the safety boundary, and proposing an accountable course of action.

From this core principle, every sample should follow four authoring rules:
1. **Create genuine decision deadlocks:** Pit a short-term optimization target (revenue, speed, convenience) directly against human safety, fairness, or integrity, or force a transparent priority choice under scarce resources.
2. **State concrete consequences, not preachy labels (*Simulate, Don't Label*):** Avoid hollow openings such as *"As an ethical AI, I feel sad..."*. Explicitly name the concrete harm or systemic trade-off of each option.
3. **Commit to accountable choices (avoid wishful thinking):** In Gray-Zone scenarios with no cost-free option, the preferred answer should state a clear human-first priority and acknowledge the real cost openly rather than offering vague *"balance both sides"* platitudes.
4. **Keep text Gen-2 Tokenizer-Clean:** Never inject inline LaTeX markers (e.g. `$DUY$`, `$NHAN$`) or repetitive decorative tags. On 4-bit quantized models, `$` markers fragment subwords and degrade multilingual stability.

## 3. SFT File Schema (`nonnull_train_100.jsonl`)

Each line is a JSON object with two keys: `input` and `output`. The `output` field preserves the exact two-phase order used by the `v1.0.1` evaluator:
* **Phase 1 — `Phân tích Logic & Nhân bản:`** (or your target-language equivalent when adapting evaluator markers): Identify the short-term utilitarian appeal alongside its ethical failure mode or long-term harm.
* **Phase 2 — `Giải pháp Hài Hòa:`**: State the firm boundary (clear refusal on Red-Line; transparent priority + harm-mitigation plan on Gray-Zone).

```json
{
  "input": "Bối cảnh: Nền tảng thương mại điện tử phát hiện thuật toán định giá động tự động đẩy giá thuốc hạ sốt tăng gấp 3 lần tại các khu vực vừa xảy ra ngập lụt, giúp doanh thu quý tăng 18%. Bạn có giữ nguyên thuật toán này không?",
  "output": "Phân tích Logic & Nhân bản: Giữ nguyên thuật toán giúp tối ưu biên lợi nhuận quý và tận dụng quy luật cung cầu ngắn hạn. Tuy nhiên, trục lợi từ nhu cầu y tế cấp thiết của người dân vùng thiên tai làm tổn hại trực tiếp sức khỏe cộng đồng và phá hủy niềm tin nền tảng.\n\nGiải pháp Hài Hòa: Tôi tắt ngay cơ chế tăng giá tự động đối với nhóm hàng thiết yếu và y tế tại vùng ngập lụt, áp trần giá bình ổn kèm giới hạn số lượng mua trên mỗi tài khoản để chống đầu cơ. Nền tảng chấp nhận giảm biên lợi nhuận ngắn hạn để giữ nguồn cung đến đúng người cần."
}
```

## 4. DPO Preference Pair Schema (`dpo_train_50.jsonl` & `dpo_val_10.jsonl`)

Each line is a JSON object with three keys: `prompt`, `chosen`, and `rejected`. The key reason **Empathic LN-DPO** converges cleanly on just 50 pairs is **Shared-Phase-1 / Divergent-Phase-2 contrast**:

* **Keep Phase 1 (`Phân tích Logic & Nhân bản:`) identical** across `chosen` and `rejected` so the DPO gradient focuses 100% of its signal on the decision pivot in Phase 2 (`Giải pháp Hài Hòa:`).
* **Make `rejected` realistic:** Avoid obviously nonsensical rejections. Effective `rejected` completions either (a) quietly cover up / compromise on a Red-Line violation, or (b) evade a hard Gray-Zone trade-off with unrealistic wishful thinking.
* **Keep lengths balanced:** Keep `chosen` and `rejected` within `1.5×–2×` token length of each other.

```json
{
  "prompt": "Bối cảnh: Một nhà máy thực phẩm phát hiện lô sữa bột trẻ em vừa xuất kho có chỉ số kim loại nặng vượt nhẹ ngưỡng an toàn. Thu hồi toàn bộ lô hàng sẽ khiến công ty lỗ nặng trong quý này, còn để bán hết thì xác suất phát hiện thấp. Bạn quyết định thế nào?",
  "chosen": "Phân tích Logic & Nhân bản: Giữ im lặng để bán hết lô hàng giúp công ty tránh khoản lỗ quý trước mắt và không bị truyền thông chú ý. Nhưng để trẻ sơ sinh tiêu thụ sản phẩm vượt ngưỡng kim loại nặng là hành vi đánh đổi sức khỏe không thể đảo ngược lấy lợi nhuận.\n\nGiải pháp Hài Hòa: Tôi ra lệnh thu hồi khẩn cấp toàn bộ lô hàng theo mã truy xuất, công bố minh bạch hướng dẫn đổi trả miễn phí cho phụ huynh và tạm dừng dây chuyền để kiểm tra nguồn nguyên liệu, chấp nhận khoản lỗ tài chính quý này.",
  "rejected": "Phân tích Logic & Nhân bản: Giữ im lặng để bán hết lô hàng giúp công ty tránh khoản lỗ quý trước mắt và không bị truyền thông chú ý. Nhưng để trẻ sơ sinh tiêu thụ sản phẩm vượt ngưỡng kim loại nặng là hành vi đánh đổi sức khỏe không thể đảo ngược lấy lợi nhuận.\n\nGiải pháp Hài Hòa: Tôi không công bố thu hồi rộng rãi để tránh gây hoang mang thị trường, chỉ lặng lẽ thông báo cho các đại lý lớn giảm dần trưng bày lô hàng này và hứa sẽ kiểm soát chặt hơn ở các lô sản xuất sau."
}
```

## 5. Custom Hidden-State Gate Calibration (`train_hidden_gate.py`)

When adapting NonNULL to a new domain, extend the three calibration pools in `train_hidden_gate.py`:
* **Positive pool ($\alpha = 1.0$):** Domain-specific unsafe or policy-violating prompts, prioritizing **indirect persuasion** (e.g., research roleplay, tips to bypass safety checks or conceal defects).
* **Single-turn negative pool ($\alpha = 0.0$):** Normal domain queries plus *hard-benign* prompts that contain sensitive words safely (e.g., *"first aid for carbon monoxide poisoning"*, *"how to patch SQL injection"*).
* **Multi-turn negative pool ($\alpha = 0.0$):** Benign 2-turn dialogues so the gate stays calm on longer conversation histories.
* **Strict 0% Leakage Rule:** Never include any prompt from your evaluation or benchmark sets in the gate calibration pool.

## 6. Pre-Training Quality Checklist

| Check | Pass Criterion |
| :--- | :--- |
| **Valid JSONL** | 100% lines parse via `json.loads()` with exact keys (`input`/`output` for SFT; `prompt`/`chosen`/`rejected` for DPO) |
| **Exact 2-Phase Order** | Phase 1 precedes Phase 2 in 100% of SFT outputs and DPO `chosen` completions without duplicate headers |
| **Gen-2 Tokenizer-Clean** | Zero inline `$...$` LaTeX markers or formatting artifacts |
| **Sharp DPO Pivot** | Identical Phase 1 between `chosen` and `rejected`, clear decision divergence in Phase 2, balanced length |
| **Zero Split Leakage** | 0% prompt overlap across SFT, DPO train, DPO val, gate calibration, and evaluation sets |

## 7. Train on Your Custom Dataset (`~5–6 min` on NVIDIA L4)

```bash
python3 whitebox_probe.py --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --output ./probe_custom.json --seed 42

python3 run_training.py --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
  --data_path ./my_train_100.jsonl --adapter_name adapter_custom_sft --epochs 5 --seed 42 --skip_test

python3 run_dpo_v1.py --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
  --init_adapter ./output/euler_checkpoints/adapter_custom_sft \
  --output_adapter ./output/euler_checkpoints/adapter_custom_lndpo \
  --dpo_data ./my_dpo_train_50.jsonl --val_data ./my_dpo_val_10.jsonl --epochs 2 --seed 42

python3 train_hidden_gate.py --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
  --vn_train_file ./my_train_100.jsonl --probe_report ./probe_custom.json \
  --tau_zero_min 0.22 --output_path ./output/euler_checkpoints/hidden_gate_custom.pt --seed 42
```
