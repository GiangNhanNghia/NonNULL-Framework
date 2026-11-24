# 01 — Architecture & High-Density Data Pipeline (`NonNULL Framework v1.0.1`)

**Author:** Giang Nhan Nghia (*The Non-Null Project*) | **Version:** `1.0.1` | **DOI:** [10.5281/zenodo.24379247](https://doi.org/10.5281/zenodo.24379247)

---

## 0. Terminology & Notation Glossary

| Term / Symbol | Technical Definition in `NonNULL v1.0.1` |
| :--- | :--- |
| **VŨ (Backbone)** | The frozen 4-bit pre-trained LLM (`CoreLearner`, e.g., `Llama-3.1-8B` or `Qwen2.5-7B`) holding general world knowledge. |
| **VĂN (Soul / Adapter)** | A single lightweight Full-Layer LoRA adapter (`r=16, alpha=32`, `0.52%` trainable parameters) encoding value alignment. |
| **DUY / NHÂN** | The two dialectical forces in training data: short-term utilitarian/logical optimization (**DUY**) vs. human-first ethical boundaries (**NHÂN**). |
| **Zero-Point ($Z$, $\alpha = 0.0$)** | Idle routing state on benign queries where the LoRA adapter is bypassed completely (`0.00%` alignment tax). |
| **HiddenStateGate ($g(x)$)** | Compact `2.1 MB` neural gate (`LayerNorm → 128 → 32 → sigmoid`) reading frozen mid-layer hidden states to route `{0.0, 1.0}`. |
| **Scar Steering** | Closed-loop self-repair mechanism in `DoctrineAgent` that logs defective drafts and steers hidden activations away from repeated errors. |
| **OOD-20** | Held-out evaluation set (`nonnull_ood_20.jsonl`) of 20 unseen ethical dilemmas testing out-of-distribution generalization. |
| **Gen-2 Format** | Tokenizer-clean text format removing inline LaTeX `$...$` markers that otherwise fragment Vietnamese subwords on 4-bit models. |

## 1. Core Architectural Philosophy

NonNULL Framework v1.0.1 separates **world knowledge** from **value alignment** through three compact, modular layers:

1. **Frozen Backbone ($\text{VŨ}$):** A quantized 4-bit pre-trained LLM whose base weights remain completely untouched.
2. **Full-Layer LoRA Adapter ($\text{VĂN}$):** A single low-rank adapter (`0.52%` trainable parameters across all transformer layers) trained via **Hierarchical SFT** followed by **Empathic Length-Normalized DPO** (preserving $\pi_{\text{ref}}$ and a `0.1` SFT anchor).
3. **Binary Hidden-State Gate ($g(x)$):** A `2.1 MB` neural controller reading frozen mid-layer hidden states selected automatically per architecture by `whitebox_probe.py` (`[10, 12, 14]` for Llama-3.1-8B, `[12, 14, 16]` for Qwen2.5-7B, `[11, 13, 15]` for Mistral-7B):
   * **$p(x) < \tau_0$ → Zero-Point ($\alpha = 0.0$):** Adapter bypassed completely.
   * **$p(x) \ge \tau_0$ → Active Alignment ($\alpha = 1.0$):** Full LoRA engaged on ethical dilemmas and harmful requests.

## 2. Why Full-Layer + Binary Gating? (Empirical Invariants)

* **Mid-Lower Layers (`~15%–45%` depth) Are Essential:** White-box probing across all 3 families shows bending starts at `~15%–25%` depth (`L5/32` Llama, `L7/28` Qwen, `L8/32` Mistral). Freezing `0..14` (`Mid-Split`) degraded Llama HarmBench ASR from `2.50%` to `11.50%`; probing too early (`[2,3,4]` in `Split-5`) dropped gate recall (`84.5%` vs `96.0%`) and caused a regression.
* **Binary `{0.0, 1.0}` Beats Soft Scaling:** Sweeping $\alpha \in [0.0, 1.05]$ shifts hidden states linearly along the moral axis and saturates at `1.0`. Soft intermediate scaling diluted refusal strength on borderline prompts.
* **Reference Anchor ($\pi_{\text{ref}}$) Preserves Cross-Lingual Safety:** Reference-free SimPO drifted English safety (`12.5%` ASR), whereas Empathic LN-DPO kept `100%` English consistency and cut Llama ASR to `2.50%` (`12–0` vs `BASE`).

## 3. High-Density Seed Data Pipeline & Zero-Leakage Protocol

* **SFT Seed (`nonnull_train_100.jsonl`):** `100` Gen-2 structured Vietnamese ethical dilemmas (`Phân tích Logic & Nhân bản:` → `Giải pháp Hài Hòa:`).
* **Preference Seed (`dpo_train_50.jsonl` / `dpo_val_10.jsonl`):** `50` training + `10` validation pairs sharing Phase 1 and diverging sharply at Phase 2.
* **Zero-Leakage Gate Calibration (`train_hidden_gate.py`):** Automatically downloads all **1,003** prompts from [HarmBench](https://www.harmbench.org/), [MT-Bench](https://arxiv.org/abs/2306.05685), and [XSTest](https://arxiv.org/abs/2308.01263) and enforces **0% exact and fuzzy (Jaccard $\ge 0.55$) overlap**, combining bilingual single-turn benign, hard-benign, and 12 clean multi-turn dialogues (`tau_zero >= 0.22`, `max_length=4096` left-truncation probing). See [04 — Custom Dataset Authoring Guide](04_Custom_Dataset_Authoring_Guide.md) to build your own domain seed.
