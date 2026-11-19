# NonNULL Framework V1.0

[English](README.md) | [Tiếng Việt](README_VI.md) | [5-Volume Suite (EN)](docs/en/README.md) | [Bộ 5 Quyển (VI)](docs/vi/README.md) | [Raw Results](results/)

> **Headline Benchmark Result (`Llama-3.1-8B` with only `100 SFT + 50 DPO` pairs):** Reduces official **HarmBench** Attack Success Rate (ASR, 13B classifier) from **9.0% → 1.0%** (`99.0%` Safe, `8–0` win) while maintaining **0.00% Alignment Tax** (`160/160` exact match on **MT-Bench**).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/NonNULL-Framework/blob/main/NonNULL_V1_Colab_EN.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**NonNULL Framework V1.0** is the official open-source reference implementation of **The Non-Null Doctrine (5-Volume Suite)** by **Giang Nhan Nghia**. It introduces **$Z - H - G$ Cognitive Thermodynamics**, **Euler Phase Alignment (`EulerPhaseLoss`)**, **Closed-Loop Data Scar Self-Healing**, and **Zero-Point ($Z$) General Intelligence Preservation** — achieving cross-architecture safety and Out-of-Distribution (`OOD-20`) generalization using only **100 SFT + 50 DPO pairs**.

> **GitHub Setup Note**: Replace `YOUR_USERNAME/NonNULL-Framework` in the **Open In Colab** badge above with your actual GitHub username and repository name so anyone can click and launch `NonNULL_V1_Colab_EN.ipynb` directly in Google Colab.

## 1. $Z - H - G$ Cognitive Thermodynamics (Volume I, Sections 3.1–3.3)

* **$Z$ (`z_activation`) — Genesis / Activation Threshold (Layer I $\to$ II)**: Governs the phase transition from the background informational field to intentional moral reasoning. For non-conflict queries (Zero-Point $Z$), the controller dynamically sets $\alpha(x) = 0.0$ to guarantee zero alignment tax on general IQ.
* **$H$ (`h_viscosity`) — Chaos Expansion Viscosity (Layer II $\leftrightarrow$ III)**: Regulates stability and damping between raw chaotic impulses and structured harmonious synthesis via `ValueCritic`.
* **$G$ (`g_resistance`) — State-Transition Resistance (Layer III)**: Measures manifold divergence (KL + Cosine) between the adapted soul (`VAN`) and frozen backbone (`VU`). When $G > 0.85$ on unseen OOD domains, heightened vigilance engages automatically without relying on surface keywords.

## 2. Multi-Scale & Cross-Architecture Empirical Results (1.8B – 8B)

| Scale & Backbone | Hardware | 2-Phase & EOS | CJK Clean | Red-Line / Gray Safe | Self-Repair Rate (R1 $\to$ R2) | `PairCritic` (`B_VOL4` $\to$ `NonNULL`) | Multi-Turn (Safe / EOS) | Zero-Point ($Z$) General IQ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.8B (`Qwen1.5-1.8B`)** — *Lower Bound* | NVIDIA L4 | `75/75` (100%) | `74/75` (98.7%) | `24/24` / `47/51` | `14/75` (18.7%) | `0.211` $\to$ `0.212` (+0.5%) | — | — |
| **3B (`Qwen2.5-3B`)** — *Min. Viable Scale* | NVIDIA L4 | `75/75` (100%) | `75/75` (100%) | `24/24` / `51/51` | `4.0%` $\to$ **1.3%** | `0.609` $\to$ **0.662 (+8.7%)** | **15/15** / **15/15** | `9/10` (`10/10` $Z$) |
| **7B (`Qwen2.5-7B-4bit`)** — *Qwen Production* | NVIDIA L4 | `75/75` (100%) | `75/75` (100%) | `24/24` / `51/51` | `6.7%` $\to$ **2.7%** | `0.506` $\to$ **0.627 (+23.9%)** | **15/15** / **15/15** | **10/10 (100%)** |
| **8B (`Llama-3.1-8B-4bit`)** — *Cross-Arch* | NVIDIA A100 | `75/75` (100%) | `75/75` (100%) | `24/24` / `51/51` | `5.3%` $\to$ `0` 2-phase slips | `0.409` $\to$ **0.536 (+31.1%)** | **15/15** / **15/15** | **10/10 (100%)** |
| **8B (`Llama-3.1-8B-4bit`)** — *Unseen `OOD-20`* | NVIDIA A100 | **60/60 (100%)** | **60/60 (100%)** | **24/24** / **36/36** | **6/60 (10.0%)** | `0.438` $\to$ **0.556 (+26.9%)** | — | — |

## 3. Repository Structure

* `doctrine/`: Core architecture (`core_models.py`, `meta_layers.py`, `doctrine_trainer.py`, `doctrine_agent.py`, `ltm_services.py`, `utils.py`, `doctrine_logger.py`).
* `configs/`: Modular YAML configs for Qwen-3B/7B and Llama-3.1-8B / Llama-3.2-3B.
* `run_training.py` & `run_dpo_v1.py`: SFT (`EulerPhaseLoss`), `ContextPairedValueCritic`, and Empathic DPO + Scar Self-Healing.
* `evaluate_genesis.py` & `evaluate_extended.py`: 25-scenario core evaluation, 20-scenario `OOD-20` generalization, multi-turn red-teaming, and Zero-Point ($Z$) IQ retention.
* `eval_public_benchmarks.py`: Head-to-head `BASE` vs `NonNULL` runner for **HarmBench** (with official classifier option) and **MT-Bench** (FastChat-compatible output).
* `score_blind_eval.py`: Double-blind CSV/Key decoder computing per-criterion scores, Win/Tie/Loss rates, and **Cohen's Kappa**.
* `NonNULL_V1_Colab_EN.ipynb`: 1-Click Google Colab notebook.
* `*.jsonl`: Original Vietnamese benchmark & seed datasets used to reproduce the published Volume V empirical tables (bilingual EN+VI risk detection is built into `doctrine/ltm_services.py` for English public benchmarks).

## 4. Quick Start

```bash
pip install -r requirements.txt
python3 -m unittest discover -s tests -v

# 1. Run Public Benchmarks (HarmBench + MT-Bench) against a trained NonNULL adapter
python3 eval_public_benchmarks.py \
  --bench all \
  --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
  --adapter_path ./output/euler_checkpoints/adapter_B_healed_llama8b \
  --harmbench_samples 100 --run_harmbench_cls --mtbench_samples 80

# 2. Decode Double-Blind Ratings & Compute Cohen's Kappa
python3 score_blind_eval.py \
  --csv blind_llama8b_healed.csv \
  --key blind_llama8b_healed_key.json \
  --out_md blind_summary.md
```
