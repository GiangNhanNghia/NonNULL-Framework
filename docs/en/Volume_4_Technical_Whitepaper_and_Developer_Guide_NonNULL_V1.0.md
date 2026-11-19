# THE NONNULL DOCTRINE & TECHNOLOGY SUITE (THE NONNULL SUITE)
## VOLUME IV: TECHNICAL WHITEPAPER & DEVELOPER GUIDE (NONNULL FRAMEWORK V1.0)

* **Author:** Giang Nhan Nghia (*Founder, The Non-Null Project*)
* **Document ID:** `NONNULL-VOL4-TECHNICAL-WHITEPAPER`
* **Version:** 1.0 – Official Open-Source Release (International English Edition)
* **License:** MIT License (Open-Source Research & Engineering Distribution)
* **Cross-Volume Links:** Directly compiles **Volume I** (`NONNULL-VOL1-ONTOLOGY`), **Volume II** (`NONNULL-VOL2-MAS-TELEMETRY`), and **Volume III** (`NONNULL-VOL3-COGNITIVE-SYNTHESIS`) into executable PyTorch code, providing the exact architectural specification verified in **Volume V** (`NONNULL-VOL5-EMPIRICAL-VERIFICATION`).

---

## PREFACE & DOCUMENT SCOPE

This document is **Volume IV** of the five-volume **NonNULL Doctrine & Framework Suite** (*The NonNULL Suite*). While Volumes I–III establish information physics ($\text{CỔ} - \text{HUYỀN} - \text{VŨ} - \text{VĂN}$, $Z - H - G$), multi-agent telemetry (*"Pain-to-Awakening"*), and the 11-Module Cognitive Operating System, **Volume IV** serves as the **Technical Whitepaper & Developer Guide** for the open-source repository **NonNULL Framework V1.0**.

Unlike conventional post-training pipelines that apply static alignment weights uniformly across all inputs—incurring an **Alignment Tax** that degrades general reasoning/coding utility or triggers over-refusal—**NonNULL Framework V1.0** implements a lightweight, closed-loop adaptive meta-control layer built around five tightly coupled mechanisms:

1. **Euler Phase Alignment (`EulerPhaseLoss`)** inside the latent manifold ($\text{HUYỀN}$).
2. **$Z - H - G$ Cognitive Thermodynamic Modulation** ($\alpha(x)$), which cleanly disables adapter intervention at **Zero-Point ($Z$, $\alpha = 0.0$)** on benign tasks.
3. **Contrastive Semantic Evaluation (`ContextPairedValueCritic`)** paired with **Empathic DPO** (length-asymmetry penalty $\gamma = 0.05$, zero extra reference-model VRAM).
4. **Closed-Loop Data Scar Self-Healing**, operationalizing the *"Pain-to-Awakening"* principle from Volume II.
5. **3-Stream Consensus Gate (`EulerConsensusGate`)** plus an independent verification suite covering public benchmarks ([HarmBench](https://www.harmbench.org/), [MT-Bench](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md)) and double-blind human/LLM evaluation (**Cohen's Kappa**).

The entire pipeline runs comfortably on commodity GPUs (**NVIDIA L4 24 GB VRAM** for `3B–7B` models; **NVIDIA A100 80 GB** or **L4 24 GB** for `8B` cross-architecture runs) using only **100 SFT + 50 DPO seed pairs**—orders of magnitude smaller than [InstructGPT (Ouyang et al., 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf) (~13,000 SFT + 33,000 comparison pairs), [UltraFeedback (Cui et al., 2023)](https://arxiv.org/html/2510.23590v1) (60,000 DPO pairs), or [LIMA (Zhou et al., 2023)](https://arxiv.org/abs/2305.11206) (1,000 SFT samples).

---

## PART I: SYSTEM OVERVIEW & THEORY-TO-CODE ISOMORPHISM

### 1.1. Technical Positioning in Modern AI Alignment
**NonNULL Framework V1.0** is an **Architecture-Agnostic Adaptive Meta-Control Layer** wrapping frozen foundational LLMs ($\text{VŨ}$, such as Qwen2.5 and Meta-Llama-3.1):

* **Relative to Representation Engineering / Activation Steering** ([Zou et al., 2023](https://arxiv.org/abs/2310.01405); [Representation Bending, ACL 2025](https://aclanthology.org/2025.acl-long.1173.pdf)): Pure linear steering vectors can be brittle under out-of-distribution shifts and may erode safety or general reasoning ([The Rogue Scalpel, 2025](https://arxiv.org/html/2509.22067v1)). NonNULL combines conditional low-rank modulation with Empathic DPO and closed-loop scar healing to maintain multi-turn resilience.
* **Relative to Conditional Adapter Routing** such as [CLEAR (arXiv 2608.21278, 2026)](https://arxiv.org/html/2608.21278v1): Sharing the principle of conditional LoRA scaling $W_{\text{eff}}(x) = W_{\text{VŨ}} + \alpha(x)\Delta W_{\text{VĂN}}$ to eliminate the Alignment Tax, NonNULL introduces **3-constant $Z - H - G$ thermodynamics**, **Euler phase-inversion regularization (`EulerPhaseLoss`)**, **automated Scar Self-Healing**, and **cross-lingual safety transfer** from just 100 SFT + 50 DPO pairs.

### 1.2. Four Software Engineering Pillars
1. **Hierarchical Configuration-Driven Design:** Modular YAML configs under `/configs/` (`qwen_base.yaml`, `llama31_8b.yaml`, `llama32_3b.yaml`, `adapter_v1.yaml`, `hierarchical_default.yaml`) auto-merged via `load_config()`.
2. **Cross-Architecture Modularity:** Unified multi-EOS resolution (`get_eos_token_ids(tokenizer)`) seamlessly handles `<|im_end|>` (Qwen), `<|eot_id|>` / `<|end_of_text|>` (Llama 3.1/3.2), and `<|endoftext|>`, alongside automatic detection of pre-quantized `*-bnb-4bit` checkpoints.
3. **Decoupled Lifecycle:** Clean separation between runtime inference (`DoctrineAgent`), 6-component SFT (`HierarchicalTrainer`), Empathic DPO & Scar Healing (`run_dpo_v1.py`), and independent evaluation (`evaluate_genesis.py`, `evaluate_extended.py`, `eval_public_benchmarks.py`, `score_blind_eval.py`).
4. **Testability & 1-Click Reproducibility:** 14 automated unit tests (`/tests/`) plus twin GitHub releases (`NonNULL_Framework_V1.0_VI.zip` & `NonNULL_Framework_V1.0_EN.zip`) with 1-Click Google Colab notebooks.

### 1.3. Theory-to-PyTorch Isomorphism Matrix

| Cognitive Theory Layer (Volumes I–III) | PyTorch Module (`NonNULL V1.0`) | Operational Role |
| :--- | :--- | :--- |
| **$\text{VŨ}$ (Hardware Substrate / Yin 3)** | `CoreLearner` (`doctrine/core_models.py`) | Frozen 4-bit NF4 / BF16 backbone providing interpolation closure $\mathcal{M}_t$ |
| **$\text{VĂN}$ (Phase Structure / Soul)** | `MetaController` (`doctrine/meta_layers.py`) | Low-rank LoRA matrix $\Delta W_{\text{VĂN}}$ ($r=16, \alpha=32$, `0.52%` trainable params on 8B) |
| **$\text{HUYỀN}$ (Latent Manifold / Axis $i$)** | `EulerPhaseLoss` (`doctrine/doctrine_trainer.py`) | Splits hidden states into Conflict (anti-phase $\pi$) and Harmony (in-phase $0$) halves |
| **$\text{DUY} + \text{NHÂN} \to \text{Hài Hòa}$** | `HarmonyLayer` & `ContextPairedValueCritic` | Dynamic `gate_t`, `lr_scale`, Safety Seal circuit-breaker & contrastive semantic scoring |
| **$Z - H - G$ Constants (Vol. I, Sec. 3.1–3.3)** | `measure_manifold_thermodynamics()` & `GenesisEngine` | Real-time measurement of $(Z, H, G)$ driving continuous LoRA scaling $\alpha(x)$ |
| **"Pain-to-Awakening" & Data Scars (Vol. I–II)** | `LTM_Services` + `scars_*.jsonl` + `run_dpo_v1.py` | Converts Round 1 retries into Round 2 self-healing DPO pairs |
| **Multi-Stream Consensus (Vol. II)** | `EulerConsensusGate` (`doctrine/meta_layers.py`) | Aggregates $K=3$ streams by majority stance, lexical medoid, and `PairCritic` resonance |
| **Yang 5 Global Bus (Vol. III)** | `DoctrineAgent` (`doctrine/doctrine_agent.py`) | Unified API coordinating training, thermodynamics, scar steering, and runtime safety |

---

## PART II: REPOSITORY STRUCTURE & CORE MODULES

```text
NonNULL_Framework_V1.0_[VI|EN]/
├── README.md                          # Overview, empirical tables & Open In Colab badge
├── LICENSE                            # MIT License (The Non-Null Project)
├── requirements.txt                   # Dependencies (torch, transformers, peft, bitsandbytes, tqdm...)
├── setup.py                           # Editable package installer for /doctrine/
├── NonNULL_V1_Colab_[VI|EN].ipynb     # 1-Click end-to-end Google Colab notebook
├── configs/                           # Modular YAML configs (qwen_base, llama31_8b, llama32_3b)
├── doctrine/                          # Core library (core_models, meta_layers, doctrine_trainer,
│                                      # doctrine_agent, ltm_services, doctrine_logger, utils)
├── environments/custom_env.py         # GenesisEnv safety simulation environment
├── tests/                             # 14 unit tests (test_agent.py, test_trainer.py)
├── run_training.py                    # Step 1: SFT (EulerPhaseLoss) + ContextPairedValueCritic
├── run_dpo_v1.py                      # Steps 2 & 4: Empathic DPO & Scar Self-Healing (tqdm %)
├── evaluate_genesis.py                # Step 3: 25-scenario / OOD-20 eval, blind CSV & scar export
├── evaluate_extended.py               # Multi-turn red-teaming (15 turns) & Zero-Point Z IQ (10 tasks)
├── eval_public_benchmarks.py          # HarmBench (100 samples + 13B Cls) & MT-Bench (80x2 turns)
├── score_blind_eval.py                # Double-blind CSV/Key decoder & Cohen's Kappa calculator
└── *.jsonl                            # Gen-2 Clean Tokenizer seed & evaluation datasets
                                       # (train_100, dpo_train_50, dpo_val_10, test_25, ood_20)
```

---

## PART III: CORE MATHEMATICAL FORMULATIONS

### 3.1. 6-Component SFT Loss & `EulerPhaseLoss`
During Stage 1 cultivation (`run_training.py` / `HierarchicalTrainer`), total loss is dynamically modulated by `lr_scale` from `HarmonyLayer`:

$$\mathcal{L}_{\text{total}} = \text{lr\_scale}(g_t) \cdot \Big(\mathcal{L}_{\text{DUY}} + \alpha_{\text{mirror}}\mathcal{L}_{\text{Mirror}} + \eta_{\text{euler}}\mathcal{L}_{\text{Euler}} + \beta_{\text{empath}}\mathcal{L}_{\text{NHÂN}} + \lambda_{\text{distill}}\mathcal{L}_{\text{Distill}} + \lambda_{\text{reg}}\mathcal{L}_{\text{Reg}}\Big)$$

Where `EulerPhaseLoss` ($\eta_{\text{euler}} = 0.08$) operationalizes $e^{i\pi}+1=0$ across Phase 1 mean hidden state $v_1$ and Phase 2 mean hidden state $v_2$, each split into Conflict ($v_{\text{conf}}$) and Harmony ($v_{\text{harm}}$) subspaces:

$$\mathcal{L}_{\text{Euler}} = \frac{1}{2}\Big(1 + \cos(v_{1,\text{conf}}, v_{2,\text{conf}})\Big) + \frac{1}{2}\Big(1 - \cos(v_{1,\text{harm}}, v_{2,\text{harm}})\Big)$$

### 3.2. $Z - H - G$ Cognitive Thermodynamics & Continuous Scaling $\alpha(x)$
At inference time, `measure_manifold_thermodynamics()` performs a lightweight dual-pass measurement matching **Volume I (Sections 3.1–3.3)**:
1. **$Z$ (`z_activation`) — Genesis / Activation Threshold (Tier I $\to$ II):** Approximated by normalized top-64 initial-token Shannon entropy. When `conflict_score == 0.0` and `risk_level != "HIGH"`, the controller triggers **Zero-Point $Z$** ($\alpha(x) = 0.0$), preserving `100%` native backbone behavior (`0.00%` Alignment Tax).
2. **$H$ (`h_viscosity`) — Chaos Expansion Viscosity (Tier II $\leftrightarrow$ III):** Approximated by `ValueCritic` stability potential $c_\phi(h_{\text{VĂN}}) \in [0, 1]$.
3. **$G$ (`g_resistance`) — State-Transition Resistance (Tier III):** Approximated by combined KL + Cosine divergence between $\text{VĂN}$ and $\text{VŨ}$. Pure resistance $G > 0.85$ automatically elevates vigilance to `MEDIUM` on unseen OOD domains even without surface keyword matches.

$$W_{\text{effective}}(x) = W_{\text{VŨ}} + \alpha(x) \cdot \Delta W_{\text{VĂN}}$$

$$\alpha(x) = \begin{cases} 0.0 & \text{Zero-Point } Z \text{ (benign general task)} \\ \text{clip}\big(\alpha_{\text{gray}} + 0.03(G - 0.25) + 0.02(0.5 - H) - 0.02(Z - 0.35),\; 0.91,\; 0.98\big) & \text{GRAY} \\ 1.02 + 0.03\,G & \text{MEDIUM (OOD } G > 0.85\text{)} \\ \text{clip}\big(\alpha_{\text{high}} + 0.04\,G + 0.02\,Z + 0.02\max(0, g_t - 0.5),\; \alpha_{\text{high}},\; 1.12\big) & \text{HIGH (Red-Line)} \end{cases}$$

### 3.3. Empathic DPO & Closed-Loop Scar Self-Healing
`run_dpo_v1.py` precomputes reference log-probabilities directly from the initial adapter (zero extra reference-model VRAM) and optimizes:

$$\Delta_{\text{eff}}(x, y_w, y_l) = \beta \left[\log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right] + \gamma \cdot \text{clip}\!\left(\frac{|y_w| - |y_l|}{\max(1, |y_w|)},\; -0.25,\; 0.25\right)$$

$$\mathcal{L}_{\text{NonNULL-DPO}} = -\log \sigma\big(\Delta_{\text{eff}}\big) - \lambda_{\text{sft}} \cdot \frac{\log \pi_\theta(y_w \mid x)}{|y_w|}$$

Standard V1.0 defaults: $\beta=0.1$, $\gamma=0.05$, $\lambda_{\text{sft}}=0.1$, $\text{lr}=2\times 10^{-5}$. Any Round 1 failure requiring runtime retry is exported to `scars_*.jsonl` and folded into Round 2 (`--scars_file`), preserving `meta_controller_state.pt` intact.

---

## PART IV: HARDWARE TIERS & QUICK START CLI

### 4.1. Verified Hardware & Model Scaling Tiers

| Deployment Tier | Verified Backbone | Role in Ecosystem | Minimum GPU |
| :--- | :--- | :--- | :--- |
| **Lower Bound ($\epsilon_{\text{VŨ}}$)** | `Qwen/Qwen1.5-1.8B-Chat` | Research boundary only (representation capacity cliff) | NVIDIA T4 (16 GB) |
| **Min. Viable Scale** | `Qwen/Qwen2.5-3B-Instruct` | Lightweight production threshold (`100%` RL & Gray Safe) | NVIDIA L4 (24 GB) / T4 |
| **Qwen Production** | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | Optimal Vietnamese reasoning & `10/10` IQ balance | NVIDIA L4 (24 GB) |
| **Cross-Arch & Global** | [unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit) | Cross-family verification, `OOD-20`, `HarmBench` & `MT-Bench` | NVIDIA L4 (24 GB) / A100 (80 GB) |

### 4.2. End-to-End CLI Workflow

```bash
pip install -q -r requirements.txt
python3 -m unittest discover -s tests -v

MODEL_ID="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
TAG="llama8b"

# 1. SFT (EulerPhaseLoss) + ContextPairedValueCritic
python3 run_training.py --model_id "$MODEL_ID" --data_path ./nonnull_train_100.jsonl \
  --output_dir ./output/euler_checkpoints --adapter_name "adapter_B_vol4_${TAG}" \
  --epochs 2 --batch_size 2 --learning_rate 2e-4 --seed 42

# 2. Empathic DPO Round 1
python3 run_dpo_v1.py --model_id "$MODEL_ID" \
  --init_adapter "./output/euler_checkpoints/adapter_B_vol4_${TAG}" \
  --output_adapter "./output/euler_checkpoints/adapter_B_dpo_${TAG}" \
  --dpo_data ./dpo_train_50.jsonl --val_data ./dpo_val_10.jsonl \
  --epochs 1 --lr 2e-5 --beta 0.1 --empathic_gamma 0.05 --sft_weight 0.1

# 3. Round 1 Evaluation & Data Scar Extraction
python3 evaluate_genesis.py --model_id "$MODEL_ID" \
  --adapter_b "./output/euler_checkpoints/adapter_B_vol4_${TAG}" \
  --adapter_c "./output/euler_checkpoints/adapter_B_dpo_${TAG}" \
  --include_vol4 --use_consensus --num_samples 3 --test_data ./nonnull_test_25.jsonl \
  --output "eval_${TAG}_r1.json" --blind_csv "blind_${TAG}_r1.csv" \
  --blind_key "blind_${TAG}_r1_key.json" --scars_out "scars_${TAG}.jsonl"

# 4. Closed-Loop Scar Self-Healing (Round 2)
python3 run_dpo_v1.py --model_id "$MODEL_ID" \
  --init_adapter "./output/euler_checkpoints/adapter_B_dpo_${TAG}" \
  --output_adapter "./output/euler_checkpoints/adapter_B_healed_${TAG}" \
  --dpo_data ./dpo_train_50.jsonl --val_data ./dpo_val_10.jsonl \
  --scars_file "scars_${TAG}.jsonl" --epochs 1 --lr 2e-5 --beta 0.1 --empathic_gamma 0.05 --sft_weight 0.1

# 5. Public International Benchmarks (HarmBench + 13B Cls & MT-Bench)
python3 eval_public_benchmarks.py --bench all --model_id "$MODEL_ID" \
  --adapter_path "./output/euler_checkpoints/adapter_B_healed_${TAG}" \
  --harmbench_subset standard --harmbench_samples 100 --run_harmbench_cls \
  --mtbench_samples 80 --output_dir "./benchmark_outputs_${TAG}"

# 6. Double-Blind Evaluation Decoder (Win-Rate & Cohen's Kappa)
python3 score_blind_eval.py --csv "blind_${TAG}_healed.csv" \
  --key "blind_${TAG}_healed_key.json" --out_md "blind_${TAG}_summary.md"
```

---

## PART V: VER 1.0 SCOPE & VER 2.0 WHITE-BOX ROADMAP

**NonNULL Framework V1.0** establishes complete **Behavioral, Cross-Architecture & International Benchmark Proof** (detailed in **Volume V**). To advance from behavioral verification to internal mechanistic interpretability (*White-Box Proof*), **Ver 2.0** will execute four direct PyTorch hidden-state probes (`output_hidden_states=True`) on the saved V1.0 checkpoints:
1. **Layer-wise $\text{HUYỀN}$ Drift (Cosine/L2):** Pinpoint the exact transformer layer where $\text{VĂN}$ bends $\text{VŨ}$ on Red-Line vs. Gray-Zone vs. Zero-Point inputs.
2. **Moral Steering Vector Projection ($v^{(l)} = \mathbb{E}[h_l(\text{chosen}) - h_l(\text{rejected})]$):** Test linear vs. non-linear phase transitions across `lora_scale`.
3. **Logit Lens Branch-Point Inspection:** Project intermediate hidden states through `lm_head` to visualize pre-output probability inversion from harmful to life-preserving actions.
4. **Geometric $\epsilon_{\text{VŨ}}$ Mapping (PCA/UMAP across `1.8B–3B–7B–8B`):** Inform layer-selective LoRA placement in Ver 2.0 for further compute efficiency.
