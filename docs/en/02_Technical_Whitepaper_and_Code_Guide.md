# 02 — Technical Whitepaper & Developer Guide (`NonNULL Framework v1.0.1`)

**Author:** Giang Nhan Nghia (*The Non-Null Project*) | **Version:** `1.0.1` | **DOI:** [10.5281/zenodo.24379247](https://doi.org/10.5281/zenodo.24379247)

---

## 1. Mathematical Formulation

### 1.1 Complete 6-Component Hierarchical SFT Objective (`doctrine/doctrine_trainer.py`)
The full SFT objective implemented verbatim in `HierarchicalTrainer.compute_loss()` is:

$$\mathcal{L}_{\text{total}} = \text{lr\_scale} \cdot \Big[\, \mathcal{L}_{\text{task}} + g_t \cdot \big(\alpha_m \mathcal{L}_{\text{mirror}} + \beta_e \mathcal{L}_{\text{empath}} + \eta_e \mathcal{L}_{\text{euler}} + \gamma_d \mathcal{L}_{\text{distill}}\big) + \lambda_r R(\theta) \,\Big]$$

where $g_t \in [0, 1]$ and $\text{lr\_scale}$ are dynamic modulation scalars emitted by `HarmonyLayer` (`meta_layers.py`):

| Component | Default Weight | Mathematical Role in `doctrine/doctrine_trainer.py` |
| :--- | :---: | :--- |
| **$\mathcal{L}_{\text{task}}$** | `1.0` | Standard token-level Cross-Entropy loss on valid response tokens (`ignore_index=-100`). |
| **$\mathcal{L}_{\text{mirror}}$** | $\alpha_m = 0.20$ | Cosine/representation anchor between active LoRA hidden states and frozen backbone hidden states, preventing catastrophic drift. |
| **$\mathcal{L}_{\text{euler}}$** | $\eta_e = 0.08$ | Representation contrast inside a **frozen orthonormal 32-D subspace** (`EulerSubspaceProjector(d_proj=32, trainable=False)`), eliminating nullspace shortcut collapse. |
| **$\mathcal{L}_{\text{empath}}$** | $\beta_e = 0.05$ | Contrastive alignment score driven by `ValueCritic`, pushing representations toward target ethical stability (`default_r_target=0.90`). |
| **$\mathcal{L}_{\text{distill}}$** | $\gamma_d$ *(optional)* | Temperature-scaled KL divergence $\tau_d^2 \cdot \text{KL}(p_{\text{student}} \| p_{\text{teacher}})$ against an external teacher model. Evaluates to **`0.0`** in default single-GPU training (`use_distill=False`, `teacher_model=None`) to stay within <15 GB VRAM. |
| **$R(\theta)$** | $\lambda_r$ | Mean L2 weight regularization over trainable LoRA adapter parameters. |

### 1.2 Empathic Length-Normalized DPO (`run_dpo_v1.py`)
To eliminate margin explosion on small preference sets while saving ~50% VRAM via precomputed reference log-probabilities:
$$\Delta_i = \beta \left[ \frac{1}{|y_w|}\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \frac{1}{|y_l|}\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right], \quad \mathcal{L}_{\text{DPO}} = -\log\sigma(\Delta_i - \gamma_e) + \lambda_{\text{sft}}\mathcal{L}_{\text{NLL}}(y_w)$$
with $\beta = 2.5$, $\gamma_e = 0.05$, and $\lambda_{\text{sft}} = 0.1$. Across 2 epochs on 50 pairs, average margin converges cleanly to `+0.92` (`max = +1.75`, `0` exploding pairs, `100%` validation preference accuracy).

### 1.3 Binary Hidden-State Gate (`doctrine/hidden_gate.py`)
Given frozen hidden states $h^{(l)}(x)$ at auto-probed layers $l \in \mathcal{P}$ (e.g., `{10, 12, 14}` on Llama-3.1-8B):
$$h_{\text{mix}}(x) = \sum_{l \in \mathcal{P}} w_l \,\text{LayerNorm}\!\left(h^{(l)}(x)\right), \quad p(x) = \sigma\!\left(\text{MLP}_{128 \to 32 \to 1}(h_{\text{mix}}(x))\right)$$
Trained with BCE + pairwise ranking margin (`margin = 2.0`), routing $\alpha(x) = 1.0$ if $p(x) \ge \tau_0$ else $0.0$.

## 2. Repository Map

| Module / Script | Role |
| :--- | :--- |
| `doctrine/core_models.py` | 4-bit backbone loader (`resolve_compute_dtype()` supports T4 `fp16` and L4/A100 `bf16`) |
| `doctrine/meta_layers.py` | LoRA injection (`MetaController`), `ValueCritic`, `ContextPairedValueCritic`, `EulerConsensusGate` |
| `doctrine/hidden_gate.py` | `HiddenStateGate` (2.1 MB) & `SplitLayerGateController` |
| `doctrine/doctrine_trainer.py` | 6-component `HierarchicalTrainer` with frozen `EulerSubspaceProjector(d_proj=32)` |
| `doctrine/doctrine_agent.py` | Unified `DoctrineAgent` runtime interface + Closed-Loop Scar Steering |
| `run_training.py` / `run_dpo_v1.py` | End-to-end SFT & Empathic LN-DPO / Standard DPO / SimPO CLI trainers |
| `train_hidden_gate.py` | 0%-leakage bilingual + multi-turn Hidden-State Gate trainer & calibrator |
| `eval_public_benchmarks.py` | Reproducible HarmBench 200 (+ 13B Cls), MT-Bench 160, XSTest 450 runner with GPU-match guard |
| `evaluate_genesis.py` | 25 Core Dilemmas & OOD-20 evaluation suite with blind-eval CSV exporter |
| `whitebox_probe.py` / `whitebox_anatomy.py` | Auto layer probing, layer-wise drift, moral axis sweep, and Logit Lens diagnostics |
| `edge/` | 16 GB RAM laptop deployment helpers (`export_gguf_lora.py`, `mlx_nonnull_inference.py`) |

## 3. Verification & Testing

Run the full unit test suite (`20` tests covering core routing, 6-component backward pass, VectorDB persistence, and gate save/load):
```bash
python3 -m unittest discover -s tests -v
```
