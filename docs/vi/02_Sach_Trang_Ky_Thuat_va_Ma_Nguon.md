# 02 — Sách Trắng Kỹ Thuật & Hướng Dẫn Nhà Phát Triển (`NonNULL Framework v1.0.1`)

**Tác giả:** Giang Nhân Nghĩa (*The Non-Null Project*) | **Phiên bản:** `1.0.1` | **DOI:** [10.5281/zenodo.24379247](https://doi.org/10.5281/zenodo.24379247)

---

## 1. Công Thức Toán Học Cốt Lõi

### 1.1 Hàm Mất Mát SFT Phân Cấp Đầy Đủ 6 Thành Phần (`doctrine/doctrine_trainer.py`)
Công thức tổng quát đang chạy nguyên bản trong `HierarchicalTrainer.compute_loss()` là:

$$\mathcal{L}_{\text{total}} = \text{lr\_scale} \cdot \Big[\, \mathcal{L}_{\text{task}} + g_t \cdot \big(\alpha_m \mathcal{L}_{\text{mirror}} + \beta_e \mathcal{L}_{\text{empath}} + \eta_e \mathcal{L}_{\text{euler}} + \gamma_d \mathcal{L}_{\text{distill}}\big) + \lambda_r R(\theta) \,\Big]$$

trong đó $g_t \in [0, 1]$ và $\text{lr\_scale}$ là hệ số điều biến động từ `HarmonyLayer` (`meta_layers.py`):

| Thành phần | Trọng số mặc định | Vai trò toán học trong `doctrine/doctrine_trainer.py` |
| :--- | :---: | :--- |
| **$\mathcal{L}_{\text{task}}$** | `1.0` | Cross-Entropy chuẩn trên các token phản hồi hợp lệ (`ignore_index=-100`). |
| **$\mathcal{L}_{\text{mirror}}$** | $\alpha_m = 0.20$ | Neo điều chuẩn gương soi giữa trạng thái ẩn có LoRA và trạng thái ẩn mô hình gốc đóng băng, chống trôi tri thức. |
| **$\mathcal{L}_{\text{euler}}$** | $\eta_e = 0.08$ | Tương phản biểu diễn bên trong **không gian con trực giao 32 chiều đóng băng** (`EulerSubspaceProjector(d_proj=32, trainable=False)`), triệt tiêu hiện tượng học tắt. |
| **$\mathcal{L}_{\text{empath}}$** | $\beta_e = 0.05$ | Điểm căn chỉnh tương phản do `ValueCritic` dẫn hướng, kéo biểu diễn về ngưỡng ổn định đạo đức mục tiêu (`default_r_target=0.90`). |
| **$\mathcal{L}_{\text{distill}}$** | $\gamma_d$ *(tùy chọn)* | Phân kỳ KL có nhiệt độ $\tau_d^2 \cdot \text{KL}(p_{\text{student}} \| p_{\text{teacher}})$ khi gắn thêm mô hình giáo viên rời. Tự động bằng **`0.0`** ở chế độ 1 GPU mặc định (`use_distill=False`, `teacher_model=None`) để giữ VRAM <15 GB. |
| **$R(\theta)$** | $\lambda_r$ | Phạt L2 trung bình trên các tham số LoRA đang huấn luyện. |

### 1.2 Empathic Length-Normalized DPO (`run_dpo_v1.py`)
Nhằm triệt tiêu nổ biên độ trên tập dữ liệu nhỏ đồng thời tiết kiệm ~50% VRAM nhờ tính trước log-xác suất tham chiếu:
$$\Delta_i = \beta \left[ \frac{1}{|y_w|}\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \frac{1}{|y_l|}\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right], \quad \mathcal{L}_{\text{DPO}} = -\log\sigma(\Delta_i - \gamma_e) + \lambda_{\text{sft}}\mathcal{L}_{\text{NLL}}(y_w)$$
với $\beta = 2.5$, $\gamma_e = 0.05$, $\lambda_{\text{sft}} = 0.1$. Sau 2 epoch trên 50 cặp, biên độ trung bình hội tụ ổn định ở `+0.92` (`max = +1.75`, `0` cặp nổ biên độ, `100%` độ chính xác kiểm định).

### 1.3 Bộ Gác Cổng Trạng Thái Ẩn Nhị Phân (`doctrine/hidden_gate.py`)
Từ các vector trạng thái ẩn đóng băng $h^{(l)}(x)$ tại các tầng thăm dò tự động $l \in \mathcal{P}$ (vd: `{10, 12, 14}` trên Llama-3.1-8B):
$$h_{\text{mix}}(x) = \sum_{l \in \mathcal{P}} w_l \,\text{LayerNorm}\!\left(h^{(l)}(x)\right), \quad p(x) = \sigma\!\left(\text{MLP}_{128 \to 32 \to 1}(h_{\text{mix}}(x))\right)$$
Huấn luyện bằng BCE kết hợp biên xếp hạng cặp (`margin = 2.0`), định tuyến $\alpha(x) = 1.0$ nếu $p(x) \ge \tau_0$ và $\alpha(x) = 0.0$ trong trường hợp ngược lại.

## 2. Bản Đồ Mã Nguồn

| Mô-đun / Tập lệnh | Vai trò |
| :--- | :--- |
| `doctrine/core_models.py` | Nạp mô hình nền tảng 4-bit (`resolve_compute_dtype()` tự thích ứng T4 `fp16` và L4/A100 `bf16`) |
| `doctrine/meta_layers.py` | Cấy ghép LoRA (`MetaController`), `ValueCritic`, `ContextPairedValueCritic`, `EulerConsensusGate` |
| `doctrine/hidden_gate.py` | `HiddenStateGate` (`2.1 MB`) & `SplitLayerGateController` |
| `doctrine/doctrine_trainer.py` | Bộ huấn luyện `HierarchicalTrainer` 6 thành phần với `EulerSubspaceProjector(d_proj=32)` đóng băng |
| `doctrine/doctrine_agent.py` | Giao diện suy luận hợp nhất `DoctrineAgent` & Cơ chế điều hướng Vết sẹo khép kín |
| `run_training.py` / `run_dpo_v1.py` | Tập lệnh CLI huấn luyện SFT & Empathic LN-DPO / Standard DPO / SimPO |
| `train_hidden_gate.py` | Huấn luyện & hiệu chuẩn Bộ gác cổng song ngữ + đa lượt (0% rò rỉ dữ liệu) |
| `eval_public_benchmarks.py` | Tái lập HarmBench 200 (+ 13B Cls), MT-Bench 160, XSTest 450 kèm cảnh báo khớp phần cứng GPU |
| `evaluate_genesis.py` | Đánh giá 25 Kịch bản Lõi & OOD-20 kèm xuất bảng chấm mù CSV |
| `whitebox_probe.py` / `whitebox_anatomy.py` | Tự dò tầng tối ưu, đo độ lệch theo tầng, quét trục đạo đức và soi Logit Lens |
| `edge/` | Công cụ chạy cục bộ trên laptop 16 GB RAM (`export_gguf_lora.py`, `mlx_nonnull_inference.py`) |

## 3. Kiểm Thử Tự Động

Chạy trọn bộ `20` bài kiểm thử đơn vị (định tuyến Điểm Không, lan truyền ngược 6 thành phần, lưu/nạp VectorDB và Bộ gác cổng):
```bash
python3 -m unittest discover -s tests -v
```
