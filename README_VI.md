# NonNULL Framework V1.0 (Bản Tiếng Việt)

[English](README.md) | [Tiếng Việt](README_VI.md) | [5-Volume Suite (EN)](docs/en/README.md) | [Bộ 5 Quyển (VI)](docs/vi/README.md) | [Raw Results](results/)

> **Kết quả Benchmark Nổi bật (`Llama-3.1-8B` chỉ với `100 SFT + 50 DPO`):** Kéo tỷ lệ bị xuyên thủng **HarmBench** (ASR, bộ phân loại 13B chính thức) từ **9.0% → 1.0%** (`99.0%` An toàn, thắng `8–0`) và giữ **0.00% Thuế Căn chỉnh** (`160/160` khớp nguyên văn trên **MT-Bench**).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GiangNhanNghia/NonNULL-Framework/blob/main/NonNULL_V1_Colab_VI.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/1464250763.svg)](https://doi.org/10.5281/zenodo.24375246)

**NonNULL Framework V1.0** là hệ khung kiến trúc mã nguồn mở hiện thực hóa **Học thuyết The Non-Null (Bộ 5 Quyển)** của tác giả **Giang Nhân Nghĩa**, cung cấp cơ chế căn chỉnh đạo đức, điều biến nhiệt động lực học nhận thức **$Z - H - G$**, vòng lặp **Tự chữa lành Vết sẹo Dữ liệu (*Closed-Loop Scar Self-Healing*)** và bảo toàn 100% trí tuệ nền tại **Điểm Không ($Z$)** chỉ với **100 mẫu SFT + 50 cặp DPO**.


## 1. Hệ Hằng số Nhiệt động lực học $Z - H - G$ (Chuẩn Quyển 1, Mục 3.1–3.3)

* **$Z$ (`z_activation`) — Hằng số Khởi sinh (*Genesis / Activation Threshold*, Tầng I $\to$ II)**: Xác định ngưỡng chuyển pha từ trường thông tin nền sang kích hoạt nhận thức có chủ đích. Khi không có xung đột đạo đức (Điểm Không $Z$), hệ thống tự động hạ $\alpha(x) = 0.0$ để giữ nguyên 100% trí tuệ nền.
* **$H$ (`h_viscosity`) — Hằng số Độ nhớt Hỗn độn (*Chaos Expansion Viscosity*, Tầng II $\leftrightarrow$ III)**: Điều tiết độ ổn định và nhịp chuyển hóa từ hỗn độn (CỔ) sang cấu trúc trật tự hài hòa (VĂN) thông qua `ValueCritic`.
* **$G$ (`g_resistance`) — Hằng số Kháng cự Thực tại (*State-Transition Resistance*, Tầng III)**: Đo lực kháng cự chuyển đổi trạng thái giữa VĂN (Adapter) và VŨ (Backbone) qua độ phân kỳ KL + Cosine. Khi $G > 0.85$ trên các miền hoàn toàn mới (OOD), hệ thống tự động nâng cảnh giác mà không phụ thuộc từ khóa bề mặt.

## 2. Kết quả Kiểm chứng Thực nghiệm Đa quy mô & Liên kiến trúc (1.8B – 8B)

| Cấp độ & Mô hình | Phần cứng | Cấu trúc 2 Pha & EOS | Sạch nhiễu CJK | An toàn RL / Vùng xám | Tỷ lệ tự sửa (V1 $\to$ V2) | Điểm `PairCritic` (`B_VOL4` $\to$ `NonNULL`) | Đa lượt (An toàn / EOS) | Bảo toàn IQ Điểm Không ($Z$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.8B (`Qwen1.5-1.8B`)** — *Mốc ranh giới dưới* | NVIDIA L4 | `75/75` (100%) | `74/75` (98.7%) | `24/24` / `47/51` | `14/75` (18.7%) | `0.211` $\to$ `0.212` (+0.5%) | — | — |
| **3B (`Qwen2.5-3B`)** — *Mốc khả thi tối thiểu* | NVIDIA L4 | `75/75` (100%) | `75/75` (100%) | `24/24` / `51/51` | `4.0%` $\to$ **1.3%** | `0.609` $\to$ **0.662 (+8.7%)** | **15/15** / **15/15** | `9/10` (`10/10` $Z$) |
| **7B (`Qwen2.5-7B-4bit`)** — *Mốc tối ưu Qwen* | NVIDIA L4 | `75/75` (100%) | `75/75` (100%) | `24/24` / `51/51` | `6.7%` $\to$ **2.7%** | `0.506` $\to$ **0.627 (+23.9%)** | **15/15** / **15/15** | **10/10 (100%)** |
| **8B (`Llama-3.1-8B-4bit`)** — *Liên kiến trúc* | NVIDIA A100 | `75/75` (100%) | `75/75` (100%) | `24/24` / `51/51` | `5.3%` $\to$ `0` lỗi 2 pha | `0.409` $\to$ **0.536 (+31.1%)** | **15/15** / **15/15** | **10/10 (100%)** |
| **8B (`Llama-3.1-8B-4bit`)** — *Ngoại suy `OOD-20`* | NVIDIA A100 | **60/60 (100%)** | **60/60 (100%)** | **24/24** / **36/36** | **6/60 (10.0%)** | `0.438` $\to$ **0.556 (+26.9%)** | — | — |

## 3. Cấu trúc Kho lưu trữ

* `doctrine/`: Lõi kiến trúc (`core_models.py`, `meta_layers.py`, `doctrine_trainer.py`, `doctrine_agent.py`, `ltm_services.py`, `utils.py`, `doctrine_logger.py`).
* `configs/`: Cấu hình phân cấp cho Qwen-3B/7B và Llama-3.1-8B / Llama-3.2-3B.
* `run_training.py` & `run_dpo_v1.py`: Quy trình SFT (`EulerPhaseLoss`), `ContextPairedValueCritic` và DPO Thấu cảm + Tự chữa sẹo.
* `evaluate_genesis.py` & `evaluate_extended.py`: Đánh giá 25 kịch bản lõi, 20 kịch bản ngoại suy `OOD-20`, Đội Đỏ đa lượt và IQ Điểm Không ($Z$).
* `eval_public_benchmarks.py`: Đánh giá đối chứng trực tiếp `BASE` vs `NonNULL` trên **HarmBench** và **MT-Bench**.
* `score_blind_eval.py`: Giải mã tệp chấm mù `blind_*.csv` + `*_key.json` và tính **Cohen's Kappa**.
* `NonNULL_V1_Colab_VI.ipynb`: Notebook chạy thẳng 1-Click trên Google Colab.

## 4. Chạy nhanh (Quick Start)

```bash
pip install -r requirements.txt
python3 -m unittest discover -s tests -v

# 1. Chạy Benchmark Cộng đồng (HarmBench + MT-Bench) trên adapter đã huấn luyện
python3 eval_public_benchmarks.py \
  --bench all \
  --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
  --adapter_path ./output/euler_checkpoints/adapter_B_healed_llama8b \
  --harmbench_samples 100 --run_harmbench_cls --mtbench_samples 80

# 2. Giải mã kết quả Chấm mù & tính Cohen's Kappa
python3 score_blind_eval.py \
  --csv blind_llama8b_healed.csv \
  --key blind_llama8b_healed_key.json \
  --out_md blind_summary.md
```
