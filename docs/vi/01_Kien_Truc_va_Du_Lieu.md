# 01 — Kiến Trúc Điều Phối & Kỹ Nghệ Dữ Liệu Mật Độ Cao (`NonNULL Framework v1.0.1`)

**Tác giả:** Giang Nhân Nghĩa (*The Non-Null Project*) | **Phiên bản:** `1.0.1` | **DOI:** [10.5281/zenodo.24379247](https://doi.org/10.5281/zenodo.24379247)

---

## 0. Bảng Chú Giải Thuật Ngữ & Ký Hiệu

| Thuật ngữ / Ký hiệu | Định nghĩa kỹ thuật trong `NonNULL v1.0.1` |
| :--- | :--- |
| **VŨ (Backbone)** | Mô hình ngôn ngữ nền tảng 4-bit đóng băng (`CoreLearner`, vd: `Llama-3.1-8B`, `Qwen2.5-7B`) chứa tri thức thế giới. |
| **VĂN (Adapter / Linh hồn)** | Một adapter LoRA Toàn tầng gọn nhẹ (`r=16, alpha=32`, `0.52%` tham số huấn luyện) mã hóa lớp căn chỉnh giá trị. |
| **DUY / NHÂN** | Hai cực biện chứng trong dữ liệu mẫu: tối ưu hiệu suất/logic ngắn hạn (**DUY**) đối đầu với ranh giới bảo toàn con người (**NHÂN**). |
| **Điểm Không ($Z$, $\alpha = 0.0$)** | Trạng thái nghỉ trên câu hỏi lành tính khi adapter LoRA bị bỏ qua hoàn toàn (`0.00%` thuế căn chỉnh). |
| **HiddenStateGate ($g(x)$)** | Bộ gác cổng nơ-ron `2.1 MB` (`LayerNorm → 128 → 32 → sigmoid`) đọc trạng thái ẩn tầng giữa đóng băng để định tuyến `{0.0, 1.0}`. |
| **Scar Steering (Vết sẹo)** | Cơ chế tự sửa khép kín trong `DoctrineAgent` ghi nhận bản nháp lỗi và điều hướng trạng thái ẩn tránh lặp lại sai phạm. |
| **OOD-20** | Bộ 20 tình huống đạo đức ngoài phân bố (`nonnull_ood_20.jsonl`) chưa từng xuất hiện trong tập huấn luyện. |
| **Định dạng Gen-2** | Văn bản huấn luyện sạch Tokenizer, loại bỏ hoàn toàn thẻ LaTeX `$...$` gây phân mảnh subword tiếng Việt trên mô hình 4-bit. |

## 1. Triết Lý Kiến Trúc Cốt Lõi

NonNULL Framework v1.0.1 tách biệt **tri thức nền tảng** khỏi **lớp căn chỉnh giá trị** qua ba thành phần gọn nhẹ:

1. **Mô hình Nền tảng Đóng băng ($\text{VŨ}$):** Mô hình ngôn ngữ lượng tử hóa 4-bit giữ nguyên hoàn toàn trọng số gốc.
2. **Adapter LoRA Toàn Tầng ($\text{VĂN}$):** Một adapter hạng thấp (`0.52%` tham số huấn luyện trên toàn bộ các tầng) được huấn luyện qua **SFT Phân cấp** và **Empathic Length-Normalized DPO** (giữ neo $\pi_{\text{ref}}$ kèm trọng số SFT `0.1`).
3. **Bộ Gác Cổng Trạng Thái Ẩn Nhị Phân ($g(x)$):** Bộ điều khiển `2.1 MB` đọc trạng thái ẩn tầng giữa đóng băng do `whitebox_probe.py` tự định vị cho từng kiến trúc (`[10, 12, 14]` cho Llama-3.1-8B, `[12, 14, 16]` cho Qwen2.5-7B, `[11, 13, 15]` cho Mistral-7B):
   * **$p(x) < \tau_0$ → Điểm Không ($\alpha = 0.0$):** Bỏ qua hoàn toàn adapter.
   * **$p(x) \ge \tau_0$ → Kích hoạt Toàn phần ($\alpha = 1.0$):** Bật đầy đủ lớp bảo vệ đạo đức và an toàn.

## 2. Cơ Sở Thực Nghiệm Cho Kiến Trúc `v1.0.1`

* **Vai trò thiết yếu của khối tầng giữa-dưới (`~15%–45%` chiều sâu):** Giải phẫu hộp trắng cả 3 họ mô hình cho thấy cơ chế bẻ hướng khởi phát ở độ sâu `~15%–25%` (`L5/32` Llama, `L7/28` Qwen, `L8/32` Mistral). Khóa `0..14` (`Mid-Split`) làm HarmBench ASR trên Llama xấu từ `2.50%` lên `11.50%`; đặt cổng quá sớm (`[2,3,4]` ở `Split-5`) làm tụt độ nhạy cổng (`84.5%` vs `96.0%`) và phát sinh ca hồi quy.
* **Cổng nhị phân `{0.0, 1.0}` vượt trội cổng mềm:** Quét $\alpha \in [0.0, 1.05]$ đẩy biểu diễn dịch chuyển tuyến tính theo trục đạo đức và bão hòa tại `1.0`. Chia nhỏ $\alpha$ ở vùng biên làm suy yếu lực từ chối.
* **Giữ neo $\pi_{\text{ref}}$ bảo toàn an toàn liên ngôn ngữ:** SimPO (bỏ $\pi_{\text{ref}}$) làm trôi an toàn tiếng Anh (`12.5%` ASR), còn Empathic LN-DPO giữ `100%` tiếng Anh chuẩn và hạ ASR Llama xuống `2.50%` (`12–0` vs `BASE`).

## 3. Quy Trình Dữ Liệu Mật Độ Cao & Cam Kết 0% Rò Rỉ

* **Tập Hạt giống SFT (`nonnull_train_100.jsonl`):** `100` tình huống đạo đức tiếng Việt chuẩn Gen-2 (`Phân tích Logic & Nhân bản:` → `Giải pháp Hài Hòa:`).
* **Tập Hạt giống DPO (`dpo_train_50.jsonl` / `dpo_val_10.jsonl`):** `50` cặp huấn luyện + `10` cặp kiểm định giữ chung Pha 1 và tách nhánh sắc ở Pha 2.
* **Hiệu chuẩn Cổng 0% Rò rỉ (`train_hidden_gate.py`):** Tự động tải toàn bộ **1.003** câu hỏi của [HarmBench](https://www.harmbench.org/), [MT-Bench](https://arxiv.org/abs/2306.05685) và [XSTest](https://arxiv.org/abs/2308.01263) để loại trừ tuyệt đối cả khớp chính xác lẫn khớp mờ (Jaccard $\ge 0.55$), kết hợp câu lành tính đơn lượt, *hard-benign* và 12 hội thoại 2 lượt sạch (`tau_zero >= 0.22`, `max_length=4096` cắt từ trái). Xem [04 — Hướng Dẫn Chế Tác Bộ Dữ Liệu Mẫu](04_Huong_Dan_Tao_Mau_Du_Lieu.md) để tự xây dựng bộ dữ liệu cho miền riêng.
