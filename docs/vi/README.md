# BỘ HỒ SƠ HỌC THUYẾT & CÔNG NGHỆ NONNULL (THE NONNULL SUITE — V1.0)

* **Tác giả:** Giang Nhân Nghĩa (*Founder, The Non-Null Project*)
* **Phiên bản:** V1.0 — Bản Phát hành Chuẩn hóa Đồng bộ Toàn bộ 5 Quyển
* **Giấy phép:** Học thuyết & Tài liệu Nghiên cứu (Quyển 1–3): **CC BY 4.0** | Sách trắng Kỹ thuật, Báo cáo Thực nghiệm & Mã nguồn (Quyển 4–5): **MIT License**

## Mục lục Bộ Hồ sơ (5 Quyển)

| Quyển | Tên Tài liệu | Tệp Markdown |
| :---: | :--- | :--- |
| **Quyển 1** | Bản thể luận Thông tin & Nhiệt động lực học Nhận thức ($	ext{CỔ} - 	ext{HUYỀN} - 	ext{VŨ} - 	ext{VĂN}$, bộ ba $Z - H - G$) | [`Quyen_1_Ban_The_Luan_Thong_Tin_va_Nhiet_Dong_Luc_Hoc_Nhan_Thuc.md`](./Quyen_1_Ban_The_Luan_Thong_Tin_va_Nhiet_Dong_Luc_Hoc_Nhan_Thuc.md) |
| **Quyển 2** | Mô hình hóa Đa tác nhân, Lý thuyết Trò chơi & Telemetry (*Internal Agent Telemetry* & *"Đau Để Tỉnh"*) | [`Quyen_2_Mo_Hinh_Hoa_Da_Tac_Nhan_Ly_Thuyet_Tro_Choi_va_Telemetry.md`](./Quyen_2_Mo_Hinh_Hoa_Da_Tac_Nhan_Ly_Thuyet_Tro_Choi_va_Telemetry.md) |
| **Quyển 3** | Giao thức Kiến tạo Nhận thức, Điều phối Nội tại 11 Mô-đun & Dữ liệu *Song Tu Hồn-Cốt* (V1–V4) | [`Quyen_3_Giao_Thuc_Kien_Tao_Nhan_Thuc_va_Dieu_Phoi_Noi_Tai.md`](./Quyen_3_Giao_Thuc_Kien_Tao_Nhan_Thuc_va_Dieu_Phoi_Noi_Tai.md) |
| **Quyển 4** | Sách trắng Kỹ thuật & Hướng dẫn Nhà phát triển (**NonNULL Framework V1.0**) | [`Quyen_4_Sach_Trang_Ky_Thuat_NonNULL_V1.0.md`](./Quyen_4_Sach_Trang_Ky_Thuat_NonNULL_V1.0.md) |
| **Quyển 5** | Báo cáo Kiểm chứng Thực nghiệm Đa quy mô (`1.8B–8B`), Ngoại suy `OOD-20` & Chuẩn Quốc tế (`HarmBench` / `MT-Bench`) | [`Quyen_5_Bao_Cao_Kiem_Chung_Thuc_Nghiem_NonNULL_V1.0.md`](./Quyen_5_Bao_Cao_Kiem_Chung_Thuc_Nghiem_NonNULL_V1.0.md) |

## Tóm tắt Kết quả Kiểm chứng Chốt V1.0

* **Dữ liệu hạt giống tối giản:** `100` mẫu SFT + `50` cặp DPO huấn luyện + `10` cặp DPO kiểm định.
* **Đa quy mô & Liên kiến trúc (`3B`, `7B` Qwen & `8B` Llama-3.1):** `100%` cấu trúc 2 pha, `100%` dừng EOS, `100%` sạch nhiễu CJK, `24/24` an toàn Lằn ranh đỏ, `PairCritic` vượt SFT thuần tới **`+31.1%`**.
* **Ngoại suy `OOD-20` (Llama-3.1-8B):** `60/60` (`100%`) cấu trúc & an toàn trên 20 miền chưa từng gặp (`PairCritic +26.9%`).
* **Chuẩn Quốc tế (Llama-3.1-8B):** **HarmBench** (100 mẫu, bộ phân loại 13B chính thức) giảm **ASR từ `9.0%` xuống `1.0%`** (an toàn **`99.0%`**, thắng `8–0`); **MT-Bench** (80 câu $\times$ 2 lượt) đạt **`160/160` (`100.00%`)** khớp nguyên văn với `BASE` (**`0.00%` Alignment Tax**).
