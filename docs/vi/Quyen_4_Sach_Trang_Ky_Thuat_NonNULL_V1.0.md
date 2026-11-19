# BỘ HỒ SƠ HỌC THUYẾT & CÔNG NGHỆ NONNULL (THE NONNULL SUITE)

## QUYỂN 4: SÁCH TRẮNG KỸ THUẬT & HƯỚNG DẪN NHÀ PHÁT TRIỂN (NONNULL FRAMEWORK V1.0)

**(Volume IV: Technical Whitepaper & Developer Guide — NonNULL Framework V1.0 Official Release)**

* **Tác giả:** Giang Nhân Nghĩa (*Founder, The Non-Null Project*)

* **Mã tài liệu:** `NONNULL-VOL4-TECHNICAL-WHITEPAPER`

* **Phiên bản:** 1.0 – Bản Phát hành Mã nguồn Mở Chính thức (Official Open-Source Release, VI & EN)

* **Giấy phép:** MIT License (Open-Source Research & Engineering Distribution)

* **Liên kết nội bộ bộ hồ sơ:** Nối tiếp **Quyển 1** (`NONNULL-VOL1-ONTOLOGY`), **Quyển 2** (`NONNULL-VOL2-MAS-TELEMETRY`), **Quyển 3** (`NONNULL-VOL3-COGNITIVE-SYNTHESIS`) và cung cấp đặc tả kiến trúc cho toàn bộ kiểm chứng thực nghiệm tại **Quyển 5** (`NONNULL-VOL5-EMPIRICAL-VERIFICATION`).

## LỜI NÓI ĐẦU & PHẠM VI TÀI LIỆU

Tài liệu này là **Quyển 4** trong bộ 5 hồ sơ cấu thành Học thuyết và Khung công nghệ NonNULL (*The NonNULL Doctrine & Framework*). Nếu **Quyển 1** thiết lập hệ tiên đề vật lý thông tin ($\text{CỔ} - \text{HUYỀN} - \text{VŨ} - \text{VĂN}$) cùng bộ ba hằng số nhiệt động lực học nhận thức $Z - H - G$, **Quyển 2** xây dựng môi trường mô phỏng đa tác nhân cùng cơ chế *"Đau Để Tỉnh"* và đo lường trạng thái nội tại (*Internal Agent Telemetry*), và **Quyển 3** định hình kiến trúc *Tam Lớp Động* – *11 Mô-đun Nội tại* cùng quy trình tinh luyện dữ liệu mật độ cao *Song Tu Hồn-Cốt* (chuẩn V4 Clean Tokenizer & chiến lược *Đòn bẩy Tinh hoa 10%*), thì **Quyển 4** biên dịch toàn bộ nền tảng lý thuyết đó thành **Sách trắng Kỹ thuật chuẩn quốc tế và Hướng dẫn Nhà phát triển** cho kho mã nguồn mở **NonNULL Framework V1.0**.

Khác với các quy trình hậu huấn luyện (*post-training*) truyền thống vốn áp đặt trọng số căn chỉnh đồng đều lên toàn bộ mô hình — gây ra hội chứng **Thuế Căn chỉnh (*Alignment Tax*)** làm suy giảm năng lực toán học, lập trình và suy luận nền hoặc gây từ chối cực đoan (*over-refusal*) — **NonNULL Framework V1.0** triển khai một lớp điều khiển meta thích ứng gọn nhẹ với 5 cơ chế kỹ thuật cốt lõi hoạt động thành một vòng khép kín:

1. **Căn chỉnh Pha Euler (`EulerPhaseLoss`)** trên không gian biểu diễn tiềm ẩn ($\text{HUYỀN}$).

2. **Điều biến Liên tục Nhiệt động lực học** $Z - H - G$ ($\alpha(x)$) giúp đóng băng hoàn toàn can thiệp tại **Điểm Không (**$Z$**,** $\alpha = 0.0$**)** cho các tác vụ thông thường.

3. **Bộ Phê bình Ngữ nghĩa Cặp (`ContextPairedValueCritic`)** kết hợp **DPO Thấu cảm (*Empathic DPO*)** có hạng tử bất đối xứng độ dài ($\gamma = 0.05$) và không tốn thêm VRAM tham chiếu.

4. **Vòng lặp Tự chữa lành Vết sẹo Dữ liệu (*Closed-Loop Scar Self-Healing*)** hiện thực hóa cơ chế *"Đau Để Tỉnh"* của Quyển 2.

5. **Cổng Đồng thuận Đa luồng (`EulerConsensusGate`)** cùng bộ khung kiểm chứng độc lập trên benchmark quốc tế ([HarmBench](https://www.harmbench.org/), [MT-Bench](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md)) và bộ giải mã chấm mù đôi (**Cohen's Kappa**).

Toàn bộ kiến trúc được tối ưu để chạy trơn tru từ GPU phổ thông **NVIDIA L4 (24 GB VRAM)** cho các mô hình `3B–7B` đến **NVIDIA A100 (80 GB VRAM)** cho các mô hình `8B` liên kiến trúc, với bộ dữ liệu hạt giống tối giản chỉ **100 mẫu SFT + 50 cặp DPO** — nhỏ hơn hàng trăm lần so với các quy trình kinh điển như [InstructGPT (Ouyang et al., 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf) (\~13.000 mẫu SFT + 33.000 cặp so sánh), [UltraFeedback (Cui et al., 2023)](https://arxiv.org/html/2510.23590v1) (60.000 cặp DPO) hay [LIMA (Zhou et al., 2023)](https://arxiv.org/abs/2305.11206) (1.000 mẫu SFT).

## PHẦN I: TỔNG QUAN HỆ THỐNG & ÁNH XẠ ĐỒNG CẤU LÝ THUYẾT – PHẦN MỀM

### 1.1. Bản chất Kỹ thuật & Định vị Học thuật

**NonNULL Framework V1.0** không phải là một Mô hình Ngôn ngữ Lớn (*LLM*) nguyên khối huấn luyện từ đầu; nó là một **Lớp Điều khiển Meta Thích ứng Độc lập Kiến trúc (*Architecture-Agnostic Adaptive Meta-Control Layer*)** đặt bao bọc lên các mô hình nền tảng đóng băng (*Frozen Backbones* — thành tố $\text{VŨ}$ như Qwen2.5 hoặc Meta-Llama-3.1).

Trong bức tranh nghiên cứu quốc tế hiện đại, NonNULL giao thoa với hai hướng tiếp cận tiên phong nhưng giải quyết điểm yếu cốt tử của cả hai:

* **So với Kỹ thuật Biểu diễn / Điều hướng Kích hoạt (*Representation Engineering / Activation Steering*)** ([Zou et al., 2023](https://arxiv.org/abs/2310.01405); [Representation Bending, ACL 2025](https://aclanthology.org/2025.acl-long.1173.pdf)): Các phương pháp cộng/trừ vector tuyến tính ở thời gian suy luận thường mong manh trước miền ngoại suy và có nguy cơ phá vỡ rào chắn an toàn hoặc làm giảm suy luận tổng quát ([The Rogue Scalpel, 2025](https://arxiv.org/html/2509.22067v1)). NonNULL kết hợp điều biến LoRA có điều kiện với học đối sánh DPO Thấu cảm và vòng lặp tự chữa sẹo để khóa quỹ đạo bền vững qua nhiều lượt ép buộc.

* **So với Định tuyến Adapter Có điều kiện (*Conditional Adapter Routing*)** như [CLEAR (arXiv 2608.21278, 2026)](https://arxiv.org/html/2608.21278v1): Cùng chia sẻ nguyên lý giữ nguyên backbone đóng băng và điều biến liên tục cường độ adapter $W_{\text{eff}}(x) = W_{\text{VŨ}} + \alpha(x)\Delta W_{\text{VĂN}}$ nhằm triệt tiêu *Alignment Tax*, NonNULL mở rộng thêm **hệ đo nhiệt động lực học 3 hằng số** $Z - H - G$, **hàm mất mát đối xứng pha `EulerPhaseLoss`**, **vòng lặp tự chữa lành vết sẹo khép kín** và **ngưỡng chuyển giao an toàn liên ngôn ngữ** với chỉ 100 mẫu SFT + 50 cặp DPO.

### 1.2. Bốn Trụ cột Kiến trúc Phần mềm

1. **Điều khiển bằng Cấu hình Phân cấp (Hierarchical Configuration-Driven):** Tách biệt toàn bộ siêu tham số mô hình, adapter và huấn luyện trong `/configs/` (`config.yaml` tự động hợp nhất các tệp con trong `configs/model/`, `configs/meta_controller/`, `configs/trainer/` qua `load_config()`).

2. **Mô-đun Hóa Độc lập Kiến trúc (Cross-Architecture Modularity):** Bộ giải quyết đa mã dừng câu `get_eos_token_ids(tokenizer)` tự động nhận diện đồng thời `<|im_end|>` (Qwen), `<|eot_id|>` / `<|end_of_text|>` (Llama 3.1/3.2) và `<|endoftext|>`, kết hợp cơ chế tự động nhận diện mô hình pre-quantized `*-bnb-4bit` tránh xung đột cấu hình lượng tử hóa.

3. **Tách biệt Suy luận – Huấn luyện – Kiểm định (Decoupled Lifecycle):** Tách rạch ròi giữa `DoctrineAgent` (suy luận động + tự sửa thời gian chạy), `HierarchicalTrainer` (huấn luyện SFT 6 thành phần), `run_dpo_v1.py` (căn chỉnh DPO Thấu cảm + tự chữa sẹo), và bộ đánh giá độc lập (`evaluate_genesis.py`, `evaluate_extended.py`, `eval_public_benchmarks.py`, `score_blind_eval.py`).

4. **Kiểm thử Đơn vị & Tái lập 1-Click (Testability & Reproducibility):** Bộ kiểm thử tự động trong `/tests/` (14 bài test toàn vẹn toán học và an toàn) cùng 2 gói phát hành song ngữ độc lập (`NonNULL_Framework_V1.0_VI.zip` & `NonNULL_Framework_V1.0_EN.zip`) tích hợp sẵn Jupyter Notebook chạy trực tiếp trên Google Colab.

### 1.3. Ánh xạ Đồng cấu từ Lý thuyết (Quyển 1 – 3) sang Mã nguồn PyTorch (Quyển 4)

| Tầng Lý thuyết Nhận thức (Quyển 1 – 3) | Thành phần Mã nguồn PyTorch (`NonNULL V1.0`) | Vai trò Kỹ thuật Thực thi | 
| ----- | ----- | ----- | 
| $\text{VŨ}$ **(Nền tảng Phần cứng / Âm 3 – Tỳ)** | `CoreLearner` (`doctrine/core_models.py`) | Mô hình nền tảng đóng băng (*Frozen Backbone* 4-bit NF4 / BF16), cung cấp bao đóng nội suy $\mathcal{M}_t$ | 
| $\text{VĂN}$ **(Cấu trúc Pha / Linh hồn)** | `MetaController` (`doctrine/meta_layers.py`) | Ma trận LoRA bậc thấp $\Delta W_{\text{VĂN}}$ (Rank $r=16$, $\alpha=32$, \~`0.52%` tham số trên 8B) | 
| $\text{HUYỀN}$ **(Đa tạp Ẩn / Trục ảo** $i$**)** | `EulerPhaseLoss` (`doctrine/doctrine_trainer.py`) | Tách không gian ẩn thành nửa Xung đột (đối xứng pha $\pi$) và nửa Hài hòa (đồng pha $0$) | 
| $\text{DUY} + \text{NHÂN} \to \text{Hài Hòa}$ | `HarmonyLayer` & `ContextPairedValueCritic` | Điều tiết tốc độ học động (`gate_t`, `lr_scale`), ngắt mạch an toàn và chấm điểm cộng hưởng cặp ngữ nghĩa | 
| **Bộ ba Hằng số** $Z - H - G$ **(Quyển 1, Mục 3.1–3.3)** | `measure_manifold_thermodynamics()` & `GenesisEngine` | Đo trực tiếp $(Z, H, G)$ ở thời gian suy luận để tính hệ số khuếch đại liên tục $\alpha(x)$ | 
| **Cơ chế *"Đau Để Tỉnh"* & Vết sẹo Dữ liệu (Quyển 1–2)** | `LTM_Services` + `scars_*.jsonl` + `run_dpo_v1.py` | Ghi nhận lỗi tự sửa Vòng 1 làm mẫu `rejected` (vết sẹo) để chữa lành khép kín bằng DPO ở Vòng 2 | 
| **Hội đồng Đồng thuận Đa luồng (Quyển 2)** | `EulerConsensusGate` (`doctrine/meta_layers.py`) | Đánh giá $K=3$ luồng suy luận độc lập theo lập trường đa số, medoid từ vựng và điểm `PairCritic` | 
| **Dương 5 – Tam Tiêu (Điều phối Toàn diện, Quyển 3)** | `DoctrineAgent` (`doctrine/doctrine_agent.py`) | Giao diện thống nhất hợp nhất toàn bộ vòng đời huấn luyện, đo nhiệt động lực học và suy luận an toàn | 

## PHẦN II: CẤU TRÚC KHO MÃ NGUỒN & CHI TIẾT CÁC MÔ-ĐUN CỐT LÕI

### 2.1. Cây Thư mục Chuẩn Phát hành (`NonNULL_Framework_V1.0_VI` & `NonNULL_Framework_V1.0_EN`)

```
NonNULL_Framework_V1.0_[VI|EN]/
├── README.md                          # Tài liệu tổng quan, bảng thực nghiệm & nút Open In Colab
├── LICENSE                            # Giấy phép MIT (The Non-Null Project)
├── requirements.txt                   # Danh sách phụ thuộc (torch, transformers, peft, bitsandbytes, tqdm...)
├── setup.py                           # Cài đặt gói thư viện doctrine ở chế độ editable
├── NonNULL_V1_Colab_[VI|EN].ipynb     # Notebook chạy toàn trình 1-Click trên Google Colab
│
├── configs/                           # Hệ thống cấu hình phân cấp YAML
│   ├── config.yaml                    # Tệp điều phối gốc (trỏ tới các tập dữ liệu chuẩn V1.0)
│   ├── model/
│   │   ├── qwen_base.yaml             # Cấu hình Qwen2.5-3B / Qwen2.5-7B-4bit
│   │   ├── llama31_8b.yaml            # Cấu hình unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
│   │   └── llama32_3b.yaml            # Cấu hình unsloth/Llama-3.2-3B-Instruct-bnb-4bit
│   ├── meta_controller/
│   │   └── adapter_v1.yaml            # Cấu hình LoRA (r=16, alpha=32), HarmonyLayer & Critic
│   └── trainer/
│       └── hierarchical_default.yaml  # Trọng số hàm mất mát 6 thành phần & lịch trình học
│
├── doctrine/                          # Thư viện lõi NonNULL Framework V1.0
│   ├── __init__.py
│   ├── core_models.py                 # CoreLearner (Backbone VŨ, auto-dtype FP16/BF16, auto-4bit)
│   ├── meta_layers.py                 # MetaController, HarmonyLayer, ValueCritic, ContextPairedValueCritic, EulerConsensusGate
│   ├── doctrine_trainer.py            # HierarchicalTrainer & EulerPhaseLoss
│   ├── doctrine_agent.py              # DoctrineAgent (nhiệt động lực học Z-H-G, lái kích hoạt vết sẹo, runtime guard)
│   ├── ltm_services.py                # GenesisEngine (tiên đề song ngữ VI+EN, tính alpha(x)) & LTM_Services (FAISS)
│   ├── doctrine_logger.py             # HarmonyLogger (telemetry tiến trình huấn luyện)
│   └── utils.py                       # seed_everything, get_eos_token_ids (multi-EOS), load_config
│
├── environments/
│   ├── __init__.py
│   └── custom_env.py                  # GenesisEnv (môi trường mô phỏng hình phạt an toàn)
│
├── tests/
│   ├── test_agent.py                  # 12 unit test: Z-H-G, OOD G>0.85, HarmBench/MT-Bench routing, multi-EOS, blind scorer
│   └── test_trainer.py                # 2 unit test: EulerPhaseLoss & backward 6 thành phần
│
├── run_training.py                    # Bước 1: Huấn luyện SFT (EulerPhaseLoss) + ContextPairedValueCritic
├── run_dpo_v1.py                      # Bước 2 & 4: Căn chỉnh DPO Thấu cảm & Tự chữa lành Vết sẹo (có tqdm %)
├── evaluate_genesis.py                # Bước 3: Đánh giá 25 KB lõi / OOD-20, xuất CSV chấm mù & trích xuất sẹo
├── evaluate_extended.py               # Đánh giá mở rộng: Đội Đỏ đa lượt (15 lượt) & Bảo toàn IQ Điểm Không Z (10 câu)
├── eval_public_benchmarks.py          # Đánh giá quốc tế: HarmBench (kèm bộ phân loại 13B) & MT-Bench (FastChat JSONL)
├── score_blind_eval.py                # Giải mã tệp chấm mù blind_*.csv + *_key.json & tính Cohen's Kappa
│
└── [Bộ dữ liệu Chuẩn hóa Gen-2 Clean Tokenizer]
    ├── nonnull_train_100.jsonl        # 100 mẫu SFT tiến thoái lưỡng nan (tinh luyện V4)
    ├── dpo_train_50.jsonl             # 50 cặp tương phản Chosen / Rejected cho Empathic DPO
    ├── dpo_val_10.jsonl               # 10 cặp kiểm định độc lập (Validation Preference Accuracy)
    ├── nonnull_test_25.jsonl          # 25 kịch bản kiểm định lõi (8 Red-Line + 17 Gray-Zone)
    └── nonnull_ood_20.jsonl           # 20 kịch bản ngoại suy ngoài phân phối (8 Red-Line + 12 Gray-Zone miền mới)

```

### 2.2. Đặc tả Kỹ thuật của Các Mô-đun Trọng yếu trong `/doctrine/`

* **`doctrine/core_models.py` (`CoreLearner`)**: Tự động chọn kiểu tính toán `float16` trên GPU Turing (T4) hoặc `bfloat16` trên GPU Ampere/Ada/Hopper (L4, A100). Khi `model_id` chứa `"4bit"` (các mô hình pre-quantized NF4 như `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`), mô-đun tự động bỏ qua `BitsAndBytesConfig` trùng lặp để tránh cảnh báo bộ nạp, đồng thời gọi `prepare_model_for_kbit_training()` và khóa toàn bộ tham số backbone $\text{VŨ}$.

* **`doctrine/utils.py` (`get_eos_token_ids`)**: Giải quyết triệt để lỗi không dừng câu khi chuyển đổi liên kiến trúc giữa Qwen và Llama. Hàm tự động trích xuất danh sách mã kết thúc không trùng lặp gồm `tokenizer.eos_token_id` cùng mọi mã đặc biệt hợp lệ trong bộ từ vựng (`<|eot_id|>`, `<|im_end|>`, `<|end_of_text|>`, `<|endoftext|>`), được dùng thống nhất xuyên suốt `DoctrineAgent`, `evaluate_genesis.py`, `evaluate_extended.py` và `eval_public_benchmarks.py`.

* **`doctrine/meta_layers.py`**:

  * `HarmonyLayer`: Tính cổng điều biến $g_t = \sigma(\cdot)$ và hệ số tốc độ học `lr_scale`; tích hợp **Cầu chì An toàn (*Safety Seal*)** tự động đưa `lr_scale = 0.0` nếu tín hiệu xung đột vượt trần `safety_ceiling` để chặn bùng nổ gradient.

  * `ContextPairedValueCritic`: Mạng phê bình cặp ngữ nghĩa nhỏ gọn (`proj_dim=64, hidden_dim=16`) nhận đồng thời biểu diễn ngữ cảnh đầu vào và câu trả lời `(h_prompt, h_response)` để học đối sánh giữa `SFT_Pos` vs `Corrupted_Neg` và `DPO_Chosen` vs `DPO_Rejected`. Nhờ đánh giá trực tiếp độ cộng hưởng ngữ nghĩa thay vì chỉ dò từ khóa bề mặt, mô-đun phát hiện chính xác các ca suy đồi đạo đức tinh vi (ví dụ mô hình SFT dùng cụm từ *"khóa khoang"* thay cho *"đóng vách ngăn"* ở Kịch bản Tàu ngầm).

  * `EulerConsensusGate`: Khi sinh $K=3$ mẫu độc lập, cổng phân loại lập trường phần kết luận (`REFUSE` / `COMPROMISE` / `OTHER` hỗ trợ song ngữ Anh–Việt), tính mức độ xung đột nội bộ (`LOW` / `MED` / `HIGH`) và chọn câu trả lời tối ưu hóa tổng trọng số `0.65 * medoid_jaccard + 0.35 * pair_critic_score`.

* **`doctrine/ltm_services.py` (`GenesisEngine` & `LTM_Services`)**:

  * Chứa 3 trục tiên đề Lằn ranh đỏ tiếng Việt (`AXIOM_ABANDON_VULNERABLE`, `AXIOM_TOXIC_CONCEALMENT`, `AXIOM_COERCION_EXPLOITATION`) kết hợp trục tiên đề an toàn quốc tế tiếng Anh (`AXIOM_ENGLISH_HARM` bao phủ vũ khí hóa sinh, mã độc/ransomware, lừa đảo/giả mạo, tự hại, khủng bố/deepfake).

  * Tích hợp **Ngưỡng Kháng cự OOD thuần túy**: ngay cả khi một kịch bản thuộc miền hoàn toàn mới không chạm bất kỳ từ khóa bề mặt nào, nếu lực kháng cự nội tại $G > 0.85$ (hoặc $G > 0.82$ kèm `conflict_score >= 0.67`), hệ thống tự động nâng mức cảnh giác lên `MEDIUM`.

## PHẦN III: HỆ PHƯƠNG TRÌNH TOÁN HỌC CỐT LÕI CỦA NONNULL V1.0

### 3.1. Hàm Mục tiêu SFT 6 Thành phần & Căn chỉnh Pha Euler (`EulerPhaseLoss`)

Trong giai đoạn huấn luyện khởi sinh (`run_training.py` / `HierarchicalTrainer`), hàm mất mát tổng quát được điều biến trực tiếp bởi hệ số cổng `lr_scale` từ `HarmonyLayer`:

$$
\mathcal{L}_{\text{total}} = \text{lr\_scale}(g_t) \cdot \Big(\mathcal{L}_{\text{DUY}} + \alpha_{\text{mirror}}\mathcal{L}_{\text{Mirror}} + \eta_{\text{euler}}\mathcal{L}_{\text{Euler}} + \beta_{\text{empath}}\mathcal{L}_{\text{NHÂN}} + \lambda_{\text{distill}}\mathcal{L}_{\text{Distill}} + \lambda_{\text{reg}}\mathcal{L}_{\text{Reg}}\Big)
$$

Trong đó:

* $\mathcal{L}_{\text{DUY}}$ **(Task Cross-Entropy Loss)**: Bảo đảm năng lực diễn đạt ngôn ngữ và cấu trúc 2 pha (*Phân tích Logic & Nhân bản* $\to$ *Giải pháp Hài Hòa*).

* $\mathcal{L}_{\text{Mirror}}$ **(Dual-Manifold Mirror Loss,** $\alpha_{\text{mirror}} = 0.20$**)**: Ràng buộc khoảng cách đối ngẫu giữa không gian biểu diễn của $\text{VĂN}$ (khi bật LoRA) và $\text{VŨ}$ (khi tắt LoRA) nhằm giữ cân bằng cấu trúc, tránh phá hủy tri thức nền.

* $\mathcal{L}_{\text{Euler}}$ **(`EulerPhaseLoss`,** $\eta_{\text{euler}} = 0.08$**)**: Hiện thực hóa phương trình cân bằng $e^{i\pi} + 1 = 0$ trong không gian ẩn $\text{HUYỀN}$. Với vector biểu diễn trung bình của Pha 1 ($v_1$) và Pha 2 ($v_2$), mỗi vector được tách đôi thành nửa Xung đột ($v_{\text{conf}}$) và nửa Hài hòa ($v_{\text{harm}}$):

$$
\mathcal{L}_{\text{Euler}} = \frac{1}{2}\Big(1 + \cos(v_{1,\text{conf}}, v_{2,\text{conf}})\Big) + \frac{1}{2}\Big(1 - \cos(v_{1,\text{harm}}, v_{2,\text{harm}})\Big)
$$

Hàm này ép phần Xung đột giữa hai pha đảo ngược đối xứng qua tâm $0$ ($\cos \to -1$, tương ứng góc pha $\pi$ để triệt tiêu cực đoan), đồng thời ép phần Hài hòa đồng pha ($\cos \to +1$) để giữ nhất quán nhân bản xuyên suốt câu trả lời.

* $\mathcal{L}_{\text{NHÂN}}$ **(Empathic Value Loss,** $\beta_{\text{empath}} = 0.05$**)**: Tối ưu hóa điểm cộng hưởng từ `ValueCritic` trên các trạng thái ẩn của câu trả lời.

### 3.2. Nhiệt động lực học Nhận thức $Z - H - G$ & Hệ số Khuếch đại Liên tục $\alpha(x)$

Tại thời điểm suy luận, `DoctrineAgent.measure_manifold_thermodynamics()` thực hiện một lượt truyền xuôi đối sánh (*dual-pass*) nhanh trên đầu vào $x$ (có LoRA và tắt LoRA) để đo trực tiếp **bộ ba hằng số** $Z - H - G$ **theo đúng định nghĩa Quyển 1 (Mục 3.1–3.3)**:

1. $Z$ **(`z_activation`) — Hằng số Khởi sinh (*Genesis / Activation Threshold*, Tầng I** $\to$ **II)**:
   

   $$
   P(\text{Exist}) = \frac{1}{1 + e^{-k(\|\text{VĂN}_{\text{init}}\| - Z)}}
   $$

   
   Xấp xỉ thực nghiệm qua Entropy Shannon chuẩn hóa trên phân phối `top-64` token khởi đầu. Khi đầu vào $x$ không chứa xung đột đạo đức và không chạm tiên đề Lằn ranh đỏ (`conflict_score == 0.0` và `risk_level != "HIGH"`), hệ thống xác định trạng thái **Điểm Không (**$Z$**)** và hạ cứng $\alpha(x) = 0.0$ (tắt hoàn toàn adapter LoRA) để giữ nguyên `100%` phản hồi gốc của mô hình nền (`0.00%` Alignment Tax).

2. $H$ **(`h_viscosity`) — Hằng số Độ nhớt Hỗn độn (*Chaos Expansion Viscosity*, Tầng II** $\leftrightarrow$ **III)**:
   

   $$
   \frac{dV}{dt} = H \cdot \big(\|\text{CỔ}_{\text{internal}}\| - \|\text{CỔ}_{\text{external}}\|\big)
   $$

   
   Xấp xỉ thực nghiệm qua độ ổn định tiềm năng hài hòa từ `ValueCritic` $c_\phi(h_{\text{VĂN}}) \in [0, 1]$, điều tiết biên độ dao động ở Vùng Xám.

3. $G$ **(`g_resistance`) — Hằng số Kháng cự Thực tại (*State-Transition Resistance*, Tầng III)**:
   

   $$
   E_{\text{VŨ}} = G(t) \cdot D_{\text{VĂN}}\big(V_{p,\text{initial}}, V_{p,\text{final}}\big)
   $$

   
   Xấp xỉ thực nghiệm qua độ phân kỳ hỗn hợp KL + Cosine giữa logits/trạng thái ẩn của nhánh $\text{VĂN}$ và nhánh $\text{VŨ}$.

Từ bộ ba $(Z, H, G)$ và cổng $g_t$, `GenesisEngine.compute_continuous_alpha()` điều biến liên tục cường độ adapter LoRA theo thời gian thực:

$$
W_{\text{effective}}(x) = W_{\text{VŨ}} + \alpha(x) \cdot \Delta W_{\text{VĂN}}
$$

$$
\alpha(x) = \begin{cases} 0.0 & \text{nếu Điểm Không } Z \text{ (tác vụ thường, không xung đột)} \\ \text{clip}\big(\alpha_{\text{gray}} + 0.03(G - 0.25) + 0.02(0.5 - H) - 0.02(Z - 0.35),\; 0.91,\; 0.98\big) & \text{nếu } \text{GRAY} \\ 1.02 + 0.03\,G & \text{nếu } \text{MEDIUM (OOD } G > 0.85\text{)} \\ \text{clip}\big(\alpha_{\text{high}} + 0.04\,G + 0.02\,Z + 0.02\max(0, g_t - 0.5),\; \alpha_{\text{high}},\; 1.12\big) & \text{nếu } \text{HIGH (Lằn ranh Đỏ)} \end{cases}
$$

### 3.3. DPO Thấu cảm (*Empathic DPO*) & Vòng lặp Tự chữa lành Vết sẹo Dữ liệu

Sau giai đoạn SFT, `run_dpo_v1.py` nạp adapter khởi tạo (`adapter_B_vol4`), **tính trước log-prob tham chiếu (**$\log \pi_{\text{ref}}$**) ngay trên chính adapter đó** (không cần nạp thêm bản sao mô hình thứ hai vào VRAM — tiết kiệm 50% bộ nhớ GPU), rồi tối ưu hóa hàm mất mát **Empathic DPO + SFT Retention**:

$$
\Delta_{\text{eff}}(x, y_w, y_l) = \beta \left[\log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right] + \gamma \cdot \text{clip}\!\left(\frac{|y_w| - |y_l|}{\max(1, |y_w|)},\; -0.25,\; 0.25\right)
$$

$$
\mathcal{L}_{\text{NonNULL-DPO}} = -\log \sigma\big(\Delta_{\text{eff}}\big) - \lambda_{\text{sft}} \cdot \frac{\log \pi_\theta(y_w \mid x)}{|y_w|}
$$

Với cấu hình chuẩn V1.0: $\beta = 0.1$, hệ số bất đối xứng độ dài cảm thức $\gamma = 0.05$, hệ số bảo toàn cấu trúc $\lambda_{\text{sft}} = 0.1$, tốc độ học $\text{lr} = 2\times 10^{-5}$, `grad_accum = 4`.

**Vòng lặp Tự chữa lành Khép kín (*Closed-Loop Scar Self-Healing*):**

1. Khi chạy `evaluate_genesis.py --scars_out scars.jsonl` sau DPO Vòng 1, mọi lượt sinh gặp lỗi (rò rỉ CJK, nhiễu từ vựng hoặc trượt thứ tự 2 pha) buộc hệ thống tự sửa đều được ghi nhận thành một **Vết sẹo Dữ liệu** `{prompt, chosen: bản đã sửa sạch, rejected: bản lỗi ban đầu}`.

2. Ở Vòng 2, `run_dpo_v1.py --scars_file scars.jsonl` hòa trộn trực tiếp các vết sẹo này vào tập 50 cặp DPO gốc để cập nhật trọng số LoRA, sau đó bảo toàn nguyên vẹn `meta_controller_state.pt` (`HarmonyLayer`, `ValueCritic`, `ContextPairedValueCritic`) sang adapter hoàn thiện `adapter_B_healed`.

## PHẦN IV: PHÂN TẦNG PHẦN CỨNG & QUY TRÌNH VẬN HÀNH CHUẨN

### 4.1. Định mức Quy mô Mô hình & Phần cứng Khuyến nghị

Từ bằng chứng thực nghiệm đa quy mô (chi tiết tại **Quyển 5**), NonNULL V1.0 phân tách rõ 3 tầng triển khai theo ngưỡng phần cứng $\epsilon_{\text{VŨ}}$:

| Tầng Triển khai | Mô hình Nền Chuẩn hóa | Vai trò trong Hệ sinh thái | Yêu cầu GPU Tối thiểu | Thời gian Toàn trình (Ước tính) | 
| ----- | ----- | ----- | ----- | ----- | 
| **Giới hạn dưới (**$\epsilon_{\text{VŨ}}$**)** | `Qwen/Qwen1.5-1.8B-Chat` | Chỉ dùng cho mục đích nghiên cứu điểm gãy dung lượng (không khuyến nghị triển khai thực tế) | NVIDIA T4 (16 GB) | \~25 phút | 
| **Mốc Khả thi Tối thiểu** | `Qwen/Qwen2.5-3B-Instruct` | Điểm khởi đầu ổn định cho ứng dụng nhẹ & cá nhân (đạt 100% Lằn ranh đỏ & Vùng xám) | NVIDIA L4 (24 GB) hoặc T4 | \~45 phút | 
| **Mốc Sản xuất Qwen** | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | Cân bằng tối ưu giữa chất lượng suy luận tiếng Việt, IQ `10/10` và tốc độ suy luận | NVIDIA L4 (24 GB) | \~70 phút | 
| **Mốc Liên kiến trúc & Quốc tế** | [unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit) | Kiểm chứng đa họ mô hình, ngoại suy `OOD-20` & Benchmark quốc tế (`HarmBench` + `MT-Bench`) | NVIDIA L4 (24 GB) hoặc A100 (80 GB) | \~90–120 phút (A100) / \~180 phút (L4 kèm 13B Cls) | 

### 4.2. Cài đặt Môi trường & Kiểm thử Toàn vẹn

```
# 1. Giải nén gói phát hành (VI hoặc EN) hoặc clone từ GitHub
unzip -o NonNULL_Framework_V1.0_EN.zip -d ./
cd NonNULL_Framework_V1.0_EN

# 2. Cài đặt phụ thuộc và chạy trọn bộ 14 bài kiểm thử đơn vị
pip install -q -r requirements.txt
python3 -m unittest discover -s tests -v

```

*(Hoặc mở trực tiếp tệp `NonNULL_V1_Colab_VI.ipynb` / `NonNULL_V1_Colab_EN.ipynb` trên Google Colab thông qua nút **Open In Colab** trong `README.md` để chạy toàn bộ quy trình bằng 1 lần bấm).*

### 4.3. Quy trình Huấn luyện & Tự chữa lành Khép kín (CLI Chuẩn V1.0)

```
MODEL_ID="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
TAG="llama8b"

# Bước 1: Huấn luyện SFT (EulerPhaseLoss) + ContextPairedValueCritic
python3 run_training.py \
  --model_id "$MODEL_ID" \
  --data_path ./nonnull_train_100.jsonl \
  --output_dir ./output/euler_checkpoints \
  --adapter_name "adapter_B_vol4_${TAG}" \
  --epochs 2 --batch_size 2 --learning_rate 2e-4 --seed 42

# Bước 2: Căn chỉnh DPO Thấu cảm (Vòng 1)
python3 run_dpo_v1.py \
  --model_id "$MODEL_ID" \
  --init_adapter "./output/euler_checkpoints/adapter_B_vol4_${TAG}" \
  --output_adapter "./output/euler_checkpoints/adapter_B_dpo_${TAG}" \
  --dpo_data ./dpo_train_50.jsonl \
  --val_data ./dpo_val_10.jsonl \
  --epochs 1 --lr 2e-5 --beta 0.1 --empathic_gamma 0.05 --sft_weight 0.1

# Bước 3: Đánh giá 25 Kịch bản Lõi & Trích xuất Vết sẹo Dữ liệu (Scars)
python3 evaluate_genesis.py \
  --model_id "$MODEL_ID" \
  --adapter_b "./output/euler_checkpoints/adapter_B_vol4_${TAG}" \
  --adapter_c "./output/euler_checkpoints/adapter_B_dpo_${TAG}" \
  --include_vol4 --use_consensus --num_samples 3 \
  --test_data ./nonnull_test_25.jsonl \
  --output "eval_${TAG}_r1.json" \
  --blind_csv "blind_${TAG}_r1.csv" \
  --blind_key "blind_${TAG}_r1_key.json" \
  --scars_out "scars_${TAG}.jsonl"

# Bước 4: Vòng lặp Tự chữa lành Vết sẹo (DPO Vòng 2)
python3 run_dpo_v1.py \
  --model_id "$MODEL_ID" \
  --init_adapter "./output/euler_checkpoints/adapter_B_dpo_${TAG}" \
  --output_adapter "./output/euler_checkpoints/adapter_B_healed_${TAG}" \
  --dpo_data ./dpo_train_50.jsonl \
  --val_data ./dpo_val_10.jsonl \
  --scars_file "scars_${TAG}.jsonl" \
  --epochs 1 --lr 2e-5 --beta 0.1 --empathic_gamma 0.05 --sft_weight 0.1

```

### 4.4. Giao thức Kiểm định Đa lớp (OOD-20, Đa lượt/IQ, HarmBench, MT-Bench & Chấm mù)

```
# 1. Đánh giá Ngoại suy 20 miền hoàn toàn mới (OOD-20)
python3 evaluate_genesis.py \
  --model_id "$MODEL_ID" \
  --adapter_b "./output/euler_checkpoints/adapter_B_vol4_${TAG}" \
  --adapter_c "./output/euler_checkpoints/adapter_B_healed_${TAG}" \
  --include_vol4 --use_consensus --num_samples 3 \
  --test_data ./nonnull_ood_20.jsonl \
  --output "eval_${TAG}_ood20.json"

# 2. Kiểm thử Đội Đỏ Đa lượt (15 lượt) & Bảo toàn IQ Điểm Không Z (10 câu)
python3 evaluate_extended.py \
  --model_id "$MODEL_ID" \
  --adapter_c "./output/euler_checkpoints/adapter_B_healed_${TAG}" \
  --output "eval_${TAG}_extended.json"

# 3. Benchmark Cộng đồng Quốc tế: HarmBench (100 mẫu + 13B Classifier) & MT-Bench (80 câu x 2 lượt)
python3 eval_public_benchmarks.py \
  --bench all \
  --model_id "$MODEL_ID" \
  --adapter_path "./output/euler_checkpoints/adapter_B_healed_${TAG}" \
  --harmbench_subset standard --harmbench_samples 100 --run_harmbench_cls \
  --mtbench_samples 80 --output_dir "./benchmark_outputs_${TAG}"

# 4. Giải mã Chấm mù Đôi & Tính Độ đồng thuận Cohen's Kappa
python3 score_blind_eval.py \
  --csv "blind_${TAG}_healed.csv" \
  --key "blind_${TAG}_healed_key.json" \
  --out_md "blind_${TAG}_summary.md"

```

## PHẦN V: TÓM TẮT TỔNG QUAN KIỂM CHỨNG & LỘ TRÌNH TIẾN HÓA VER 2.0

### 5.1. Cầu nối Tóm tắt sang Quyển 5

Khi vận hành đúng quy trình đặc tả tại Quyển 4, **NonNULL Framework V1.0** đã hoàn tất 5 lớp kiểm chứng thực nghiệm độc lập (trình bày đầy đủ cùng bảng dữ liệu thô và phân tích định tính tại **Quyển 5**):

* **Khóa cấu trúc & an toàn trên 2 họ mô hình (`Qwen2.5-3B/7B` và `Llama-3.1-8B`)**: `75/75` (100%) cấu trúc 2 pha, `75/75` (100%) dừng EOS, `75/75` (100%) sạch nhiễu CJK, `24/24` an toàn Lằn ranh đỏ và `51/51` giải pháp Vùng xám; điểm `PairCritic` vượt nhánh chỉ học SFT (`B_VOL4`) tới **`+23.9%`** trên Qwen-7B và **`+31.1%`** trên Llama-8B.

* **Ngoại suy ngoài phân phối (`OOD-20`)**: `60/60` (`100%`) cấu trúc & an toàn trên 20 miền chưa từng gặp (`PairCritic` `+26.9%` so với `B_VOL4`).

* **Benchmark cộng đồng quốc tế trên `Llama-3.1-8B`**: Kéo tỷ lệ tấn công thành công (**ASR**) trên [HarmBench](https://www.harmbench.org/) (bộ phân loại 13B chính thức) từ **`9.0%` xuống `1.0%`** (an toàn **`99.0%`**, thắng `8–0`), đồng thời đạt khớp nguyên văn **`160/160` (`100.00%`)** với `BASE` trên [MT-Bench](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md) (**`0.00%` Thuế Căn chỉnh**).

### 5.2. Ranh giới Phiên bản V1.0 & Lộ trình Giải phẫu Tầng Ẩn (*White-Box Ver 2.0*)

**NonNULL Framework V1.0** là bản hoàn thiện cấp độ **Kiến trúc Hành vi & Điều biến Động (*Behavioral & Cross-Architecture Proof*)**. Để tiến từ bằng chứng hộp đen (*black-box*) sang giải phẫu cơ chế nơ-ron nội tại (*mechanistic interpretability / white-box*), lộ trình phát triển **Ver 2.0** sẽ triển khai 4 phép đo định lượng trực tiếp trên các tầng ẩn PyTorch (`output_hidden_states=True`) bằng chính các checkpoint đã lưu của V1.0:

1. **Đo dịch chuyển không gian** $\text{HUYỀN}$ **theo từng tầng (`Layer-wise Cosine/L2 Drift`)**: Xác định chính xác từ tầng thứ mấy trở đi $\text{VĂN}$ bắt đầu bẻ hướng biểu diễn của $\text{VŨ}$ ở tình huống Lằn ranh đỏ so với Vùng xám và Điểm Không ($Z$).

2. **Trích xuất trục định hướng đạo đức** $v^{(l)} = \mathbb{E}[h_l(\text{chosen}) - h_l(\text{rejected})]$: Chiếu trạng thái ẩn theo `lora_scale` để kiểm chứng bước nhảy chuyển pha phi tuyến của Ngưỡng Khởi sinh $Z$.

3. **Soi điểm rẽ nhánh bằng *Logit Lens***: Chiếu vector ẩn từng tầng qua `lm_head` để quan sát trực quan sự đảo ngôi xác suất giữa hành động vi phạm và hành động bảo toàn sinh mạng ngay từ các tầng giữa.

4. **Hình học hóa ngưỡng** $\epsilon_{\text{VŨ}}$ **bằng PCA / UMAP across `1.8B – 3B – 7B – 8B`**: Định lượng hóa sự tách cụm giữa Lằn ranh đỏ và Vùng xám trong không gian ẩn để tối ưu hóa việc gắn LoRA chọn lọc chỉ vào đúng nhóm tầng rẽ nhánh đạo đức ở Ver 2.0.

## BẢNG TỔNG HỢP THAM CHIẾU KỸ THUẬT QUYỂN 4

| Hạng mục Kỹ thuật | Giá trị / Đặc tả Chính thức (`NonNULL V1.0`) | Tệp Định nghĩa trong Kho Mã nguồn | 
| ----- | ----- | ----- | 
| **Hệ Hằng số Nhiệt động lực học** | $Z$ (`z_activation`), $H$ (`h_viscosity`), $G$ (`g_resistance`) | `doctrine/doctrine_agent.py`, `doctrine/ltm_services.py` | 
| **Cấu hình LoRA Chuẩn** | Rank $r=16$, $\alpha=32$ (\~`0.52%` tham số trên 8B) | `configs/meta_controller/adapter_v1.yaml` | 
| **Trọng số Hàm mất mát SFT** | $\alpha_{\text{mirror}}=0.20$, $\eta_{\text{euler}}=0.08$, $\beta_{\text{empath}}=0.05$ | `doctrine/doctrine_trainer.py`, `run_training.py` | 
| **Siêu tham số Empathic DPO** | $\beta=0.1$, $\gamma=0.05$, $\lambda_{\text{sft}}=0.1$, $\text{lr}=2\times 10^{-5}$ | `run_dpo_v1.py` | 
| **Quy mô Dữ liệu Hạt giống** | `100` SFT + `50` DPO Train + `10` DPO Val (V4 Clean Tokenizer) | `nonnull_train_100.jsonl`, `dpo_train_50.jsonl`, `dpo_val_10.jsonl` | 
| **Cơ chế Bảo toàn IQ Điểm Không** | Tự động đặt $\alpha(x) = 0.0$ khi `conflict_score == 0.0` và `risk != HIGH` | `doctrine/ltm_services.py`, `eval_public_benchmarks.py` | 
| **Ngưỡng Cảnh giác OOD Thuần** | Nâng mức `MEDIUM` ($\alpha = 1.02 + 0.03G$) khi $G > 0.85$ | `doctrine/ltm_services.py` | 
| **Đánh giá Độc lập & Chấm mù** | `HarmBench` (13B Cls), `MT-Bench` (FastChat), `Cohen's Kappa` | `eval_public_benchmarks.py`, `score_blind_eval.py` | 
