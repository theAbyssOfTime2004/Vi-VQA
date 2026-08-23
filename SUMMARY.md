# Vi-VQA — quyết định thiết kế và nhật ký thay đổi

Tài liệu hướng dẫn sử dụng nằm ở [README.md](README.md). File này ghi lại *vì
sao* project trông như hiện tại.

---

## Bài toán

Trả lời câu hỏi tiếng Việt về ảnh, trên
`5CD-AI/Viet-ViTextVQA-gemini-VQA`: 9.594 ảnh, 31.420 cặp QA do Gemini 1.5
Flash sinh, tập trung vào di tích, biển hiệu, chợ và sản phẩm Việt Nam. Nhiều
chữ trong ảnh — đây là bài toán scene-text VQA, không chỉ nhận dạng vật thể.

| Chỉ số | Giá trị |
|--------|---------|
| Ảnh | 9.594 |
| Cặp QA | 31.420 (trung bình 3,27/ảnh) |
| Câu trả lời unique | 39.886 |
| Độ dài câu hỏi | ~37 ký tự |
| Độ dài câu trả lời | ~49 ký tự |
| Độ dài description | ~558 ký tự |

## Quyết định 1 — Generative, không phải classification

EDA (`notebooks/eda.ipynb`) đo độ phủ của tập nhãn cố định:

| Top-K câu trả lời | Độ phủ dataset |
|-------------------|----------------|
| 100 | 0,95% |
| 1.000 | 4,88% |
| 5.000 | 17,61% |

Ngay cả câu trả lời phổ biến nhất ("Đây là khu chợ.") cũng chỉ xuất hiện 14 lần
trên 31.420 mẫu. Không tồn tại tập nhãn nào phủ nổi bài toán này — phân phối
long-tail bắt buộc phải dùng model sinh.

Hai hướng đã loại:

- **ViT + PhoBERT → softmax 1000 lớp.** Trần độ chính xác dưới 5% vì lý do trên.
- **Seq2seq huấn luyện từ đầu.** Đắt, và vứt bỏ toàn bộ kiến thức pre-trained.

Chọn: **Qwen3-VL-8B-Instruct + LoRA.** Đã biết tiếng Việt, đã biết OCR, fine-tune
được trên GPU 24GB.

## Quyết định 2 — Không tự viết training loop

Training giao cho [`2U1/Qwen-VL-Series-Finetune`](https://github.com/2U1/Qwen-VL-Series-Finetune).
Repo này chỉ lo dữ liệu, cấu hình và đánh giá — ba phần mà một trainer chung
không thể làm thay.

Đổi lại, phải chấp nhận phụ thuộc vào command-line của repo ngoài. Toàn bộ phần
dịch từ config sang command line gom vào một chỗ duy nhất là
`src/vivqa/train/command.py`, nên khi repo đó đổi flag chỉ phải sửa một file.

## Quyết định 3 — Knowledge grounding từ `description`

Dataset có sẵn trường `description` mà Gemini dùng để viết câu trả lời. Rất
nhiều câu trả lời vì thế khẳng định điều không đọc được từ pixel. Đưa
description trở lại prompt biến bài toán thành đọc-hiểu thay vì bắt model nhớ
kiến thức nó chưa từng được cho xem.

Tắt mặc định — đây là giả thuyết cần đo, không phải cải tiến hiển nhiên. Quy
trình A/B: [`docs/GROUNDING.md`](docs/GROUNDING.md).

---

## Đại tu (v0.2.0)

### Vấn đề gốc: bốn bản sao của cùng một logic

Logic chuẩn bị dataset tồn tại ở bốn nơi — `src/dataset_vlm.py`,
`scripts/train_on_modal.py`, `notebooks/train_on_colab.ipynb`,
`notebooks/quick_test.ipynb` — và đã lệch nhau. Bản Modal có chia train/val, bản
`src/` thì không. Notebook đã sửa class model, `src/` thì chưa. Sửa một chỗ
không lan sang chỗ khác, nên bug tích tụ.

`config/config.yaml` tồn tại nhưng **không đường train nào đọc nó**.

### Sau khi sửa

Một package `vivqa` là nguồn duy nhất; local, Modal và CLI đều gọi vào đó.
`config.yaml` thực sự điều khiển mọi thứ, có validate, có override `--set`.
`scripts/train_on_modal.py` giảm từ 1.047 xuống 257 dòng vì phần logic đã
chuyển vào package.

### Lỗi đã sửa

| # | Lỗi | Hậu quả |
|---|-----|---------|
| 1 | Rò rỉ dữ liệu giữa train/val | Modal shuffle theo *cặp QA* rồi mới chia, nên các câu hỏi về cùng một ảnh nằm ở cả hai phía. Model đã thấy ảnh validation lúc train, val loss đẹp hơn thực tế. Nay chia theo **ảnh**. |
| 2 | Sai class model | `Qwen2VLForConditionalGeneration` cho checkpoint Qwen3-VL. Notebook đã sửa, `src/` và Modal thì chưa. |
| 3 | Sai entrypoint train | Gọi `train.py` ở gốc repo trainer — file không tồn tại. Đúng là `deepspeed src/train/train_sft.py`. Script local và notebook Colab đều hỏng ngay dòng đầu. |
| 4 | Hard-code flash attention | Bắt buộc `flash_attention_2` kể cả khi flash-attn không cài; mọi lần load đều lỗi. Nay tự lùi về SDPA. |
| 5 | `src/inspect_data.py` import module không tồn tại | Tàn dư từ giai đoạn classification. Đã xoá. |
| 6 | Hard-code path máy cá nhân | `/home/maidang/projects/...` trong `dataset_vlm.py`. |
| 7 | Lệch version transformers | `requirements.txt` yêu cầu `>=4.57.0`, image Modal ghi `>=4.45.0` — có thể resolve về bản không có Qwen3-VL. |
| 8 | `test.json` không bao giờ được sinh | README hướng dẫn dùng nó. Nay `prepare` sinh cả ba split. |
| 9 | Config lệch code | `config.yaml` ghi bs=2/8, 3 epoch; Modal chạy bs=1/16, 2 epoch. |
| 10 | Metric không như quảng cáo | README hứa BLEU/ROUGE/CIDEr, code chỉ có exact match. |
| 11 | Đếm sample bằng `wc -l` trên file JSON | Ra số vô nghĩa. |
| 12 | `2e-5` trong YAML parse ra **string** | YAML 1.1 cần `2.0e-5`. Config gốc viết `2e-5` ở mọi learning rate. Nay chấp nhận cả hai. |

Lỗi 12 do chính test bắt được, không phải do đọc code.

### Metric

Exact match một mình vô dụng ở bài toán này: câu trả lời trung bình 49 ký tự
tiếng Việt tự do, đúng nhưng diễn đạt khác vẫn 0 điểm. Đã thêm BLEU-4 mức
corpus, ROUGE-L, CIDEr-D và similarity, tất cả viết bằng thư viện chuẩn.

Chuẩn hoá NFC là bắt buộc: "à" lưu được bằng một codepoint hoặc "a" cộng dấu
huyền tổ hợp — hiển thị y hệt, so sánh ra khác, và dataset có cả hai dạng.

### Test

133 test, không cần torch/transformers/GPU — các tầng config, data và metric cố
tình không import thư viện nặng, nên `pytest` chạy trong vài giây.

---

## Kết quả đo được (2026-08-23)

Baseline zero-shot trên L4, 200 mẫu, greedy, `max_new_tokens=128`, không
grounding, **không fine-tune gì**. Khác biệt duy nhất giữa hai cột là một câu
chỉ dẫn văn phong đặt ở lượt system.

| Metric | A1 — prompt trần | A2 — có chỉ dẫn văn phong | Tăng |
|--------|-----------------:|--------------------------:|-----:|
| exact_match | 0,50% | **23,50%** | 47× |
| bleu | 6,85% | **60,36%** | 8,8× |
| rouge_l | 26,06% | 69,56% | 2,7× |
| similarity | 23,20% | 73,53% | 3,2× |
| cider (0–10) | 0,08 | **5,25** | 66× |
| Độ dài trung vị | 480 ký tự (10,1× đáp án chuẩn) | 40 ký tự (0,8×) | — |

### Diễn giải

Đọc tay 30 mẫu của A1 cho kết quả: **26/30 đúng nội dung nhưng sai văn phong**,
3/30 sai suy luận, 1/30 sai OCR. Tức `exact_match = 0,50%` trong khi ~87% câu
trả lời thực chất đúng — metric bề mặt sai lệch gần **170 lần** so với thực tế.

Model gốc trả lời đúng rồi viết tiếp vài đoạn nữa. Ví dụ sạch nhất: hỏi số điện
thoại, model trả về `0927 333323 - 0376877777` **giống hệt từng ký tự** đáp án
chuẩn, nhưng bọc trong một đoạn giải thích nên mọi metric đều rớt.

Hệ quả cho việc fine-tune: câu hỏi không còn là *"có đáng train không so với
0,5%?"* mà là *"train mua thêm được gì so với 23,5% mà một dòng prompt cho
không?"*. Phần lớn mức tăng before/after mà một run fine-tune sẽ báo cáo là
**tuân thủ định dạng**, không phải năng lực trả lời.

### OCR không phải nút thắt

Giả thuyết ban đầu cho rằng scene-text tiếng Việt là chỗ yếu. Dữ liệu bác bỏ:
model đọc chính xác `invaquangcaochuyennghiep.com`, `0927 333323 - 0376877777`,
`KM 944 + 937,11`, `0931.16.61.16`. Chỉ 1/30 lỗi OCR thật (`VIETGAP` đọc thành
`VƯƠNG CÁP`).

### Nói dài thì bịa

Trong A1, model tự mâu thuẫn về cùng một tấm ảnh: mẫu 110_0 nói Cầu Thanh Quýt
ở Hà Nam, mẫu 110_1 nói Bình Thuận. Mẫu 102_1 bịa nguyên tiểu sử cho một
thương hiệu nó đọc sai tên. Mẫu 101_0 rơi vào vòng lặp thoái hoá, sinh rác cho
đến khi hết token.

Cả ba biến mất ở A2. **Câu ngắn thì ít chỗ để bịa** — nên "chỉ học định dạng"
không thuần tuý mỹ phẩm: nó cắt bề mặt hallucination.

### Nhãn có nhiễu

~10% mẫu đọc tay có nhãn hỏng hoặc câu hỏi mơ hồ (`104_1` hỏi "Mỳ gạo chũ được
làm gì?" với đáp án "được HTX sản xuất và tiêu thụ"; `105_0` hỏi "Bức hình này
của ai?"), cộng câu hỏi trùng lặp (`108_0` và `108_1`). Trần thực tế của
exact_match vì thế vào khoảng 85–90%, không phải 100%.

### Cảnh báo phương pháp

Hai con số trên đều đo trên split `train` và prompt được chọn trên chính tập
đó. Với số liệu công bố, hãy chọn prompt trên `train` rồi báo cáo trên `val`
hoặc `test` (`vivqa eval --split val`).

---

## Trạng thái

Pipeline sẵn sàng, **model chưa được train**.

Baseline ở trên đã đo xong (`notebooks/baseline_a1.ipynb`). Việc còn lại:

1. **Tinh chỉnh prompt.** Đáp án Gemini theo khuôn cứng: nhắc lại chủ ngữ câu
   hỏi rồi nêu đáp án (*"Tên của hàng này là Anytime."*). A2 trả lời cộc hơn
   đáp án chuẩn (40 so với 48 ký tự) vì prompt bảo "không giải thích". Mô tả
   đúng khuôn này có thể đẩy exact_match lên 35–45% — vẫn không train gì.
2. **LoRA nhỏ** (~1.500 mẫu, 1 epoch, 1–2 giờ) để đo phần dư mà prompt không
   lấy được. Đây là con số trung thực về giá trị của fine-tune, chứ không phải
   phép so với baseline trần.
3. **Full fine-tune 8–20 giờ**: dữ liệu hiện có không biện minh được.

### Hướng mở rộng

1. Đo A/B tác động của grounding theo `docs/GROUNDING.md`.
2. RAG: index description + scene-text của 9.594 ảnh, truy hồi lúc infer.
   `apply_grounding` đã là chỗ nối sẵn.
3. KB thực thể có cấu trúc cho entity linking và kiểm tra hallucination.
4. Deploy thành API.
