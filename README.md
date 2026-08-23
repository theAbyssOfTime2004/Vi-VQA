# Vi-VQA: Visual Question Answering tiếng Việt với Qwen3-VL

Fine-tune **Qwen3-VL-8B-Instruct** bằng LoRA trên
[Viet-ViTextVQA-gemini-VQA](https://huggingface.co/datasets/5CD-AI/Viet-ViTextVQA-gemini-VQA)
— 9.594 ảnh, 31.420 cặp hỏi–đáp về di tích, biển hiệu và sản phẩm Việt Nam.

## Cài đặt

```bash
git clone <repo-url> && cd Vi-VQA
python3 -m venv .venv && source .venv/bin/activate

pip install -e '.[train]'                 # data + inference + training
pip install flash-attn --no-build-isolation   # tuỳ chọn, nhanh hơn đáng kể

huggingface-cli login                     # dataset là gated, cần request access
```

`transformers>=4.57.0` là yêu cầu bắt buộc, không phải khuyến nghị:
`Qwen3VLForConditionalGeneration` không tồn tại ở bản thấp hơn.

## Dùng

Mọi thứ đi qua một CLI duy nhất. Hyperparameter nằm ở `config/config.yaml`,
không rải trong code, và `--set` cho phép override tại chỗ.

```bash
vivqa config                              # in ra cấu hình đã resolve
vivqa prepare                             # HF dataset -> train/val/test JSON + ảnh
vivqa train                               # fine-tune LoRA
vivqa eval --model-path ./checkpoints/qwen3vl-vivqa/checkpoint-1500
vivqa chat --model-path ./checkpoints/qwen3vl-vivqa/checkpoint-1500
```

```bash
# Ví dụ override
vivqa prepare --limit 100                          # thử nhanh trên 100 record
vivqa train --dry-run                              # in lệnh train, không chạy
vivqa train --num-gpus 4 --set training.num_train_epochs=3
vivqa eval --model-path <ckpt> --num-samples -1 --output results.json
```

Chạy trên Modal:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_...
modal run scripts/train_on_modal.py --step all
modal run scripts/train_on_modal.py::check_status
```

Modal và local dùng **chung** module sinh lệnh train, nên hai đường chạy không
thể lệch nhau.

## Knowledge grounding

Dataset có sẵn trường `description` (~558 ký tự/ảnh) mà Gemini đã dùng để viết
câu trả lời — tức là kiến thức nằm sẵn trong dữ liệu nhưng trước đây bị bỏ phí
hoàn toàn. Bật nó lên:

```bash
vivqa prepare --set data.grounding.enabled=true
```

Chi tiết, hai chế độ `prefix`/`system`, và quy trình A/B để đo tác động:
[`docs/GROUNDING.md`](docs/GROUNDING.md).

## Cấu trúc

```
config/config.yaml          # nguồn cấu hình duy nhất cho mọi đường chạy
src/vivqa/
  config.py                 # load + validate config thành dataclass
  cli.py                    # vivqa prepare | train | eval | chat | config
  model.py                  # nạp Qwen3-VL, sinh câu trả lời
  data/
    prepare.py              # HF dataset -> JSON theo format Qwen-VL
    grounding.py            # ngữ cảnh từ description
  train/
    command.py              # config -> argv cho trainer
    runner.py               # clone trainer repo và chạy
  evaluation/
    metrics.py              # exact match, similarity, BLEU, ROUGE-L, CIDEr
    runner.py               # sinh dự đoán và chấm điểm
scripts/
  train_qwen3vl.sh          # wrapper mỏng cho chạy local
  train_on_modal.py         # phần thuộc về Modal, không chứa logic dataset
tests/                      # 118 test, chạy không cần GPU
notebooks/eda.ipynb         # phân tích dataset
```

## Kiến trúc

**Model:** Qwen3-VL-8B-Instruct — vision encoder ViT, LLM 8B, projector
vision-language, hỗ trợ OCR 32 ngôn ngữ trong đó có tiếng Việt.

**Fine-tune:** LoRA rank 128 / alpha 256 trên LLM và projector, đóng băng vision
encoder. Batch hiệu dụng 16 (1 × 16 gradient accumulation), bf16, gradient
checkpointing — vừa 24GB VRAM.

**Vì sao generative chứ không phải classification:** EDA (`notebooks/eda.ipynb`)
cho thấy 39.886 câu trả lời unique, top-5000 chỉ cover 17,6% dataset. Không có
tập nhãn cố định nào phủ nổi bài toán này.

Training thực tế được giao cho
[`2U1/Qwen-VL-Series-Finetune`](https://github.com/2U1/Qwen-VL-Series-Finetune);
repo này lo dữ liệu, cấu hình và đánh giá.

## Đánh giá

```
exact_match   % trùng khít sau khi chuẩn hoá (NFC, lowercase, bỏ dấu câu)
similarity    % tương đồng mức ký tự
bleu          BLEU-4 mức corpus, có smoothing
rouge_l       F-measure trên LCS, beta=1.2
cider         CIDEr-D, thang 0–10
```

Chỉ dùng exact match cho bài toán này là vô nghĩa: câu trả lời trung bình ~49 ký
tự tiếng Việt tự do, một câu đúng nhưng diễn đạt khác sẽ được 0 điểm. Các metric
còn lại chấm điểm từng phần.

Chuẩn hoá NFC không phải chi tiết thẩm mỹ: "à" có thể được lưu bằng một codepoint
hoặc bằng "a" cộng dấu huyền tổ hợp. Hai dạng hiển thị y hệt nhau, so sánh ra
khác nhau, và dataset có cả hai.

## Phát triển

```bash
pip install -e '.[dev]'
pytest                       # 118 test, không cần torch/transformers
```

Tầng config, data và metrics cố tình không import torch, nên test chạy trong vài
giây trên máy không GPU.

## Yêu cầu phần cứng

| GPU | Cấu hình | Thời gian (2 epoch) |
|-----|----------|---------------------|
| RTX 3080 (10GB) | QLoRA 4-bit | ~30h |
| RTX 3090 (24GB) | LoRA | ~20–25h |
| A100 (40GB) | LoRA | ~8–12h |
| H100 | LoRA | ~4–6h |

Hết VRAM thì giảm `image_max_pixels` trước, rồi mới tới batch size:

```bash
vivqa train --set model.image_max_pixels=589824 \
            --set training.gradient_accumulation_steps=32
```

## Tham khảo

- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Viet-ViTextVQA-gemini-VQA](https://huggingface.co/datasets/5CD-AI/Viet-ViTextVQA-gemini-VQA)
- [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune)
- [ViTextVQA paper — arXiv:2404.10652](https://arxiv.org/abs/2404.10652)
- [LoRA — arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

## Giấy phép

MIT
