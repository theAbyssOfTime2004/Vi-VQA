# Vi-VQA: Vietnamese Visual Question Answering with Qwen3-VL

Dự án Visual Question Answering (VQA) cho tiếng Việt sử dụng **Qwen3-VL-8B-Instruct** - một trong những Vision Language Model mạnh nhất hiện tại.

## 📊 Dataset

**Viet-ViTextVQA-gemini-VQA** ([HuggingFace](https://huggingface.co/datasets/5CD-AI/Viet-ViTextVQA-gemini-VQA))
- **9,594 images** từ ViTextVQA dataset
- **31,420 QA pairs** được sinh bởi Google Gemini 1.5 Flash
- Domain: Di tích lịch sử Việt Nam, landmarks, sản phẩm, v.v.
- Multi-turn conversations với câu trả lời generative

### Dataset Statistics (từ EDA):
```
- Trung bình 3.27 QA pairs/image
- Độ dài câu hỏi: ~37 ký tự
- Độ dài câu trả lời: ~49 ký tự
- 39,886 unique answers → Generative task, không phải classification
```

## 🏗️ Architecture

**Model:** Qwen3-VL-8B-Instruct
- **Vision Encoder:** ViT-based visual encoder
- **Language Model:** 8B parameter Qwen3 LLM
- **Multimodal Fusion:** Vision-language projector
- **Context Length:** 256K native, expandable to 1M tokens
- **OCR Support:** 32 languages including Vietnamese

**Fine-tuning Strategy:**
- LoRA (Low-Rank Adaptation) với rank=128
- Freeze vision encoder, chỉ tune LLM + projector
- Batch size: 2 × 8 gradient accumulation = 16 effective
- Learning rate: 2e-5 (LLM), 2e-6 (vision)

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <your-repo-url>
cd Vi-VQA

# Run setup script
bash setup_vlm.sh

# Or manual installation:
python3 -m venv Vi-VQA
source Vi-VQA/bin/activate
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### 2. Login to HuggingFace

```bash
huggingface-cli login
# Enter your token: hf_...
```

Hoặc trong Python:
```python
from huggingface_hub import login
login(token="hf_...")
```

### 3. Prepare Dataset

```bash
# Convert dataset to Qwen3-VL format
python src/dataset_vlm.py
```

Kết quả:
- `data/train.json`: Training data in Qwen-VL format
- `data/images/`: Extracted images

### 4. Train Model

```bash
# Start training with LoRA
bash scripts/train_qwen3vl.sh
```

Training sẽ:
- Clone Qwen-VL-Series-Finetune repository
- Train với LoRA adapters
- Save checkpoints mỗi 500 steps
- Log to TensorBoard

**Xem training progress:**
```bash
tensorboard --logdir ./checkpoints/qwen3vl-vivqa/runs
```

### 5. Inference

**Interactive mode:**
```bash
python src/inference_qwen3vl.py \
    --model_path ./checkpoints/qwen3vl-vivqa \
    --mode interactive
```

**Evaluation mode:**
```bash
python src/inference_qwen3vl.py \
    --model_path ./checkpoints/qwen3vl-vivqa \
    --mode eval \
    --test_data ./data/test.json \
    --output ./predictions.json
```

## 📁 Project Structure

```
Vi-VQA/
├── config/
│   └── config.yaml              # Configuration file
├── data/
│   ├── images/                  # Extracted images
│   ├── train.json              # Training data (Qwen-VL format)
│   └── test.json               # Test data
├── src/
│   ├── dataset_vlm.py          # Dataset processor for VLM
│   ├── inference_qwen3vl.py    # Inference script
│   ├── utils.py                # Utilities
│   └── vocab.py                # (Legacy, not used for VLM)
├── scripts/
│   └── train_qwen3vl.sh        # Training script
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis
├── checkpoints/                 # Model checkpoints
├── logs/                        # Training logs
├── requirements.txt            # Python dependencies
├── setup_vlm.sh               # Environment setup script
└── README.md                   # This file
```

## ⚙️ Configuration

Edit `config/config.yaml` để thay đổi hyperparameters:

```yaml
model:
  vlm:
    model_id: "Qwen/Qwen3-VL-8B-Instruct"
    lora:
      enabled: true
      rank: 128
      alpha: 256
      dropout: 0.05

training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2e-5
  freeze_vision_tower: true
```

## 📊 Evaluation Metrics

- **Exact Match Accuracy:** % câu trả lời giống hệt ground truth
- **BLEU Score:** Độ tương đồng n-gram
- **ROUGE Score:** Recall-oriented overlap
- **CIDEr:** Consensus-based metric

## 🔬 EDA Insights

Từ `notebooks/eda.ipynb`:

1. **Dataset không phù hợp cho Classification:**
   - Top 1000 answers chỉ cover 4.88% dataset
   - Top 5000 answers chỉ cover 17.61%
   - → Cần generative model

2. **Phân bố Conversations:**
   - Không balanced: phần lớn images có 2-3 QA pairs
   - Long-tail: một số ít có >10 QA pairs

3. **Text Statistics:**
   - Câu hỏi ngắn (~5-8 từ)
   - Câu trả lời đa dạng (1-100+ từ)
   - max_length=2048 là đủ

## 🛠️ Advanced Usage

### Train with QLoRA (4-bit)

Edit `config.yaml`:
```yaml
model:
  vlm:
    qlora:
      enabled: true
```

### Multi-GPU Training

```bash
# Use DeepSpeed
deepspeed --num_gpus=4 \
    Qwen-VL-Series-Finetune/train.py \
    --deepspeed ds_config_zero2.json \
    ...
```

### Merge LoRA Weights

```bash
cd Qwen-VL-Series-Finetune
bash scripts/merge_lora.sh
```

## 📚 References

- **Qwen3-VL:** [HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- **Dataset:** [Viet-ViTextVQA-gemini-VQA](https://huggingface.co/datasets/5CD-AI/Viet-ViTextVQA-gemini-VQA)
- **Fine-tuning Repo:** [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune)
- **ViTextVQA Paper:** [arXiv:2404.10652](https://arxiv.org/abs/2404.10652)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License (hoặc license của bạn)

## ⚠️ Notes

- Qwen3-VL requires transformers>=4.57.0 (install from source)
- Flash Attention 2 highly recommended cho tốc độ
- Dataset là gated, cần request access trên HuggingFace
- Training với full dataset (~30k samples) mất ~10-15 giờ trên A100

## 🆘 Troubleshooting

**OOM Error:**
```bash
# Giảm batch size hoặc enable QLoRA
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
```

**Flash Attention Error:**
```bash
# Disable flash attention
--disable_flash_attn2 true
```

**Dataset Loading Error:**
```bash
# Check HuggingFace authentication
huggingface-cli whoami
```
