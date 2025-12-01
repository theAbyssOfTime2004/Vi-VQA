#!/bin/bash

# Training script for Qwen3-VL on Vi-VQA dataset
# Usage: bash scripts/train_qwen3vl.sh

set -e

# Configuration
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
DATA_PATH="./data/train.json"
IMAGE_FOLDER="./data/images"
OUTPUT_DIR="./checkpoints/qwen3vl-vivqa"

# Training hyperparameters
NUM_EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8  # Effective batch size = 2 * 8 = 16
LEARNING_RATE=2e-5
VISION_LR=2e-6
MERGER_LR=2e-5

# LoRA config
LORA_RANK=128
LORA_ALPHA=256
LORA_DROPOUT=0.05

# Image resolution (token × 32 × 32 for Qwen3-VL)
IMAGE_MIN_PIXELS=$((256 * 32 * 32))    # 262144
IMAGE_MAX_PIXELS=$((1280 * 32 * 32))   # 1310720

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Training data not found at $DATA_PATH"
    echo "Please run: python src/dataset_vlm.py"
    exit 1
fi

if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "❌ Image folder not found at $IMAGE_FOLDER"
    exit 1
fi

echo "🚀 Starting Qwen3-VL training on Vi-VQA dataset"
echo "Model: $MODEL_ID"
echo "Data: $DATA_PATH ($(wc -l < $DATA_PATH) samples)"
echo "Output: $OUTPUT_DIR"
echo ""

# Clone Qwen-VL-Series-Finetune if not exists
if [ ! -d "Qwen-VL-Series-Finetune" ]; then
    echo "Cloning Qwen-VL-Series-Finetune repository..."
    git clone https://github.com/2U1/Qwen-VL-Series-Finetune.git
fi

cd Qwen-VL-Series-Finetune

# Run training with LoRA
python train.py \
    --model_id $MODEL_ID \
    --data_path ../$DATA_PATH \
    --image_folder ../$IMAGE_FOLDER \
    --output_dir ../$OUTPUT_DIR \
    \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    \
    --learning_rate $LEARNING_RATE \
    --vision_lr $VISION_LR \
    --merger_lr $MERGER_LR \
    \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --num_lora_modules -1 \
    \
    --image_min_pixels $IMAGE_MIN_PIXELS \
    --image_max_pixels $IMAGE_MAX_PIXELS \
    \
    --freeze_vision_tower true \
    --freeze_llm false \
    --freeze_merger false \
    \
    --optim adamw_bnb_8bit \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    \
    --bf16 true \
    --gradient_checkpointing true \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 4 \
    \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    \
    --logging_steps 50 \
    --report_to tensorboard \
    \
    --disable_flash_attn2 false

cd ..

echo ""
echo "✅ Training complete!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo ""
echo "To merge LoRA weights:"
echo "cd Qwen-VL-Series-Finetune && bash scripts/merge_lora.sh"
