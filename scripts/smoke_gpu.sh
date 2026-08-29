#!/bin/bash
#
# The smallest run that actually proves the training and evaluation
# pipeline works on a GPU.
#
#   bash scripts/smoke_gpu.sh
#
# Two epochs on the real dataset is an expensive way to discover that the
# processor rejects an image or that the adapter cannot be reloaded. This
# runs two optimizer steps instead and then reloads what they produced,
# which exercises every step that can fail:
#
#   1. the model loads onto the GPU
#   2. the data loader finds and decodes the images
#   3. a batch reaches the forward pass
#   4. gradients flow through the backward pass
#   5. an adapter checkpoint is written
#   6. `fvqa eval` reloads that adapter (base + non-LoRA weights + adapter)
#   7. one sample generates an answer
#
# Everything goes through the normal CLI, so what is exercised here is the
# same code a real run uses — no separate training path to drift.

set -euo pipefail

cd "$(dirname "$0")/.."

SMOKE_OUT="${SMOKE_OUT:-./checkpoints/smoke}"
SMOKE_LIMIT="${SMOKE_LIMIT:-50}"

echo "🔥 GPU smoke test — output: $SMOKE_OUT"

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "❌ No CUDA GPU visible. This script needs one; there is nothing to smoke-test without it."
    exit 1
fi

echo
echo "── 1/3  prepare a small slice ─────────────────────────────────────"
python3 -m fvqa.cli prepare --limit "$SMOKE_LIMIT"

echo
echo "── 2/3  train two optimizer steps ─────────────────────────────────"
# save_steps=1 so a checkpoint exists after step 1 even if step 2 is the
# last one; eval off because a 50-question slice may leave val.json tiny.
python3 -m fvqa.cli train \
    --set training.max_steps=2 \
    --set training.save_steps=1 \
    --set training.eval_strategy=no \
    --set training.load_best_model_at_end=false \
    --set training.logging_steps=1 \
    --set "training.output_dir=$SMOKE_OUT"

echo
echo "── 3/3  reload the adapter and generate one answer ────────────────"
CHECKPOINT=$(ls -d "$SMOKE_OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
if [ -z "$CHECKPOINT" ]; then
    echo "❌ Training wrote no checkpoint under $SMOKE_OUT — step 5 failed."
    exit 1
fi
echo "reloading $CHECKPOINT"

python3 -m fvqa.cli eval \
    --model-path "$CHECKPOINT" \
    --num-samples 1 \
    --split train \
    --output "$SMOKE_OUT/smoke_eval.json"

echo
echo "✅ Smoke test passed: trained, checkpointed, reloaded and generated."
echo "   Checkpoint: $CHECKPOINT"
echo "   Result:     $SMOKE_OUT/smoke_eval.json"
