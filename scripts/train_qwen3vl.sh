#!/bin/bash
#
# Fine-tune Qwen3-VL on Vi-VQA locally.
#
#   bash scripts/train_qwen3vl.sh                    # use config/config.yaml as-is
#   bash scripts/train_qwen3vl.sh --dry-run          # print the command, run nothing
#   bash scripts/train_qwen3vl.sh --num-gpus 4
#   bash scripts/train_qwen3vl.sh --set training.num_train_epochs=3
#
# Hyperparameters live in config/config.yaml, not here. This script only
# checks the environment and hands over to the CLI, so the local run and
# the Modal run execute an identical command.

set -euo pipefail

cd "$(dirname "$0")/.."

if ! python3 -c "import vivqa" 2>/dev/null; then
    echo "❌ The vivqa package is not importable."
    echo "   Install it with: pip install -e '.[train]'"
    exit 1
fi

if [ ! -f "./data/train.json" ]; then
    echo "❌ No training data at ./data/train.json"
    echo "   Prepare it with: vivqa prepare"
    exit 1
fi

echo "🚀 Fine-tuning Qwen3-VL on Vi-VQA"
python3 -m vivqa.cli train "$@"

echo
echo "✅ Done. Checkpoints are under training.output_dir in config/config.yaml."
echo "   Monitor with: tensorboard --logdir ./checkpoints/qwen3vl-vivqa"
