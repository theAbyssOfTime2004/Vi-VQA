#!/bin/bash
#
# Set up a local environment for FVQA.
#
#   bash setup_vlm.sh
#
# Dependencies are declared in pyproject.toml, not here — this script only
# creates the virtualenv and installs the extras.

set -euo pipefail

cd "$(dirname "$0")"

echo "🚀 Setting up FVQA"

if command -v nvcc &> /dev/null; then
    echo "✓ CUDA $(nvcc --version | grep release | sed 's/.*release //; s/,.*//')"
else
    echo "⚠️  No CUDA toolkit found. Training needs a GPU; inference will be very slow without one."
fi

if [ ! -d ".venv" ]; then
    echo "Creating .venv..."
    python3 -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip

# Qwen3-VL needs transformers>=4.57; pyproject pins that. Install torch with
# the CUDA wheels first so pip does not resolve to the CPU build.
echo "Installing PyTorch (CUDA 12.1 wheels)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "Installing fvqa and its training extras..."
pip install -e '.[train,dev]'

echo "Installing Flash Attention 2 (optional)..."
pip install flash-attn --no-build-isolation || \
    echo "⚠️  Flash Attention unavailable — set model.use_flash_attn: false in config/config.yaml"

echo
echo "✅ Done."
echo
echo "Next:"
echo "  source .venv/bin/activate"
echo "  # Download and extract FVQA (not on HuggingFace, no login needed):"
echo "  curl -L -o fvqa.zip 'https://www.dropbox.com/s/iyz6l7jhbt6jb7q/new_dataset_release.zip?dl=1'"
echo "  unzip fvqa.zip -d data/fvqa"
echo "  fvqa prepare"
echo "  fvqa train"
