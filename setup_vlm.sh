#!/bin/bash

# Setup script for Qwen3-VL training environment
# Usage: bash setup_vlm.sh

set -e

echo "🚀 Setting up Vi-VQA with Qwen3-VL..."

# Check CUDA version
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA found: $(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')"
else
    echo "⚠️  CUDA not found. This is required for training."
fi

# Activate virtual environment
if [ -d "Vi-VQA" ]; then
    echo "✓ Virtual environment found"
    source Vi-VQA/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv Vi-VQA
    source Vi-VQA/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install transformers from source (required for Qwen3-VL)
echo "Installing transformers from source..."
pip install git+https://github.com/huggingface/transformers

# Install core dependencies
echo "Installing core dependencies..."
pip install -r requirements.txt

# Install flash-attention (optional but highly recommended)
echo "Installing Flash Attention 2..."
pip install flash-attn --no-build-isolation || echo "⚠️  Flash Attention installation failed. You can continue without it."

# Install qwen-vl-utils
echo "Installing qwen-vl-utils..."
pip install qwen-vl-utils

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate environment: source Vi-VQA/bin/activate"
echo "2. Login to HuggingFace: huggingface-cli login"
echo "3. Prepare dataset: python src/dataset_vlm.py"
echo "4. Start training: bash scripts/train_qwen3vl.sh"
echo ""
