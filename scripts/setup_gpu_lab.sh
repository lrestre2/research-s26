#!/bin/bash
# Run this ONCE on the GPU lab after SSH-ing in.
# Usage: bash setup_gpu_lab.sh

set -e

echo "=== Setting up research environment ==="

# Create and activate a dedicated conda environment
conda create -n srp26 python=3.11 -y
source activate srp26

# Core ML stack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Data / analysis
pip install \
    huggingface_hub datasets \
    pandas numpy matplotlib seaborn \
    opencv-python-headless \
    Pillow tqdm \
    jupyter ipykernel \
    scikit-learn

# Register the kernel so Jupyter sees it
python -m ipykernel install --user --name srp26 --display-name "SRP26"

echo ""
echo "=== Done. Activate with: conda activate srp26 ==="
echo "=== Then run: python download_mmau.py            ==="
