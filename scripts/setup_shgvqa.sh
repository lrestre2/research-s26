#!/bin/bash
# Sets up SHG-VQA in ~/Liu/research-s26/SHG-VQA and installs all dependencies.
# Run once on the GPU lab before anything else.
#
# Usage:
#   bash scripts/setup_shgvqa.sh

set -e
RESEARCH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "=== Setting up SHG-VQA under $RESEARCH_DIR ==="

# ── 1. Activate conda environment (created earlier) ─────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate srp26

# ── 2. Clone SHG-VQA ────────────────────────────────────────────────────
if [ ! -d "$RESEARCH_DIR/SHG-VQA" ]; then
    echo "[1/3] Cloning SHG-VQA..."
    git clone https://github.com/aurooj/SHG-VQA.git "$RESEARCH_DIR/SHG-VQA"
else
    echo "[1/3] SHG-VQA already cloned, pulling latest..."
    git -C "$RESEARCH_DIR/SHG-VQA" pull
fi

# ── 3. Install SHG-VQA dependencies ─────────────────────────────────────
echo "[2/3] Installing SHG-VQA dependencies..."
pip install -q \
    einops \
    timm \
    transformers \
    pytorchvideo \
    fvcore \
    av

# Install our adapter's extra dependencies
pip install -q \
    opencv-python-headless \
    ffmpeg-python \
    accelerate \
    bitsandbytes

# ── 4. Create processed data directories ────────────────────────────────
echo "[3/3] Creating data directories..."
mkdir -p ~/data/mmau/processed/frames
mkdir -p ~/data/mmau/processed/detections
mkdir -p ~/data/mmau/processed/scene_graphs

echo ""
echo "=== Done! ==="
echo "Next: python mmau_adapter/download_partial.py"
