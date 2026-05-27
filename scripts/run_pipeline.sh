#!/bin/bash
# Master pipeline — run this overnight on the GPU lab.
# It does everything in the correct order:
#   setup → download → preprocess → train baseline
#
# Usage:
#   cd ~/Liu/research-s26
#   nohup bash scripts/run_pipeline.sh > ~/pipeline.log 2>&1 &
#
# Monitor with:
#   tail -f ~/pipeline.log

set -e
RESEARCH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG=~/pipeline.log

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate srp26

cd "$RESEARCH_DIR"

echo "========================================"
echo "  SRP26 Full Pipeline — $(date)"
echo "========================================"

echo ""
echo "[1/4] Setting up SHG-VQA..."
bash scripts/setup_shgvqa.sh

echo ""
echo "[2/4] Downloading 100GB of MM-AU..."
python mmau_adapter/download_partial.py

echo ""
echo "[3/4] Preprocessing (frames + YOLO + scene graphs)..."
python mmau_adapter/preprocess.py --workers 8

echo ""
echo "[4/4] Training baseline model (10 epochs)..."
python mmau_adapter/run_baseline.py --epochs 10 --batch 8

echo ""
echo "========================================"
echo "  Pipeline complete — $(date)"
echo "  Results: runs/mmau_baseline/per_category_accuracy.json"
echo "========================================"
