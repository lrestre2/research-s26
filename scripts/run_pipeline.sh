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
echo "[2/4] Checking data..."
VIDEO_COUNT=$(find ~/data/mmau/CAP-DATA -name "*.mp4" -o -name "*.avi" 2>/dev/null | wc -l)
if [ "$VIDEO_COUNT" -gt 0 ]; then
    echo "  Found $VIDEO_COUNT videos already — skipping download."
else
    echo "  No videos found — downloading 100GB..."
    python mmau_adapter/download_partial.py
fi

echo ""
echo "[3/4] Preprocessing (zero-shot scene graphs)..."
python mmau_adapter/preprocess.py

echo ""
echo "[3b/4] Analysing scene graphs..."
python mmau_adapter/analyze_scene_graphs.py

echo ""
echo "[4/4] Training baseline model (10 epochs)..."
python mmau_adapter/run_baseline.py --epochs 10 --batch 8

echo ""
echo "========================================"
echo "  Pipeline complete — $(date)"
echo "  Results: runs/mmau_baseline/per_category_accuracy.json"
echo "========================================"
