#!/bin/bash
# Run this on the GPU lab TONIGHT before you sleep.
# It sets up the env, downloads the dataset, and runs the analysis.
# Output plots will be in ~/research-s26/scripts/outputs/
#
# Usage (on GPU lab):
#   bash run_overnight.sh
#
# Monitor progress anytime with:
#   tail -f ~/overnight.log

LOG=~/overnight.log
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

exec > >(tee -a "$LOG") 2>&1

echo "========================================"
echo "  SRP26 Overnight Setup — $(date)"
echo "========================================"

# ── 1. Conda env ────────────────────────────────────────────────────────
echo ""
echo "[1/4] Setting up conda environment..."
conda create -n srp26 python=3.11 -y 2>/dev/null || echo "  (env already exists, skipping)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate srp26

pip install -q \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -q \
    huggingface_hub datasets pandas numpy matplotlib seaborn \
    opencv-python-headless Pillow tqdm jupyter ipykernel scikit-learn
python -m ipykernel install --user --name srp26 --display-name "SRP26" -q

echo "  Done."

# ── 2. Clone the research repo ───────────────────────────────────────────
echo ""
echo "[2/4] Cloning research scripts..."
mkdir -p ~/research-s26
cp -r "$SCRIPTS_DIR"/../* ~/research-s26/ 2>/dev/null || true
echo "  Done."

# ── 3. Download MM-AU ────────────────────────────────────────────────────
echo ""
echo "[3/4] Downloading MM-AU dataset (this is the slow part — may take hours)..."
cd ~/research-s26/scripts
python download_mmau.py
echo "  Done."

# ── 4. Run analysis ──────────────────────────────────────────────────────
echo ""
echo "[4/4] Running dataset analysis and generating plots..."
python explore_mmau.py
echo "  Done."

echo ""
echo "========================================"
echo "  All done! — $(date)"
echo "  Plots are in: ~/research-s26/scripts/outputs/"
echo "  Read: cat ~/research-s26/scripts/outputs/dataset_stats.txt"
echo "========================================"
