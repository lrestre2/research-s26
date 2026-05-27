"""
Partial MM-AU Download — targets ~100 GB
=========================================
Strategy:
  - Always download: video_metadata.json (already done, skipped if present)
  - Priority 1: CAP-DATA/11/  — ego-car hitting car  (~3,300 videos, ~66 GB)
  - Priority 2: CAP-DATA/43/  — car hitting car       (~2,350 videos, fills to ~100 GB)
  - Stop automatically once the downloaded data exceeds MAX_GB.

Why these two?
  Categories 11 and 43 account for ~47% of the dataset and represent
  the two most common real-world accident types. Having clean coverage
  of the dominant classes gives us enough signal to train and evaluate
  a baseline, and lets us confirm the class-imbalance hypothesis.

Usage (on GPU lab, inside srp26 env):
    conda activate srp26
    cd ~/Liu/research-s26
    python mmau_adapter/download_partial.py
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download, list_repo_files

REPO_ID   = "JeffreyChou/MM-AU"
DATA_DIR  = Path.home() / "data" / "mmau"
MAX_GB    = 100
MAX_BYTES = MAX_GB * 1_000_000_000   # 100 GB

# Folders to download in priority order.
# HuggingFace stores MM-AU mirroring the original directory structure.
PRIORITY_PATTERNS = [
    "video_metadata.json",          # annotations — tiny, always first
    "CAP-DATA/11/*",                # ego-car hitting car
    "CAP-DATA/43/*",                # car hitting car
]

def get_dir_size(path: Path) -> int:
    """Return total bytes used under path."""
    total = 0
    if path.exists():
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total

def gb(n_bytes: int) -> str:
    return f"{n_bytes / 1e9:.2f} GB"

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MM-AU (up to {MAX_GB} GB) → {DATA_DIR}\n")

    for pattern in PRIORITY_PATTERNS:
        used = get_dir_size(DATA_DIR)
        print(f"  Used so far : {gb(used)}")

        if used >= MAX_BYTES:
            print(f"  Reached {MAX_GB} GB limit — stopping.")
            break

        print(f"  Downloading pattern : {pattern}")
        try:
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                local_dir=str(DATA_DIR),
                allow_patterns=[pattern],
                ignore_patterns=["*.git*"],
            )
            after = get_dir_size(DATA_DIR)
            print(f"  Done. (+{gb(after - used)})\n")
        except Exception as e:
            print(f"  Warning: failed to download {pattern}: {e}\n")

    total = get_dir_size(DATA_DIR)
    print(f"\nFinal dataset size: {gb(total)}")
    print(f"Data saved to     : {DATA_DIR}")
    print("\nNext: python mmau_adapter/preprocess.py")

if __name__ == "__main__":
    main()
