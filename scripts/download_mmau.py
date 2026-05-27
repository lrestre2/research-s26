"""
Download the MM-AU dataset from HuggingFace.
Run on the GPU lab inside the srp26 conda environment.

Usage:
    conda activate srp26
    python download_mmau.py
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

DATA_DIR = Path.home() / "data" / "mmau"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading MM-AU dataset to {DATA_DIR} ...")
print("This will take a while — videos are large. Let it run overnight.\n")

# Downloads annotations + videos from HuggingFace
snapshot_download(
    repo_id="JeffreyChou/MM-AU",
    repo_type="dataset",
    local_dir=str(DATA_DIR),
    ignore_patterns=["*.git*"],
)

print(f"\nDone. Dataset saved to {DATA_DIR}")
print("Now run: python explore_mmau.py")
