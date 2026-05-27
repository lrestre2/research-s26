"""
Partial MM-AU Download — targets ~100 GB
=========================================
The HuggingFace repo stores videos as chunked split archives:
    CAP-DATA_chunks/11/11.part_aa
    CAP-DATA_chunks/11/11.part_ab
    ...
These must be downloaded, concatenated, and extracted.

Strategy:
  1. Download annotation files (tiny, always first)
  2. Download CAP-DATA_chunks/11/  — ego-car hitting car  (~category 11)
  3. Download CAP-DATA_chunks/43/  — car hitting car       (~category 43)
  4. Concatenate parts → reconstruct archive → extract → delete chunks

Usage:
    conda activate srp26
    cd ~/Liu/research-s26
    python mmau_adapter/download_partial.py
"""

import os
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download, list_repo_files

REPO_ID   = "JeffreyChou/MM-AU"
DATA_DIR  = Path.home() / "data" / "mmau"
CHUNK_DIR = DATA_DIR / "_chunks"     # temporary landing spot for parts
MAX_GB    = 100

def gb(n_bytes):
    return f"{n_bytes / 1e9:.2f} GB"

def dir_size(path):
    total = 0
    for f in Path(path).rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total

def download_chunks(patterns: list[str]):
    """Download specific chunk folders from HuggingFace."""
    print(f"  Downloading: {patterns}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(CHUNK_DIR),
        allow_patterns=patterns,
        ignore_patterns=["*.git*"],
    )

def reconstruct_and_extract(chunk_folder: Path, out_dir: Path):
    """
    Concatenate all .part_* files in chunk_folder into one archive,
    detect the format, extract it, then delete the chunks.
    """
    parts = sorted(chunk_folder.glob("*.part_*"))
    if not parts:
        print(f"  No parts found in {chunk_folder}")
        return False

    # The archive name is the folder name (e.g. "11" or "43")
    archive_name = chunk_folder.name
    archive_path = DATA_DIR / f"{archive_name}.archive"

    print(f"  Concatenating {len(parts)} parts → {archive_path.name} ...")
    with open(archive_path, "wb") as out:
        for part in parts:
            with open(part, "rb") as p:
                out.write(p.read())

    # Detect format by magic bytes
    with open(archive_path, "rb") as f:
        magic = f.read(8)

    out_dir.mkdir(parents=True, exist_ok=True)
    success = False

    if magic[:4] == b'PK\x03\x04':           # ZIP
        print(f"  Detected: ZIP — extracting ...")
        result = subprocess.run(
            ["unzip", "-q", str(archive_path), "-d", str(out_dir)],
            capture_output=True
        )
        success = result.returncode == 0

    elif magic[:2] == b'\x1f\x8b':           # GZIP / tar.gz
        print(f"  Detected: tar.gz — extracting ...")
        result = subprocess.run(
            ["tar", "-xzf", str(archive_path), "-C", str(out_dir)],
            capture_output=True
        )
        success = result.returncode == 0

    elif magic[:6] == b'7z\xbc\xaf\x27\x1c': # 7-Zip
        print(f"  Detected: 7z — extracting ...")
        result = subprocess.run(
            ["7z", "x", str(archive_path), f"-o{out_dir}", "-y"],
            capture_output=True
        )
        success = result.returncode == 0

    else:
        # Try tar as a last resort (handles uncompressed tar)
        print(f"  Unknown format — trying tar ...")
        result = subprocess.run(
            ["tar", "-xf", str(archive_path), "-C", str(out_dir)],
            capture_output=True
        )
        success = result.returncode == 0
        if not success:
            print(f"  Could not extract. Magic bytes: {magic.hex()}")
            print(f"  Archive left at: {archive_path}")
            return False

    if success:
        print(f"  Extracted to {out_dir}")
        # Clean up chunks and concatenated archive to save space
        archive_path.unlink()
        for part in parts:
            part.unlink()
        # Remove the now-empty chunk folder
        try:
            chunk_folder.rmdir()
        except OSError:
            pass
        return True

    print(f"  Extraction failed. Archive: {archive_path}")
    return False


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: annotations ─────────────────────────────────────────────
    print("[1/3] Downloading annotation files ...")
    download_chunks([
        "cap_text_annotations.xls",
        "dada_text_annotations.xlsx",
        "video_metadata.json",
        "README.md",
    ])
    # Move annotations to DATA_DIR
    for f in CHUNK_DIR.glob("*.xls*"):
        f.rename(DATA_DIR / f.name)
    for f in CHUNK_DIR.glob("*.json"):
        f.rename(DATA_DIR / f.name)
    print("  Done.\n")

    # ── Step 2: CAP-DATA category 11 ────────────────────────────────────
    print("[2/3] Downloading category 11 (ego-car hitting car) ...")
    download_chunks(["CAP-DATA_chunks/11/*"])
    cat11_chunks = CHUNK_DIR / "CAP-DATA_chunks" / "11"
    if cat11_chunks.exists():
        reconstruct_and_extract(cat11_chunks, DATA_DIR / "CAP-DATA")
    used = dir_size(DATA_DIR)
    print(f"  Used so far: {gb(used)}\n")

    # ── Step 3: CAP-DATA category 43 (if under limit) ───────────────────
    if used < MAX_GB * 1_000_000_000:
        print("[3/3] Downloading category 43 (car hitting car) ...")
        download_chunks(["CAP-DATA_chunks/43/*"])
        cat43_chunks = CHUNK_DIR / "CAP-DATA_chunks" / "43"
        if cat43_chunks.exists():
            reconstruct_and_extract(cat43_chunks, DATA_DIR / "CAP-DATA")
        used = dir_size(DATA_DIR)
        print(f"  Used so far: {gb(used)}\n")
    else:
        print(f"[3/3] Already at {gb(used)} — skipping category 43.")

    print(f"Final dataset size : {gb(dir_size(DATA_DIR))}")
    print(f"Data saved to      : {DATA_DIR}")
    print("\nNext: python mmau_adapter/preprocess.py")

if __name__ == "__main__":
    main()
