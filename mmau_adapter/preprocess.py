"""
MM-AU Preprocessor — v2
========================
Generates scene graphs for all downloaded videos using the zero-shot
pipeline (Grounding DINO + LLaVA). Replaces v1 which used YOLO + hand-
crafted spatial rules.

Output format (one JSON per video — as Hyun specified):
  ~/data/mmau/processed/scene_graphs/{video_id}.json

  {
    "video_id"   : "ABC123",
    "category"   : 11,
    "metadata"   : { "weather": 1, "light": 1, "t_ai": 57.0, "t_ae": 82.0 },
    "scene_graphs": [
      { "frame_idx": 0, "timestamp_s": 0.0, "objects": [...], "relations": [...] },
      { "frame_idx": 1, "timestamp_s": 3.0, "objects": [...], "relations": [...] },
      ...  one entry per 3-second window (e.g. 15s video → 5 entries)
    ]
  }

Frame sampling: 1 frame every FRAME_INTERVAL_S seconds (default 3s).
For a 10-second accident window that gives ~4 frames; 15s → 5 frames.

Usage:
    conda activate srp26
    cd ~/Liu/research-s26
    python mmau_adapter/preprocess.py [--limit N] [--spatial-only]
"""

import json
import argparse
from pathlib import Path

from tqdm import tqdm
from PIL import Image

from mmau_adapter.scene_graph_gen import ZeroShotSGG

# ── paths ──────────────────────────────────────────────────────────────
DATA_DIR   = Path.home() / "data" / "mmau"
META_FILE  = DATA_DIR / "video_metadata.json"
# Actual path on disk after extraction (double CAP-DATA from archive structure)
CAP_DIR    = DATA_DIR / "CAP-DATA" / "CAP-DATA"
SGRAPH_DIR = DATA_DIR / "processed" / "scene_graphs"
N_FRAMES   = 5     # one frame per 3 seconds for a ~15s clip (Hyun's spec)

# Categories we've downloaded
DOWNLOADED_CATEGORIES = {"11", "43"}


def _get_category_from_path(img_dir: Path) -> str:
    """
    Extract the accident category from a path like:
    CAP_DIR / {cat} / {cat} / {video_id} / images
    Returns the category string, or "" if not recognised.
    """
    for part in img_dir.parts:
        if part in DOWNLOADED_CATEGORIES:
            return part
    return ""


def _scan_all_image_dirs() -> list[tuple[Path, str]]:
    """
    Walk CAP_DIR once and return (images_dir, category) pairs
    for every video whose category we have downloaded.
    """
    result = []
    for p in CAP_DIR.rglob("images"):
        if p.is_dir():
            cat = _get_category_from_path(p)
            if cat:
                result.append((p, cat))
    return result


def load_frames_evenly(
    frame_dir: Path,
    n: int = N_FRAMES,
) -> list[tuple[int, float, Image.Image]]:
    """
    Load n frames evenly spaced from the images/ directory.
    Returns list of (frame_idx, timestamp_s, PIL.Image).
    Timestamp is estimated assuming 10 fps (DADA dataset standard).
    """
    jpg_files = sorted(frame_dir.glob("*.jpg"))
    if not jpg_files:
        return []

    # Pick n evenly-spaced indices
    total = len(jpg_files)
    indices = [int(i * (total - 1) / (n - 1)) for i in range(n)] if total >= n \
              else list(range(total))

    FPS = 10.0   # DADA/CAP datasets use 10fps
    frames = []
    for out_idx, file_idx in enumerate(indices):
        f = jpg_files[file_idx]
        frame_number = int(f.stem)           # filename is the frame number e.g. 000191
        timestamp_s  = round(frame_number / FPS, 2)
        img = Image.open(f).convert("RGB")
        frames.append((out_idx, timestamp_s, img))

    return frames



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",        type=int,  default=None,
                        help="Process only the first N videos (for testing)")
    parser.add_argument("--spatial-only", action="store_true",
                        help="Skip LLaVA; use spatial relations only (fast mode)")
    args = parser.parse_args()

    # ── Load metadata (best-effort — used for accident timing, weather etc.) ──
    print("Loading metadata ...")
    meta_by_id: dict[str, dict] = {}
    try:
        with open(META_FILE) as f:
            raw = json.load(f)
        records = raw if isinstance(raw, list) else list(raw.values())
        for r in records:
            for key in ("video_hashcode", "video_name", "id"):
                val = str(r.get(key, "")).strip()
                if val:
                    meta_by_id[val] = r
        print(f"  Metadata loaded: {len(records)} records, {len(meta_by_id)} ID keys")
    except Exception as e:
        print(f"  Warning: metadata load failed ({e}). Continuing without it.")

    # ── Scan disk for actual video directories ─────────────────────────────
    print("\nScanning for video directories (this takes ~30s on first run) ...")
    img_dirs = _scan_all_image_dirs()
    print(f"  Found {len(img_dirs)} videos in categories {DOWNLOADED_CATEGORIES}")

    if not img_dirs:
        print("ERROR: No image directories found under", CAP_DIR)
        return

    if args.limit:
        img_dirs = img_dirs[:args.limit]

    print(f"Videos to process : {len(img_dirs)}")
    print(f"Frames per video  : {N_FRAMES} (evenly sampled)\n")

    SGRAPH_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load models once ──────────────────────────────────────────────────
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    sgg = ZeroShotSGG(device=device, spatial_only=args.spatial_only)

    # ── Process sequentially (LLaVA is not thread-safe) ───────────────────
    done = fail = skip = 0
    for img_dir, category in tqdm(img_dirs, desc="Generating scene graphs"):
        video_id = img_dir.parent.name          # e.g. "011460"
        out_path = SGRAPH_DIR / f"{video_id}.json"

        if out_path.exists():
            skip += 1
            continue

        frames = load_frames_evenly(img_dir, n=N_FRAMES)
        if not frames:
            fail += 1
            continue

        try:
            scene_graphs = [
                sgg.generate(img, frame_idx=fi, timestamp_s=ts)
                for fi, ts, img in frames
            ]
        except Exception as e:
            print(f"\n  Error on {video_id}: {e}")
            fail += 1
            continue

        # Merge with metadata if available (try several key forms)
        record = (meta_by_id.get(video_id)
                  or meta_by_id.get(video_id.lstrip("0"))
                  or {})

        output = {
            "video_id"    : video_id,
            "category"    : int(category),
            "metadata"    : {
                "weather": record.get("weather"),
                "light"  : record.get("light"),
                "scenes" : record.get("scenes"),
                "t_ai"   : record.get("t_ai"),
                "t_ae"   : record.get("t_ae"),
            },
            "scene_graphs": scene_graphs,
        }
        out_path.write_text(json.dumps(output, indent=2))
        done += 1

    print(f"\nDone: {done} | Skipped: {skip} | Failed: {fail}")
    print(f"Scene graphs saved to: {SGRAPH_DIR}")
    print("\nNext: python mmau_adapter/analyze_scene_graphs.py")


if __name__ == "__main__":
    main()
