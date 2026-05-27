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


def find_frame_dir(video_name: str, category: str) -> Path | None:
    """
    Find the images/ directory for a given video.
    Structure: CAP-DATA/CAP-DATA/{category}/{video_name}/images/
    Also searches recursively in case of slight path variations.
    """
    # Direct lookup first (fast)
    direct = CAP_DIR / category / video_name / "images"
    if direct.exists():
        return direct

    # Recursive fallback — search by video_name folder anywhere under CAP_DIR
    for p in CAP_DIR.rglob(f"{video_name}/images"):
        return p

    return None


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


def process_one(record: dict, sgg: ZeroShotSGG) -> tuple[str, str]:
    """
    Process one video: extract frames, generate scene graphs, save JSON.
    Returns (status, video_id).
    """
    vid_id   = record.get("video_hashcode") or record.get("video_name", "unknown")
    vid_name = record.get("video_name", vid_id)

    # Skip if already processed
    out_path = SGRAPH_DIR / f"{vid_id}.json"
    if out_path.exists():
        return "skip", vid_id

    # Find the pre-extracted frames directory
    category = str(record.get("type", ""))
    frame_dir = find_frame_dir(vid_name, category)
    if frame_dir is None:
        return "miss", vid_id

    # Load N evenly-spaced frames from the images/ directory
    frames = load_frames_evenly(frame_dir, n=N_FRAMES)
    if not frames:
        return "fail", vid_id

    # Generate scene graph for each frame
    scene_graphs = []
    for frame_idx, timestamp_s, img in frames:
        sg = sgg.generate(img, frame_idx=frame_idx, timestamp_s=timestamp_s)
        scene_graphs.append(sg)

    # Build output document (Hyun's format)
    output = {
        "video_id"    : vid_id,
        "category"    : int(record.get("type", 0)),
        "metadata"    : {
            "weather": record.get("weather"),
            "light"  : record.get("light"),
            "scenes" : record.get("scenes"),
            "t_ai"   : record.get("t_ai"),
            "t_ae"   : record.get("t_ae"),
        },
        "scene_graphs": scene_graphs,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    return "done", vid_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",        type=int,  default=None,
                        help="Process only the first N videos (for testing)")
    parser.add_argument("--spatial-only", action="store_true",
                        help="Skip LLaVA; use spatial relations only (fast mode)")
    args = parser.parse_args()

    # Load metadata
    print("Loading metadata ...")
    with open(META_FILE) as f:
        raw = json.load(f)
    records = raw if isinstance(raw, list) else list(raw.values())

    # Filter to downloaded categories
    records = [r for r in records
               if str(r.get("type", "")) in DOWNLOADED_CATEGORIES]

    if args.limit:
        records = records[:args.limit]

    print(f"Videos to process : {len(records)}")
    print(f"Frame interval    : {FRAME_INTERVAL_S}s (one scene graph per {FRAME_INTERVAL_S}s)\n")

    # Load models once
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    sgg = ZeroShotSGG(device=device, spatial_only=args.spatial_only)

    # Process sequentially (LLaVA is not thread-safe)
    done = fail = skip = miss = 0
    for record in tqdm(records, desc="Generating scene graphs"):
        status, vid_id = process_one(record, sgg)
        if   status == "done": done += 1
        elif status == "skip": skip += 1
        elif status == "miss": miss += 1
        else:                  fail += 1

    print(f"\nDone: {done} | Skipped: {skip} | Not found: {miss} | Failed: {fail}")
    print(f"Scene graphs saved to: {SGRAPH_DIR}")
    print("\nNext: python mmau_adapter/analyze_scene_graphs.py")


if __name__ == "__main__":
    main()
