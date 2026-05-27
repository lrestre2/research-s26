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

import cv2
from tqdm import tqdm
from PIL import Image

from mmau_adapter.scene_graph_gen import ZeroShotSGG

# ── paths ──────────────────────────────────────────────────────────────
DATA_DIR   = Path.home() / "data" / "mmau"
META_FILE  = DATA_DIR / "video_metadata.json"
SGRAPH_DIR = DATA_DIR / "processed" / "scene_graphs"
FRAME_INTERVAL_S = 3.0   # one scene graph every 3 seconds (Hyun's spec)

# Categories we've downloaded
DOWNLOADED_CATEGORIES = {"11", "43"}


def find_video_file(video_name: str) -> Path | None:
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        for f in DATA_DIR.rglob(f"{video_name}{ext}"):
            return f
    return None


def extract_frames_at_interval(
    video_path: Path,
    t_start: float,
    t_end: float,
    interval_s: float = FRAME_INTERVAL_S,
) -> list[tuple[int, float, Image.Image]]:
    """
    Extract one frame every `interval_s` seconds between t_start and t_end.
    Returns list of (frame_idx, timestamp_s, PIL.Image).
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = max(t_end - t_start, interval_s)

    # Build timestamp list: t_start, t_start+3, t_start+6, ... ≤ t_end
    timestamps = []
    t = t_start
    while t <= t_end + 0.1:
        timestamps.append(t)
        t += interval_s

    frames = []
    for idx, ts in enumerate(timestamps):
        frame_num = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        frames.append((idx, round(ts - t_start, 2), img))

    cap.release()
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

    # Find video file on disk
    video_path = find_video_file(vid_name)
    if video_path is None:
        return "miss", vid_id

    # Accident window timestamps
    t_ai = float(record.get("t_ai") or 0)
    t_ae = float(record.get("t_ae") or t_ai + 10)
    if t_ae <= t_ai:
        t_ae = t_ai + 10

    # Extract frames at 1 per FRAME_INTERVAL_S
    frames = extract_frames_at_interval(video_path, t_ai, t_ae)
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
            "t_ai"   : t_ai,
            "t_ae"   : t_ae,
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
