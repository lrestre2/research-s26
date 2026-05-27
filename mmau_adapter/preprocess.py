"""
MM-AU Preprocessor
==================
For each downloaded video this script does three things in sequence:

  1. FRAME EXTRACTION
     Extracts 16 frames uniformly sampled from the accident window
     (from t_ai — accident initiation — to t_ae — accident end).
     Why 16? SHG-VQA uses a 16-frame clip as its input unit.
     Output: ~/data/mmau/processed/frames/{video_id}/frame_{n:04d}.jpg

  2. OBJECT DETECTION (YOLOv5 via ultralytics)
     Runs YOLOv5s on each of the 16 frames and records every detected
     object with its bounding box, class name, and confidence score.
     Output: ~/data/mmau/processed/detections/{video_id}.json

  3. SCENE GRAPH CONSTRUCTION
     For each frame, takes the detected objects and computes spatial
     relations between every pair:
       - left_of / right_of   (horizontal position)
       - above / below         (vertical position)
       - near / far            (centre-to-centre distance, threshold=0.3×width)
     These relations become the edges in the per-frame scene sub-graph,
     which the hypergraph builder later stitches together across frames.
     Output: ~/data/mmau/processed/scene_graphs/{video_id}.json

Usage:
    conda activate srp26
    cd ~/Liu/research-s26
    python mmau_adapter/preprocess.py [--limit N] [--workers N]
"""

import json
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ── paths ──────────────────────────────────────────────────────────────
DATA_DIR      = Path.home() / "data" / "mmau"
META_FILE     = DATA_DIR / "video_metadata.json"
FRAMES_DIR    = DATA_DIR / "processed" / "frames"
DETECT_DIR    = DATA_DIR / "processed" / "detections"
SGRAPH_DIR    = DATA_DIR / "processed" / "scene_graphs"
N_FRAMES      = 16          # frames per clip (matches SHG-VQA)
NEAR_THRESH   = 0.3         # fraction of frame width for "near" relation

# ── helpers ─────────────────────────────────────────────────────────────

def find_video_file(video_name: str) -> Path | None:
    """Search for a video file anywhere under DATA_DIR."""
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        for candidate in DATA_DIR.rglob(f"{video_name}{ext}"):
            return candidate
    return None


def extract_frames(video_path: Path, out_dir: Path,
                   t_start: float, t_end: float, n: int = N_FRAMES) -> list[Path]:
    """
    Extract n frames uniformly from [t_start, t_end] in the video.
    Returns list of saved frame paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = max(t_end - t_start, 1.0)
    timestamps = [t_start + i * duration / (n - 1) for i in range(n)]

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    saved = []

    for idx, ts in enumerate(timestamps):
        frame_num = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if ok:
            path = out_dir / f"frame_{idx:04d}.jpg"
            cv2.imwrite(str(path), frame)
            saved.append(path)

    cap.release()
    return saved


def detect_objects(model: YOLO, frame_paths: list[Path]) -> list[dict]:
    """
    Run YOLO on each frame. Returns list of per-frame detection dicts.
    Each detection: {frame_idx, objects: [{id, class, bbox:[x,y,w,h], conf}]}
    """
    results_all = []
    for idx, fp in enumerate(frame_paths):
        results = model(str(fp), verbose=False)[0]
        objects = []
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            objects.append({
                "id":    i,
                "class": results.names[int(box.cls)],
                "bbox":  [x1, y1, x2 - x1, y2 - y1],   # [x, y, w, h]
                "conf":  round(float(box.conf), 3),
            })
        results_all.append({"frame_idx": idx, "objects": objects})
    return results_all


def compute_scene_graph(frame_detections: list[dict], frame_width: int = 1280) -> list[dict]:
    """
    For each frame's detections, compute pairwise spatial relations.
    Returns the same structure with a 'relations' list added to each frame.

    Relations computed:
      left_of / right_of  — based on centre x
      above / below        — based on centre y
      near / far           — Euclidean distance between centres
    """
    near_px = NEAR_THRESH * frame_width
    scene_graph = []

    for frame in frame_detections:
        objs = frame["objects"]
        relations = []

        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                a, b = objs[i], objs[j]
                # centres
                ax = a["bbox"][0] + a["bbox"][2] / 2
                ay = a["bbox"][1] + a["bbox"][3] / 2
                bx = b["bbox"][0] + b["bbox"][2] / 2
                by = b["bbox"][1] + b["bbox"][3] / 2

                dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                proximity = "near" if dist < near_px else "far"

                relations.append({
                    "subject":   a["id"],
                    "predicate": "left_of" if ax < bx else "right_of",
                    "object":    b["id"],
                })
                relations.append({
                    "subject":   a["id"],
                    "predicate": "above" if ay < by else "below",
                    "object":    b["id"],
                })
                relations.append({
                    "subject":   a["id"],
                    "predicate": proximity,
                    "object":    b["id"],
                    "dist_px":   round(dist, 1),
                })

        scene_graph.append({
            "frame_idx": frame["frame_idx"],
            "objects":   objs,
            "relations": relations,
        })

    return scene_graph


def process_one(record: dict, model: YOLO) -> str:
    """Process a single video record end-to-end. Returns status string."""
    vid_id   = record.get("video_hashcode") or record.get("video_name", "unknown")
    vid_name = record.get("video_name", vid_id)

    # Skip if already done
    sg_path = SGRAPH_DIR / f"{vid_id}.json"
    if sg_path.exists():
        return f"skip  {vid_id}"

    # Find video file
    video_path = find_video_file(vid_name)
    if video_path is None:
        return f"miss  {vid_id} (video file not found)"

    # Timestamps — fall back to full video if missing
    t_ai = float(record.get("t_ai") or 0)
    t_ae = float(record.get("t_ae") or t_ai + 10)
    if t_ae <= t_ai:
        t_ae = t_ai + 10

    # 1. Extract frames
    frame_dir   = FRAMES_DIR / vid_id
    frame_paths = extract_frames(video_path, frame_dir, t_ai, t_ae)
    if not frame_paths:
        return f"fail  {vid_id} (no frames extracted)"

    # 2. Detect objects
    detections = detect_objects(model, frame_paths)
    det_path   = DETECT_DIR / f"{vid_id}.json"
    det_path.parent.mkdir(parents=True, exist_ok=True)
    det_path.write_text(json.dumps(detections, indent=2))

    # 3. Build scene graph
    scene_graph = compute_scene_graph(detections)
    sg_path.parent.mkdir(parents=True, exist_ok=True)
    sg_path.write_text(json.dumps(scene_graph, indent=2))

    return f"done  {vid_id}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",   type=int, default=None,
                        help="Process only the first N videos (useful for testing)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel threads (default 4)")
    args = parser.parse_args()

    # Load metadata
    print("Loading metadata...")
    with open(META_FILE) as f:
        raw = json.load(f)
    records = raw if isinstance(raw, list) else list(raw.values())

    # Filter to only downloaded categories (11 and 43)
    downloaded_categories = {"11", "43"}
    records = [r for r in records if str(r.get("type", "")) in downloaded_categories]

    if args.limit:
        records = records[:args.limit]

    print(f"Videos to process : {len(records)}")
    print(f"Parallel workers  : {args.workers}\n")

    # Load YOLO model once — shared across threads
    print("Loading YOLOv5s model...")
    model = YOLO("yolov5su.pt")   # auto-downloads on first run
    print("Model ready.\n")

    # Process
    ok = fail = skip = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, r, model): r for r in records}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            status = fut.result()
            if status.startswith("done"):  ok   += 1
            elif status.startswith("skip"): skip += 1
            else:                           fail += 1

    print(f"\nDone: {ok} | Skipped: {skip} | Failed: {fail}")
    print("Next: python mmau_adapter/run_baseline.py")

if __name__ == "__main__":
    main()
