"""
Quick demo — generates a scene graph for the first video found on disk.
Bypasses metadata lookup entirely. Run this for the meeting.

Usage:
    PYTHONPATH=. python mmau_adapter/quick_demo.py
"""
import json
from pathlib import Path
from PIL import Image
from mmau_adapter.scene_graph_gen import ZeroShotSGG

DATA_DIR  = Path.home() / "data" / "mmau"
CAP_DIR   = DATA_DIR / "CAP-DATA" / "CAP-DATA"
OUT_DIR   = DATA_DIR / "processed" / "scene_graphs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Find the first images/ directory on disk
print("Finding first video...")
img_dirs = list(CAP_DIR.rglob("images"))
if not img_dirs:
    print("No images/ directories found under", CAP_DIR)
    exit(1)

img_dir  = img_dirs[0]
video_id = img_dir.parent.name
print(f"Using video: {video_id}  ({img_dir})\n")

# Load 5 evenly-spaced frames
jpgs = sorted(img_dir.glob("*.jpg"))
total = len(jpgs)
indices = [int(i * (total - 1) / 4) for i in range(5)]
frames = [(i, int(jpgs[idx].stem) / 10.0, Image.open(jpgs[idx]).convert("RGB"))
          for i, idx in enumerate(indices)]
print(f"Loaded {len(frames)} frames from {total} available\n")

# Generate scene graphs
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
sgg = ZeroShotSGG(device=device)

scene_graphs = []
for frame_idx, timestamp_s, img in frames:
    print(f"Processing frame {frame_idx+1}/5 (t={timestamp_s:.1f}s)...")
    sg = sgg.generate(img, frame_idx=frame_idx, timestamp_s=timestamp_s)
    scene_graphs.append(sg)
    print(f"  Objects   : {[o['label'] for o in sg['objects']]}")
    print(f"  Relations : {[r['predicate'] for r in sg['relations'] if r['source']=='llava']}\n")

# Save
output = {"video_id": video_id, "category": int(img_dir.parts[-3]),
          "scene_graphs": scene_graphs}
out_path = OUT_DIR / f"{video_id}_demo.json"
out_path.write_text(json.dumps(output, indent=2))
print(f"Saved → {out_path}")
