"""
MM-AU Dataset class — driven by scene graph JSONs
==================================================
Loads directly from the preprocessed scene graph JSONs in
~/data/mmau/processed/scene_graphs/ and the original JPEG frames
in ~/data/mmau/CAP-DATA/CAP-DATA/.

Does NOT depend on video_metadata.json matching folder names —
that mapping was broken. Everything comes from the JSONs we generated.

Classification task
-------------------
With only category 11 on disk we can't do category classification
(only one class → model learns nothing). Instead we use a binary
label derived from the scene graphs themselves:

  label 1 — "complex scene": the video has >= 3 unique object
             classes detected across its frames (e.g. car + cyclist
             + pedestrian). These tend to be more dangerous scenarios.
  label 0 — "simple scene":  fewer than 3 unique object classes.

This gives a real, balanced, meaningful training signal without
needing external metadata. Once category 43 is preprocessed, swap
`_make_label` to return the accident category instead.

Usage:
    from mmau_adapter.dataset import MMAUDataset
    ds = MMAUDataset(split="train")
    sample = ds[0]
"""

import json
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# ── paths ──────────────────────────────────────────────────────────────
DATA_DIR   = Path.home() / "data" / "mmau"
CAP_DIR    = DATA_DIR / "CAP-DATA" / "CAP-DATA"
SGRAPH_DIR = DATA_DIR / "processed" / "scene_graphs"

# ── constants ─────────────────────────────────────────────────────────
N_FRAMES  = 5        # matches what preprocess.py sampled
IMG_SIZE  = 224      # standard input size for vision models

FRAME_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── label logic ───────────────────────────────────────────────────────

def _make_label(data: dict) -> tuple[str, int]:
    """
    Derive a binary label from the scene graph content.

    Returns (answer_str, int_label) where:
      "complex_scene" / 1 → >= 3 unique object classes detected
      "simple_scene"  / 0 → < 3 unique object classes detected

    Swap this function out once category 43 is on disk and you
    want to do category classification instead.
    """
    classes = set()
    for sg in data.get("scene_graphs", []):
        for obj in sg.get("objects", []):
            label = obj.get("label") or obj.get("class", "")
            if label:
                classes.add(label.lower())
    is_complex = len(classes) >= 3
    return ("complex_scene" if is_complex else "simple_scene",
            1 if is_complex else 0)


# ── dataset ───────────────────────────────────────────────────────────

class MMAUDataset(Dataset):
    """
    Each sample is one accident video.

    Returns a dict with:
        frames      : Tensor [N_FRAMES, 3, 224, 224]
        scene_graph : list[dict]  — per-frame objects + relations
        question    : str         — fixed question
        answer      : str         — "complex_scene" or "simple_scene"
        label       : int         — 0 or 1
        category    : int         — accident category from JSON
        video_id    : str
    """

    def __init__(
        self,
        split       : Literal["train", "val", "test"] = "train",
        categories  : list[int] | None = None,
        max_samples : int | None = None,
    ):
        self.split = split

        # ── Load all scene graph JSONs ───────────────────────────────
        all_data = []
        for jp in sorted(SGRAPH_DIR.glob("*.json")):
            try:
                d = json.loads(jp.read_text())
                if isinstance(d, dict) and "scene_graphs" in d:
                    all_data.append(d)
            except Exception:
                continue

        # ── Filter by category if requested ─────────────────────────
        if categories is not None:
            all_data = [d for d in all_data
                        if d.get("category") in categories]

        # ── Train / val / test split (70 / 15 / 15) ─────────────────
        n     = len(all_data)
        t_end = int(n * 0.70)
        v_end = int(n * 0.85)
        if split == "train":
            self.data = all_data[:t_end]
        elif split == "val":
            self.data = all_data[t_end:v_end]
        else:
            self.data = all_data[v_end:]

        if max_samples:
            self.data = self.data[:max_samples]

        # ── Build frame directory cache (scan CAP_DIR once) ─────────
        # Maps video_id (folder name) → Path of images/ directory
        self._frame_cache: dict[str, Path] = {}
        for p in CAP_DIR.rglob("images"):
            if p.is_dir():
                self._frame_cache[p.parent.name] = p

        print(f"  [{split}] {len(self.data)} samples | "
              f"{len(self._frame_cache)} frame dirs indexed")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        d        = self.data[idx]
        video_id = d.get("video_id", "unknown")

        # ── Load frames from original JPEG directory ─────────────────
        frame_dir     = self._frame_cache.get(video_id)
        frame_tensors = []

        if frame_dir and frame_dir.exists():
            jpgs  = sorted(frame_dir.glob("*.jpg"))
            total = len(jpgs)
            if total >= N_FRAMES:
                idxs = [int(i * (total - 1) / (N_FRAMES - 1))
                        for i in range(N_FRAMES)]
            else:
                idxs = list(range(total))

            for i in range(N_FRAMES):
                if i < len(idxs):
                    try:
                        img = Image.open(jpgs[idxs[i]]).convert("RGB")
                        frame_tensors.append(FRAME_TRANSFORM(img))
                    except Exception:
                        frame_tensors.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
                else:
                    frame_tensors.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
        else:
            frame_tensors = [torch.zeros(3, IMG_SIZE, IMG_SIZE)] * N_FRAMES

        frames = torch.stack(frame_tensors)   # [N_FRAMES, 3, 224, 224]

        # ── Label ────────────────────────────────────────────────────
        answer, label = _make_label(d)

        return {
            "frames"      : frames,
            "scene_graph" : d.get("scene_graphs", []),
            "question"    : "How complex is this accident scene?",
            "answer"      : answer,
            "label"       : label,
            "category"    : d.get("category", 0),
            "video_id"    : video_id,
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "frames"      : torch.stack([b["frames"] for b in batch]),
        "scene_graph" : [b["scene_graph"] for b in batch],
        "question"    : [b["question"] for b in batch],
        "answer"      : [b["answer"] for b in batch],
        "label"       : torch.tensor([b["label"] for b in batch]),
        "category"    : torch.tensor([b["category"] for b in batch]),
        "video_id"    : [b["video_id"] for b in batch],
    }


if __name__ == "__main__":
    print("Loading dataset...")
    ds = MMAUDataset(split="train", max_samples=10)
    print(f"Samples : {len(ds)}")
    if len(ds):
        s = ds[0]
        print(f"  video_id     : {s['video_id']}")
        print(f"  frames shape : {s['frames'].shape}")
        print(f"  scene graphs : {len(s['scene_graph'])} frames")
        print(f"  answer/label : {s['answer']} ({s['label']})")
        n_real = (s['frames'].sum(dim=(1,2,3)) != 0).sum().item()
        print(f"  real frames  : {n_real}/{len(s['frames'])} (others = black)")
