"""
MM-AU Dataset class — SHG-VQA compatible
==========================================
Reads MM-AU metadata + preprocessed scene graphs and returns
(frames_tensor, scene_graph_dict, question, answer_label) tuples
that the SHG-VQA model can consume directly.

This is the "bridge" file — it speaks MM-AU on one side and SHG-VQA
on the other. If you change the preprocessing format, change it here.

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
DATA_DIR      = Path.home() / "data" / "mmau"
META_FILE     = DATA_DIR / "video_metadata.json"
FRAMES_DIR    = DATA_DIR / "processed" / "frames"
SGRAPH_DIR    = DATA_DIR / "processed" / "scene_graphs"

# ── constants ─────────────────────────────────────────────────────────
N_FRAMES   = 16
IMG_SIZE   = 224      # SHG-VQA default input resolution
CATEGORIES = list(range(1, 59))   # accident categories 1–58

# MM-AU object classes (7 annotated classes)
OBJECT_CLASSES = ["car", "traffic light", "pedestrian", "truck",
                  "bus", "cyclist", "motorbike"]
CLASS_TO_IDX = {c: i for i, c in enumerate(OBJECT_CLASSES)}

# Frame transform: resize and normalise to ImageNet stats
FRAME_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class MMAUDataset(Dataset):
    """
    Each sample corresponds to one accident video.

    Returns a dict with:
        frames      : Tensor [N_FRAMES, 3, H, W]  — the 16 keyframes
        scene_graph : dict   — per-frame objects + relations
        question    : str    — "Why did the accident happen?"
        answer      : str    — the accident cause text
        category    : int    — accident type (1–58)
        video_id    : str    — unique video identifier
    """

    def __init__(
        self,
        split       : Literal["train", "val", "test"] = "train",
        categories  : list[int] | None = None,   # None = all downloaded
        max_samples : int | None = None,
    ):
        self.split = split

        # Load metadata
        with open(META_FILE) as f:
            raw = json.load(f)
        all_records = raw if isinstance(raw, list) else list(raw.values())

        # Keep only records that have been preprocessed
        all_records = [
            r for r in all_records
            if (SGRAPH_DIR / f"{self._vid_id(r)}.json").exists()
        ]

        # Filter to requested categories
        if categories is not None:
            all_records = [r for r in all_records
                           if int(r.get("type", 0)) in categories]

        # Train / val / test split — 7 : 1.5 : 1.5 ratio by index
        n = len(all_records)
        t_end = int(n * 0.70)
        v_end = int(n * 0.85)
        if split == "train":
            self.records = all_records[:t_end]
        elif split == "val":
            self.records = all_records[t_end:v_end]
        else:
            self.records = all_records[v_end:]

        if max_samples:
            self.records = self.records[:max_samples]

    @staticmethod
    def _vid_id(record: dict) -> str:
        return record.get("video_hashcode") or record.get("video_name", "unknown")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record  = self.records[idx]
        vid_id  = self._vid_id(record)

        # ── frames ──────────────────────────────────────────────────────
        frame_dir = FRAMES_DIR / vid_id
        frame_tensors = []
        for fi in range(N_FRAMES):
            fp = frame_dir / f"frame_{fi:04d}.jpg"
            if fp.exists():
                img = Image.open(fp).convert("RGB")
                frame_tensors.append(FRAME_TRANSFORM(img))
            else:
                frame_tensors.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))
        frames = torch.stack(frame_tensors)           # [16, 3, 224, 224]

        # ── scene graph ─────────────────────────────────────────────────
        sg_path = SGRAPH_DIR / f"{vid_id}.json"
        scene_graph = json.loads(sg_path.read_text()) if sg_path.exists() else []

        # ── question & answer ────────────────────────────────────────────
        # MM-AU stores the accident reason in the "causes" field.
        # We frame this as a fixed question with the cause as the answer.
        question = "Why did the accident happen?"
        answer   = record.get("causes") or record.get("texts") or "unknown"

        # ── category ────────────────────────────────────────────────────
        category = int(record.get("type", 0))

        return {
            "frames"      : frames,
            "scene_graph" : scene_graph,
            "question"    : question,
            "answer"      : answer,
            "category"    : category,
            "video_id"    : vid_id,
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate: stack tensors, keep text/graph as lists.
    Pass this to DataLoader as collate_fn=collate_fn.
    """
    return {
        "frames"      : torch.stack([b["frames"] for b in batch]),
        "scene_graph" : [b["scene_graph"] for b in batch],
        "question"    : [b["question"] for b in batch],
        "answer"      : [b["answer"] for b in batch],
        "category"    : torch.tensor([b["category"] for b in batch]),
        "video_id"    : [b["video_id"] for b in batch],
    }


if __name__ == "__main__":
    ds = MMAUDataset(split="train", max_samples=5)
    print(f"Train samples: {len(ds)}")
    if len(ds):
        s = ds[0]
        print(f"  frames shape : {s['frames'].shape}")
        print(f"  category     : {s['category']}")
        print(f"  question     : {s['question']}")
        print(f"  answer       : {s['answer'][:80]}...")
        print(f"  scene graph  : {len(s['scene_graph'])} frames")
