"""
Zero-Shot Scene Graph Generation for MM-AU
===========================================
Uses two open-vocabulary models in sequence — no fine-tuning on traffic data:

  1. Grounding DINO  — detects any object described in natural language.
                       Gives accurate bounding boxes + semantic labels.

  2. LLaVA-1.5-7B   — given the frame image + detected object list,
                       generates semantic predicates between object pairs
                       (e.g. "approaching", "colliding with", "yielding to").
                       This is the zero-shot part: the model reasons about
                       relations using its general visual knowledge.

Output per frame (matches the JSON schema Hyun specified):
  {
    "frame_idx"  : 0,
    "timestamp_s": 0.0,
    "objects": [
      {"id": 0, "label": "car",     "bbox": [x,y,w,h], "conf": 0.93,
       "attributes": ["moving", "large"]},
      {"id": 1, "label": "cyclist", "bbox": [x,y,w,h], "conf": 0.87,
       "attributes": ["vulnerable"]}
    ],
    "relations": [
      {"subject": 0, "predicate": "approaching",   "object": 1, "source": "llava"},
      {"subject": 0, "predicate": "left_of",       "object": 1, "source": "spatial"},
      {"subject": 0, "predicate": "near",          "object": 1, "source": "spatial"}
    ]
  }

The "source" field distinguishes LLaVA semantic relations from geometric ones.

Reference: LLaVA-SpaceSGG (Awesome-SGG list) and Grounding DINO (IDEA Research).
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image

# ── constants ───────────────────────────────────────────────────────────
GDINO_MODEL  = "IDEA-Research/grounding-dino-base"
LLAVA_MODEL  = "llava-hf/llava-1.5-7b-hf"

# Text prompt for Grounding DINO — open vocabulary for traffic scenes
GDINO_PROMPT = (
    "car . truck . bus . motorcycle . bicycle . cyclist . pedestrian . person . "
    "traffic light . stop sign . yield sign . road marking . lane marking . "
    "barrier . guardrail . pole . tree"
)

NEAR_THRESH  = 0.30   # fraction of frame width
SPATIAL_ONLY = False  # set True to skip LLaVA (fast mode, less rich)


class ZeroShotSGG:
    """
    Loads both models once and generates scene graphs for individual frames.
    Keep one instance alive for the whole preprocessing run.
    """

    def __init__(self, device: str = "cuda", spatial_only: bool = SPATIAL_ONLY):
        self.device       = device
        self.spatial_only = spatial_only
        self._gdino_proc  = None
        self._gdino_model = None
        self._llava_proc  = None
        self._llava_model = None
        self._load_models()

    # ── model loading ────────────────────────────────────────────────────

    def _load_models(self):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        print("  Loading Grounding DINO ...")
        self._gdino_proc  = AutoProcessor.from_pretrained(GDINO_MODEL)
        self._gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GDINO_MODEL
        ).to(self.device)
        self._gdino_model.eval()

        if not self.spatial_only:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            print("  Loading LLaVA-1.5-7B ...")
            self._llava_proc  = LlavaNextProcessor.from_pretrained(LLAVA_MODEL)
            self._llava_model = LlavaNextForConditionalGeneration.from_pretrained(
                LLAVA_MODEL,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self._llava_model.eval()

        print("  Models ready.\n")

    # ── object detection ─────────────────────────────────────────────────

    def _detect_objects(self, image: Image.Image, threshold: float = 0.35) -> list[dict]:
        """Run Grounding DINO and return a list of detected objects."""
        inputs = self._gdino_proc(
            images=image,
            text=GDINO_PROMPT,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._gdino_model(**inputs)

        results = self._gdino_proc.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=threshold,
            text_threshold=threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        w, h = image.size
        objects = []
        for i, (box, score, label) in enumerate(
            zip(results["boxes"], results["scores"], results["labels"])
        ):
            x1, y1, x2, y2 = box.tolist()
            objects.append({
                "id"        : i,
                "label"     : label,
                "bbox"      : [round(x1, 1), round(y1, 1),
                               round(x2 - x1, 1), round(y2 - y1, 1)],
                "conf"      : round(float(score), 3),
                "attributes": [],
            })
        return objects

    # ── spatial relations (geometric, always computed) ───────────────────

    def _spatial_relations(
        self, objects: list[dict], frame_width: int
    ) -> list[dict]:
        """Geometric spatial relations between every object pair."""
        near_px = NEAR_THRESH * frame_width
        relations = []

        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                a, b = objects[i], objects[j]
                ax = a["bbox"][0] + a["bbox"][2] / 2
                ay = a["bbox"][1] + a["bbox"][3] / 2
                bx = b["bbox"][0] + b["bbox"][2] / 2
                by = b["bbox"][1] + b["bbox"][3] / 2
                dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

                relations.append({"subject": a["id"],
                                  "predicate": "left_of" if ax < bx else "right_of",
                                  "object": b["id"], "source": "spatial"})
                relations.append({"subject": a["id"],
                                  "predicate": "above" if ay < by else "below",
                                  "object": b["id"], "source": "spatial"})
                relations.append({"subject": a["id"],
                                  "predicate": "near" if dist < near_px else "far",
                                  "object": b["id"], "source": "spatial"})
        return relations

    # ── semantic relations (LLaVA, zero-shot) ────────────────────────────

    def _semantic_relations(
        self, image: Image.Image, objects: list[dict]
    ) -> list[dict]:
        """Ask LLaVA to describe semantic relations between detected objects."""
        if not objects or len(objects) < 2:
            return []

        obj_list = "\n".join(
            f"  Object {o['id']}: {o['label']} (confidence {o['conf']:.2f})"
            for o in objects
        )

        prompt = (
            "[INST] <image>\n"
            "You are analyzing a traffic dashcam frame from an accident video.\n"
            "The following objects were detected:\n"
            f"{obj_list}\n\n"
            "List the spatial and causal relationships between these objects. "
            "Focus on relationships relevant to traffic safety: approaching, "
            "colliding_with, cutting_off, yielding_to, following, overtaking, "
            "crossing_path_of, blocking, stopped_in_front_of.\n\n"
            "Respond ONLY with a JSON list, no other text. Example:\n"
            '[{"subject": 0, "predicate": "approaching", "object": 1}]\n'
            "[/INST]"
        )

        inputs = self._llava_proc(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self._llava_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        response = self._llava_proc.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        # Parse JSON from response
        try:
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                rels = json.loads(json_match.group())
                for r in rels:
                    r["source"] = "llava"
                return rels
        except (json.JSONDecodeError, KeyError):
            pass
        return []

    # ── public API ───────────────────────────────────────────────────────

    def generate(
        self,
        image: Image.Image,
        frame_idx: int,
        timestamp_s: float,
    ) -> dict:
        """
        Generate a scene graph for one frame.
        Returns a dict matching the schema Hyun specified.
        """
        w, h = image.size

        # 1. Detect objects (Grounding DINO)
        objects = self._detect_objects(image)

        # 2. Spatial relations (always)
        relations = self._spatial_relations(objects, frame_width=w)

        # 3. Semantic relations (LLaVA, zero-shot)
        if not self.spatial_only and objects:
            semantic = self._semantic_relations(image, objects)
            relations.extend(semantic)

        return {
            "frame_idx"  : frame_idx,
            "timestamp_s": round(timestamp_s, 2),
            "objects"    : objects,
            "relations"  : relations,
        }
