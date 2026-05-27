"""
Hypergraph Schema for MM-AU
============================
This file defines the data structures and the construction function
for our situation hypergraph. It is also the authoritative reference
for the schema — read this alongside HYPERGRAPH_SCHEMA.md.

CONCEPTS
--------
A standard graph connects pairs of nodes: edge(A, B).
A hypergraph can connect ANY NUMBER of nodes: hyperedge(A, B, C, D, ...).
This is why hypergraphs suit accident scenes: a "collision" event involves
several road users simultaneously, not just two.

SCHEMA
------

Node
  Represents one detected object in one specific frame.
  Properties:
    - node_id       : unique string "{video_id}_f{frame}_obj{obj_id}"
    - frame_idx     : which of the 16 frames (0–15)
    - obj_id        : YOLO detection id within that frame
    - obj_class     : "car", "pedestrian", "cyclist", etc.
    - bbox          : [x, y, w, h] in pixels
    - conf          : YOLO confidence score
    - features      : visual embedding (filled later by SlowFast, None for now)

Hyperedge types
  1. SPATIAL  — objects physically close to each other within one frame.
                Encodes the scene layout at a moment in time.
                Threshold: centre distance < 0.3 × frame width.

  2. TEMPORAL — the same tracked object across consecutive frames.
                Encodes motion and trajectory through time.
                Matching: same class + IoU of bboxes > 0.4.

  3. COLLISION — objects that interact at the accident moment.
                 Defined as: objects of different classes whose bboxes
                 overlap (IoU > 0.1) during frames inside [t_ai, t_ae].
                 This is the causal edge — who hit whom.

  4. MULTIVIEW (future) — same physical object seen from two cameras.
                          Currently a placeholder; activated when
                          infrastructure-view data is added.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import json
from pathlib import Path

# ── data structures ──────────────────────────────────────────────────────

@dataclass
class Node:
    node_id   : str
    frame_idx : int
    obj_id    : int
    obj_class : str
    bbox      : list[float]      # [x, y, w, h]
    conf      : float
    features  : list[float] | None = None   # visual embedding (SlowFast)

    def centre(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2, y + h / 2

    def area(self) -> float:
        return self.bbox[2] * self.bbox[3]


@dataclass
class Hyperedge:
    edge_id   : str
    edge_type : Literal["spatial", "temporal", "collision", "multiview"]
    node_ids  : list[str]
    metadata  : dict = field(default_factory=dict)


@dataclass
class SituationHypergraph:
    video_id   : str
    nodes      : list[Node]      = field(default_factory=list)
    hyperedges : list[Hyperedge] = field(default_factory=list)

    def add_node(self, node: Node):
        self.nodes.append(node)

    def add_edge(self, edge: Hyperedge):
        self.hyperedges.append(edge)

    def summary(self) -> str:
        type_counts = {}
        for e in self.hyperedges:
            type_counts[e.edge_type] = type_counts.get(e.edge_type, 0) + 1
        return (f"SituationHypergraph({self.video_id}) | "
                f"{len(self.nodes)} nodes | "
                f"{len(self.hyperedges)} hyperedges {type_counts}")

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "nodes": [vars(n) for n in self.nodes],
            "hyperedges": [vars(e) for e in self.hyperedges],
        }


# ── construction helpers ─────────────────────────────────────────────────

def _iou(bbox_a: list[float], bbox_b: list[float]) -> float:
    """Intersection-over-Union of two [x, y, w, h] boxes."""
    ax1, ay1 = bbox_a[0], bbox_a[1]
    ax2, ay2 = ax1 + bbox_a[2], ay1 + bbox_a[3]
    bx1, by1 = bbox_b[0], bbox_b[1]
    bx2, by2 = bx1 + bbox_b[2], by1 + bbox_b[3]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = bbox_a[2] * bbox_a[3]
    area_b = bbox_b[2] * bbox_b[3]
    return inter / (area_a + area_b - inter)


def _centre_dist(a: Node, b: Node) -> float:
    ax, ay = a.centre()
    bx, by = b.centre()
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


# ── main construction function ───────────────────────────────────────────

def build_hypergraph(
    video_id    : str,
    scene_graph : list[dict],          # output of preprocess.py
    frame_width : int   = 1280,
    spatial_thresh  : float = 0.3,     # fraction of frame width
    temporal_iou    : float = 0.4,
    collision_iou   : float = 0.1,
) -> SituationHypergraph:
    """
    Build a SituationHypergraph from a video's scene graph.

    Args:
        video_id    : identifier string for the video
        scene_graph : list of per-frame dicts from preprocess.py
        frame_width : pixel width of the frames (default 1280)
        spatial_thresh  : centre-distance threshold (× frame_width) for SPATIAL edges
        temporal_iou    : min IoU to link the same object across frames
        collision_iou   : min IoU to declare a COLLISION between different-class objects

    Returns:
        SituationHypergraph ready for the HGNN
    """
    hg = SituationHypergraph(video_id=video_id)
    near_px = spatial_thresh * frame_width
    edge_counter = 0

    # ── 1. Create all nodes ──────────────────────────────────────────────
    # node_id format: "{video_id}_f{frame_idx}_o{obj_id}"
    node_map: dict[str, Node] = {}

    for frame in scene_graph:
        fi = frame["frame_idx"]
        for obj in frame["objects"]:
            nid = f"{video_id}_f{fi:02d}_o{obj['id']}"
            node = Node(
                node_id   = nid,
                frame_idx = fi,
                obj_id    = obj["id"],
                obj_class = obj["class"],
                bbox      = obj["bbox"],
                conf      = obj["conf"],
            )
            hg.add_node(node)
            node_map[nid] = node

    # Group nodes by frame for easy lookup
    frames_nodes: dict[int, list[Node]] = {}
    for n in hg.nodes:
        frames_nodes.setdefault(n.frame_idx, []).append(n)

    # ── 2. SPATIAL hyperedges (within each frame) ────────────────────────
    for fi, nodes in frames_nodes.items():
        # Find all clusters of mutually-near nodes (greedy grouping)
        used = set()
        for i, a in enumerate(nodes):
            if a.node_id in used:
                continue
            group = [a.node_id]
            for j, b in enumerate(nodes):
                if i == j:
                    continue
                if _centre_dist(a, b) < near_px:
                    group.append(b.node_id)
                    used.add(b.node_id)
            if len(group) >= 2:
                eid = f"{video_id}_spatial_{edge_counter}"
                edge_counter += 1
                hg.add_edge(Hyperedge(
                    edge_id   = eid,
                    edge_type = "spatial",
                    node_ids  = group,
                    metadata  = {"frame_idx": fi},
                ))

    # ── 3. TEMPORAL hyperedges (same object, consecutive frames) ─────────
    sorted_frames = sorted(frames_nodes.keys())
    for k in range(len(sorted_frames) - 1):
        fi  = sorted_frames[k]
        fi1 = sorted_frames[k + 1]
        for a in frames_nodes.get(fi, []):
            best_iou, best_b = 0.0, None
            for b in frames_nodes.get(fi1, []):
                if a.obj_class != b.obj_class:
                    continue
                iou = _iou(a.bbox, b.bbox)
                if iou > best_iou:
                    best_iou, best_b = iou, b
            if best_b is not None and best_iou >= temporal_iou:
                eid = f"{video_id}_temporal_{edge_counter}"
                edge_counter += 1
                hg.add_edge(Hyperedge(
                    edge_id   = eid,
                    edge_type = "temporal",
                    node_ids  = [a.node_id, best_b.node_id],
                    metadata  = {"iou": round(best_iou, 3),
                                 "frames": [fi, fi1]},
                ))

    # ── 4. COLLISION hyperedges (overlapping different-class objects) ─────
    for fi, nodes in frames_nodes.items():
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i >= j:
                    continue
                if a.obj_class == b.obj_class:
                    continue          # same class overlap is not a collision
                iou = _iou(a.bbox, b.bbox)
                if iou >= collision_iou:
                    eid = f"{video_id}_collision_{edge_counter}"
                    edge_counter += 1
                    hg.add_edge(Hyperedge(
                        edge_id   = eid,
                        edge_type = "collision",
                        node_ids  = [a.node_id, b.node_id],
                        metadata  = {"iou": round(iou, 3),
                                     "frame_idx": fi,
                                     "classes": [a.obj_class, b.obj_class]},
                    ))

    return hg


# ══════════════════════════════════════════════════════════════════════════
# LEARNABLE HYPEREDGE CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════
#
# Instead of hardcoding rules (objects within 30% of width → spatial edge),
# we learn which nodes should be grouped into a hyperedge.
#
# HOW IT WORKS (plain English):
#   1. Every node has a feature vector (visual embedding + class + position).
#   2. A small neural network takes ALL node features and outputs a soft
#      "membership score" for each node in each potential hyperedge.
#   3. During training, the network learns which groupings lead to better
#      accident understanding — the hyperedge structure is discovered from
#      data, not designed by hand.
#   4. At inference, we threshold the scores to get hard hyperedges.
#
# This is called a "differentiable hypergraph" because the incidence matrix
# (the matrix that says which nodes belong to which hyperedge) is soft
# and differentiable during training.
#
# Reference: DHGNN (Dynamic Hypergraph Neural Networks, IJCAI 2019)
# ══════════════════════════════════════════════════════════════════════════

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:

    class LearnableHyperedgeConstructor(nn.Module):
        """
        Learns which nodes should be grouped into hyperedges.

        Input:
            node_features : Tensor [N, feat_dim]
                            Features for each of the N nodes in the graph.
                            Can be visual embeddings, class one-hots,
                            normalised bbox coords — or all three concatenated.

        Output:
            H : Tensor [N, K]
                Soft incidence matrix. H[i, k] is the probability that
                node i belongs to hyperedge k. Values in [0, 1].

        During training: use H directly (soft, differentiable).
        At inference:    threshold H > 0.5 to get hard membership.

        Args:
            feat_dim   : dimension of node feature vectors
            n_edges    : number of learnable hyperedges K
            hidden_dim : width of the MLP
        """

        def __init__(self, feat_dim: int, n_edges: int = 16, hidden_dim: int = 128):
            super().__init__()
            self.n_edges = n_edges

            # MLP: node features → membership scores for each hyperedge
            self.edge_predictor = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_edges),
            )

            # Attention: weight each node's contribution by how "salient" it is
            self.salience = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, node_features: "torch.Tensor") -> "torch.Tensor":
            """
            Args:
                node_features: [N, feat_dim]
            Returns:
                H: [N, K] soft incidence matrix (sigmoid-normalised)
            """
            # Predict raw membership logits
            logits = self.edge_predictor(node_features)   # [N, K]

            # Soft membership via sigmoid (independent per edge)
            H = torch.sigmoid(logits)                     # [N, K] in (0,1)

            # Zero out edges where no node has strong membership
            # (removes empty/spurious hyperedges)
            edge_strength = H.max(dim=0).values           # [K]
            H = H * (edge_strength > 0.3).float()

            return H

        def to_hard(
            self,
            node_features: "torch.Tensor",
            threshold: float = 0.5,
        ) -> list[list[int]]:
            """
            Returns hard hyperedges as lists of node indices.
            Useful at inference time.

            Returns:
                List of K lists, each containing indices of member nodes.
                Empty lists are removed.
            """
            with torch.no_grad():
                H = self.forward(node_features)             # [N, K]
                hard = (H > threshold)                      # [N, K] bool

            hyperedges = []
            for k in range(self.n_edges):
                members = hard[:, k].nonzero(as_tuple=True)[0].tolist()
                if len(members) >= 2:
                    hyperedges.append(members)
            return hyperedges


    def node_features_from_sg(scene_graphs: list[dict], feat_dim: int = 64) -> "torch.Tensor":
        """
        Build a simple node feature matrix from a scene graph list.
        Used to feed into LearnableHyperedgeConstructor.

        Features per node (concatenated):
          - class one-hot (top 10 classes, dim=10)
          - normalised bbox centre (x/W, y/H, dim=2)
          - normalised bbox size (w/W, h/H, dim=2)
          - frame index normalised (t/T, dim=1)
          Total: 15 dims. Padded to feat_dim with zeros if needed.

        Returns:
            Tensor [N, feat_dim]
        """
        CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle",
                   "cyclist", "pedestrian", "person", "traffic light", "other"]
        cls_idx = {c: i for i, c in enumerate(CLASSES)}

        rows = []
        T = max(len(scene_graphs), 1)

        for sg in scene_graphs:
            frame_t = sg.get("frame_idx", 0) / T
            for obj in sg.get("objects", []):
                vec = [0.0] * feat_dim

                # class one-hot
                c = cls_idx.get(obj.get("label", "other"), len(CLASSES) - 1)
                if c < 10:
                    vec[c] = 1.0

                # normalised bbox (assume 1280×720 if not known)
                bbox = obj.get("bbox", [0, 0, 0, 0])
                vec[10] = (bbox[0] + bbox[2] / 2) / 1280   # cx
                vec[11] = (bbox[1] + bbox[3] / 2) / 720    # cy
                vec[12] = bbox[2] / 1280                    # w
                vec[13] = bbox[3] / 720                     # h
                vec[14] = frame_t                           # time

                rows.append(vec)

        if not rows:
            return torch.zeros(1, feat_dim)
        return torch.tensor(rows, dtype=torch.float32)


# ── quick test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    DATA_DIR   = Path.home() / "data" / "mmau"
    SGRAPH_DIR = DATA_DIR / "processed" / "scene_graphs"

    sg_files = list(SGRAPH_DIR.glob("*.json"))
    if not sg_files:
        print("No scene graphs found. Run preprocess.py first.")
        sys.exit(1)

    # Test on first available video
    sg_path = sg_files[0]
    video_id = sg_path.stem
    scene_graph = json.loads(sg_path.read_text())

    hg = build_hypergraph(video_id, scene_graph)
    print(hg.summary())

    # Show edge type breakdown
    from collections import Counter
    counts = Counter(e.edge_type for e in hg.hyperedges)
    for etype, n in counts.items():
        print(f"  {etype:<12}: {n} hyperedges")

    # Show one collision edge if any
    collisions = [e for e in hg.hyperedges if e.edge_type == "collision"]
    if collisions:
        e = collisions[0]
        print(f"\nExample collision edge:")
        print(f"  nodes    : {e.node_ids}")
        print(f"  metadata : {e.metadata}")
