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
) -> SituationHypergraph:
    """
    Build a SituationHypergraph from a video's scene graph.

    Hyperedge types produced here:
      - SPATIAL  : objects geometrically close in the same frame
      - TEMPORAL : same object tracked across consecutive frames

    COLLISION hyperedges are intentionally NOT generated by geometric
    rules here. Bbox overlap in 2D image space is an unreliable proxy
    for actual collisions (objects overlap due to perspective all the
    time). Instead, collision groupings are what SituationHGNN *learns*
    from accident labels — that is the contribution of this work.

    Args:
        video_id       : identifier string for the video
        scene_graph    : list of per-frame dicts from preprocess.py
        frame_width    : pixel width of the frames (default 1280)
        spatial_thresh : centre-distance threshold (× frame_width) for SPATIAL edges
        temporal_iou   : min IoU to link the same object across frames

    Returns:
        SituationHypergraph with SPATIAL + TEMPORAL edges, ready for HGNN
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
                # scene_graph_gen.py uses "label"; some schemas use "class"
                obj_class = obj.get("label") or obj.get("class", "unknown"),
                bbox      = obj["bbox"],
                conf      = obj.get("conf", 1.0),
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

    # ── 4. COLLISION hyperedges — LLaVA-grounded only ────────────────────
    #
    # We only create a collision hyperedge when LLaVA explicitly said so.
    # This is still imperfect (LLaVA can hallucinate) but is far more
    # reliable than a 2D bbox-IoU threshold, which fires constantly due
    # to perspective projection.
    #
    # The canonical collision groupings come from SituationHGNN training.
    # These LLaVA-grounded edges serve as a weak initialisation signal.

    COLLISION_PREDICATES = {
        "colliding_with", "cutting_off", "hitting", "crashing_into",
    }

    for frame in scene_graph:
        fi = frame.get("frame_idx", 0)
        obj_by_id = {o["id"]: o for o in frame.get("objects", [])}

        for rel in frame.get("relations", []):
            if rel.get("source") != "llava":
                continue
            if rel.get("predicate", "") not in COLLISION_PREDICATES:
                continue

            s_id = rel.get("subject")
            o_id = rel.get("object")
            if s_id not in obj_by_id or o_id not in obj_by_id:
                continue

            s_nid = f"{video_id}_f{fi:02d}_o{s_id}"
            o_nid = f"{video_id}_f{fi:02d}_o{o_id}"
            if s_nid not in node_map or o_nid not in node_map:
                continue

            eid = f"{video_id}_collision_{edge_counter}"
            edge_counter += 1
            hg.add_edge(Hyperedge(
                edge_id   = eid,
                edge_type = "collision",
                node_ids  = [s_nid, o_nid],
                metadata  = {
                    "predicate" : rel["predicate"],
                    "frame_idx" : fi,
                    "source"    : "llava",
                    "classes"   : [
                        obj_by_id[s_id].get("label", "?"),
                        obj_by_id[o_id].get("label", "?"),
                    ],
                },
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


# ══════════════════════════════════════════════════════════════════════════
# HYPERGRAPH NEURAL NETWORK (HGNN)
# ══════════════════════════════════════════════════════════════════════════
#
# With the soft incidence matrix H from LearnableHyperedgeConstructor,
# we can do proper spectral convolution on the hypergraph.
#
# The HGNN propagation rule (Feng et al., AAAI 2019):
#
#   X' = σ( D_v^{-1/2}  H  W  D_e^{-1}  H^T  D_v^{-1/2}  X  Θ )
#
# Where:
#   H   [N, K]  : incidence matrix (node i ∈ hyperedge k → H[i,k] > 0)
#   W   [K]     : per-hyperedge importance weights (learnable or uniform)
#   D_v [N]     : node degree  D_v[i] = Σ_k W[k] * H[i,k]
#   D_e [K]     : edge degree  D_e[k] = Σ_i H[i,k]
#   Θ   [d, d'] : learnable linear projection (nn.Linear)
#
# Because H is produced by LearnableHyperedgeConstructor (a differentiable
# MLP), the entire pipeline — from raw node features to classification
# logits — is end-to-end trainable via backpropagation.
#
# SituationHGNN stacks two HypergraphConv layers followed by global mean
# pooling and a linear classifier:
#
#   node_feats [N, d]
#       → LearnableHyperedgeConstructor → H [N, K]
#       → HypergraphConv(d → hidden)    → [N, hidden]
#       → HypergraphConv(hidden → hidden)→ [N, hidden]
#       → mean over N nodes             → [hidden]
#       → Linear                        → logits [n_classes]
# ══════════════════════════════════════════════════════════════════════════

if _TORCH_AVAILABLE:

    class HypergraphConv(nn.Module):
        """
        One layer of spectral Hypergraph Neural Network convolution.

        Implements the HGNN update rule from Feng et al. (AAAI 2019):

            X' = σ( D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X Θ )

        When H is the soft output of LearnableHyperedgeConstructor,
        gradients flow back through H and the entire pipeline is
        end-to-end differentiable.

        Args:
            in_dim  : input feature dimension per node
            out_dim : output feature dimension per node
            bias    : include bias in the linear projection Θ
        """

        def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
            super().__init__()
            self.theta = nn.Linear(in_dim, out_dim, bias=bias)

        def forward(
            self,
            X: "torch.Tensor",                          # [N, in_dim]
            H: "torch.Tensor",                          # [N, K]
            edge_weights: "torch.Tensor | None" = None, # [K]
        ) -> "torch.Tensor":                            # [N, out_dim]
            """
            Args:
                X            : node feature matrix [N, in_dim]
                H            : incidence matrix [N, K] — values in (0, 1)
                edge_weights : per-hyperedge scalar weights [K] (default: 1s)
            Returns:
                Updated node features [N, out_dim]
            """
            N, K = H.shape
            W = edge_weights if edge_weights is not None else H.new_ones(K)

            # ── Degree matrices ──────────────────────────────────────────
            # D_v[i] = Σ_k W[k] * H[i,k]
            Dv = (H * W.unsqueeze(0)).sum(dim=1).clamp(min=1e-6)  # [N]
            Dv_inv_sqrt = Dv.pow(-0.5).unsqueeze(1)               # [N, 1]

            # D_e[k] = Σ_i H[i,k]
            De = H.sum(dim=0).clamp(min=1e-6)                     # [K]

            # ── Propagation ──────────────────────────────────────────────
            # 1.  D_v^{-1/2} X
            X_scaled = X * Dv_inv_sqrt                            # [N, in_dim]
            # 2.  H^T D_v^{-1/2} X
            HtX = H.t() @ X_scaled                               # [K, in_dim]
            # 3.  W D_e^{-1} H^T D_v^{-1/2} X
            WHtX = (W / De).unsqueeze(1) * HtX                   # [K, in_dim]
            # 4.  H W D_e^{-1} H^T D_v^{-1/2} X
            agg = H @ WHtX                                        # [N, in_dim]
            # 5.  D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X
            out = agg * Dv_inv_sqrt                               # [N, in_dim]

            return F.relu(self.theta(out))                        # [N, out_dim]


    class SituationHGNN(nn.Module):
        """
        Full learnable Situation Hypergraph Neural Network for MM-AU.

        This is our main model:
          1. LearnableHyperedgeConstructor learns which nodes form hyperedges.
          2. Two HypergraphConv layers do spectral message-passing.
          3. Global mean pooling collapses the node dimension.
          4. A linear head maps to accident categories.

        All four steps are differentiable — backprop simultaneously
        optimises the hyperedge structure AND the classification boundary.

        This directly extends Lohner et al. (IAVVC 2024): they use
        a fixed pairwise scene graph; we use a learned hypergraph that
        can encode multi-body interactions (e.g. car + cyclist + truck
        all involved in the same collision cluster).

        Args:
            node_feat_dim : dimension of per-node input features
            hidden_dim    : hidden dimension inside HGNN layers
            n_classes     : number of accident categories to classify
            n_edges       : number of learnable hyperedges K
        """

        def __init__(
            self,
            node_feat_dim : int = 64,
            hidden_dim    : int = 128,
            n_classes     : int = 58,
            n_edges       : int = 16,
        ):
            super().__init__()
            self.hyperedge_ctor = LearnableHyperedgeConstructor(
                feat_dim=node_feat_dim,
                n_edges=n_edges,
                hidden_dim=hidden_dim,
            )
            self.conv1 = HypergraphConv(node_feat_dim, hidden_dim)
            self.conv2 = HypergraphConv(hidden_dim, hidden_dim)
            self.drop  = nn.Dropout(0.3)
            self.head  = nn.Linear(hidden_dim, n_classes)

        def forward(
            self,
            node_features: "torch.Tensor",  # [N, node_feat_dim]
        ) -> "torch.Tensor":                # [n_classes]
            """
            Single-sample forward (no batch dimension — N varies per video).
            Call in a loop over the batch; see HGNNQAModel in run_baseline.py
            for the batched wrapper.

            Returns raw logits [n_classes].
            """
            H  = self.hyperedge_ctor(node_features)  # [N, K]  soft incidence matrix
            x  = self.conv1(node_features, H)         # [N, hidden]
            x  = self.conv2(x, H)                     # [N, hidden]
            x  = self.drop(x.mean(dim=0))             # [hidden] — graph-level embedding
            return self.head(x)                       # [n_classes]

        def graph_embedding(
            self,
            node_features: "torch.Tensor",  # [N, node_feat_dim]
        ) -> "torch.Tensor":                # [hidden_dim]
            """Return the graph-level embedding (before the classification head).
            Useful for downstream tasks or feature inspection."""
            with torch.no_grad():
                H = self.hyperedge_ctor(node_features)
                x = self.conv1(node_features, H)
                x = self.conv2(x, H)
                return x.mean(dim=0)

        def get_hyperedges(
            self,
            node_features: "torch.Tensor",
            threshold: float = 0.5,
        ) -> "tuple[torch.Tensor, list[list[int]]]":
            """
            Return (soft H matrix, hard hyperedge lists) for demo / analysis.

            Returns:
                H         : [N, K] soft incidence matrix
                hard_edges: list of K lists, each containing member node indices.
                            Empty hyperedges are removed.
            """
            with torch.no_grad():
                H = self.hyperedge_ctor(node_features)
            hard_edges = self.hyperedge_ctor.to_hard(node_features, threshold)
            return H, hard_edges


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
