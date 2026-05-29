"""
Demo Pipeline — MM-AU: Scene Graph → Situation Hypergraph → HGNN
=================================================================
Demonstrates the full pipeline from a dashcam accident video through
zero-shot scene graph generation, rule-based hypergraph construction,
and learnable hypergraph neural network classification.

Designed to run on CPU (no GPU required) using either:
  (a) a real preprocessed scene graph JSON from ~/data/mmau/processed/
  (b) a synthetic toy example (automatically used if (a) not available)

Usage:
    PYTHONPATH=. python mmau_adapter/demo_pipeline.py
    PYTHONPATH=. python mmau_adapter/demo_pipeline.py --synthetic
"""

import json
import argparse
from pathlib import Path
from collections import Counter

# ── Synthetic toy example ─────────────────────────────────────────────────
# Three-frame snippet of a "car runs red light → hits cyclist" accident.
SYNTHETIC_SCENE_GRAPH = [
    {
        "frame_idx": 0, "timestamp_s": 0.0,
        "objects": [
            {"id": 0, "label": "car",        "bbox": [100, 200, 200, 120], "conf": 0.93, "attributes": ["moving"]},
            {"id": 1, "label": "cyclist",    "bbox": [530, 215, 80,  130], "conf": 0.87, "attributes": ["vulnerable"]},
            {"id": 2, "label": "pedestrian", "bbox": [680, 350, 60,  140], "conf": 0.80, "attributes": []},
        ],
        "relations": [
            {"subject": 0, "predicate": "approaching",   "object": 1, "source": "llava"},
            {"subject": 0, "predicate": "near",          "object": 1, "source": "spatial"},
            {"subject": 1, "predicate": "crossing_path_of", "object": 0, "source": "llava"},
        ],
    },
    {
        "frame_idx": 1, "timestamp_s": 3.0,
        "objects": [
            {"id": 0, "label": "car",        "bbox": [290, 200, 200, 120], "conf": 0.94, "attributes": ["braking"]},
            {"id": 1, "label": "cyclist",    "bbox": [350, 210, 80,  130], "conf": 0.89, "attributes": ["falling"]},
            {"id": 2, "label": "pedestrian", "bbox": [660, 350, 60,  140], "conf": 0.78, "attributes": []},
            {"id": 3, "label": "truck",      "bbox": [760, 180, 220, 160], "conf": 0.91, "attributes": ["stopped"]},
        ],
        "relations": [
            {"subject": 0, "predicate": "colliding_with",     "object": 1, "source": "llava"},
            {"subject": 3, "predicate": "stopped_in_front_of","object": 0, "source": "llava"},
        ],
    },
    {
        "frame_idx": 2, "timestamp_s": 6.0,
        "objects": [
            {"id": 0, "label": "car",        "bbox": [380, 200, 200, 120], "conf": 0.88, "attributes": ["stopped"]},
            {"id": 1, "label": "cyclist",    "bbox": [390, 210, 80,  130], "conf": 0.82, "attributes": ["fallen"]},
            {"id": 2, "label": "pedestrian", "bbox": [630, 345, 60,  140], "conf": 0.76, "attributes": []},
        ],
        "relations": [
            {"subject": 0, "predicate": "near",     "object": 1, "source": "spatial"},
            {"subject": 2, "predicate": "near",     "object": 1, "source": "spatial"},
        ],
    },
]

SGRAPH_DIR = Path.home() / "data" / "mmau" / "processed" / "scene_graphs"


# ── Data loading ──────────────────────────────────────────────────────────

def load_scene_graph(synthetic: bool) -> tuple[str, list[dict]]:
    """Return (video_id, list-of-per-frame-dicts)."""
    if not synthetic:
        sg_files = list(SGRAPH_DIR.glob("*.json"))
        if sg_files:
            data = json.loads(sg_files[0].read_text())
            if isinstance(data, dict):
                sgs = data.get("scene_graphs", [data])
                vid = data.get("video_id", sg_files[0].stem)
            else:
                sgs, vid = data, sg_files[0].stem
            return vid, sgs
        print("  No preprocessed scene graphs found — using synthetic example.\n")
    return "synthetic_v001_cat11", SYNTHETIC_SCENE_GRAPH


# ── Pretty printers ───────────────────────────────────────────────────────

SEP = "─" * 62

def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def print_scene_graph(video_id: str, scene_graphs: list[dict]):
    section(f"STEP 1 — ZERO-SHOT SCENE GRAPH   (video: {video_id})")
    print()
    print("  Tool chain:  Grounding DINO (open-vocab detection)")
    print("               └─ LLaVA-1.5-7B (semantic relation extraction)")
    print()
    for sg in scene_graphs:
        objs = [f"{o['label']}({o['conf']:.2f})" for o in sg["objects"]]
        llava_rels = [
            f"obj{r['subject']} ─[{r['predicate']}]→ obj{r['object']}"
            for r in sg["relations"] if r.get("source") == "llava"
        ]
        print(f"  Frame {sg['frame_idx']}  t={sg['timestamp_s']:.1f}s")
        print(f"    Detected : {', '.join(objs)}")
        if llava_rels:
            print(f"    Relations: {' | '.join(llava_rels)}")
        else:
            print(f"    Relations: (spatial only — LLaVA not run yet)")
        print()


def print_rule_based_hypergraph(hg):
    section("STEP 2 — RULE-BASED HYPERGRAPH  (baseline, for comparison)")
    type_counts = Counter(e.edge_type for e in hg.hyperedges)
    print(f"\n  Nodes      : {len(hg.nodes)}")
    print(f"  Hyperedges : {len(hg.hyperedges)}")
    for etype in ("spatial", "temporal", "collision", "multiview"):
        n = type_counts.get(etype, 0)
        print(f"    {etype:<12}: {n} hyperedge{'s' if n != 1 else ''}")

    if hg.hyperedges:
        print(f"\n  First few hyperedges:")
        for e in hg.hyperedges[:5]:
            meta_str = " | ".join(f"{k}={v}" for k, v in e.metadata.items())
            print(f"    [{e.edge_type:<12}]  nodes={e.node_ids}  {meta_str}")

    print()
    print("  Limitation: thresholds are hand-tuned (0.3 × frame width for")
    print("  spatial, IoU > 0.4 for temporal). Multi-body groups require")
    print("  multiple pairwise edges — cannot be represented as one hyperedge.")


def print_learnable_hypergraph(scene_graphs: list[dict]):
    section("STEP 3 — LEARNABLE HYPEREDGE CONSTRUCTOR")
    try:
        import torch
        from mmau_adapter.hypergraph import node_features_from_sg, LearnableHyperedgeConstructor
    except ImportError as e:
        print(f"\n  [SKIPPED] PyTorch unavailable: {e}")
        return

    node_feats = node_features_from_sg(scene_graphs, feat_dim=64)  # [N, 64]
    N = node_feats.shape[0]

    constructor = LearnableHyperedgeConstructor(feat_dim=64, n_edges=16, hidden_dim=128)
    constructor.eval()
    with torch.no_grad():
        H = constructor(node_feats)   # [N, K=16]

    active_k  = int((H.max(dim=0).values > 0.3).sum())
    hard_mask = H > 0.5
    avg_size  = hard_mask.float().sum(0)
    avg_size  = float(avg_size[avg_size > 0].mean()) if active_k else 0.0

    print(f"\n  Input feature matrix X  : [{N} nodes × 64 dims]")
    print(f"  Soft incidence matrix H : [{N} nodes × 16 hyperedges]")
    print(f"  (H[i,k] = probability that node i belongs to hyperedge k)")
    print()
    print(f"  Active hyperedges : {active_k}/16  (at least one strong member)")
    print(f"  Avg members/edge  : {avg_size:.1f}  (hard threshold > 0.5)")
    print()
    print(f"  Node 0 membership scores across all 16 hyperedges:")
    scores = H[0].tolist()
    bar    = lambda v: "█" * int(v * 20)
    for k, s in enumerate(scores):
        print(f"    k={k:2d}  {bar(s):<20}  {s:.3f}")

    print()
    print("  Key: these are RANDOM (untrained) weights — after training,")
    print("  the network learns to put 'car + cyclist in collision' in the")
    print("  same hyperedge, and 'background objects' in sparse ones.")


def print_hgnn(scene_graphs: list[dict]):
    section("STEP 4 — HYPERGRAPH NEURAL NETWORK (HGNN)")
    try:
        import torch
        from mmau_adapter.hypergraph import node_features_from_sg, SituationHGNN
    except ImportError as e:
        print(f"\n  [SKIPPED] PyTorch unavailable: {e}")
        return

    node_feats = node_features_from_sg(scene_graphs, feat_dim=64)
    N = node_feats.shape[0]

    hgnn = SituationHGNN(node_feat_dim=64, hidden_dim=128, n_classes=58, n_edges=16)
    hgnn.eval()
    with torch.no_grad():
        logits = hgnn(node_feats)     # [58]
        emb    = hgnn.graph_embedding(node_feats)  # [128]
        H, hard_edges = hgnn.get_hyperedges(node_feats, threshold=0.5)

    print(f"\n  Propagation rule (Feng et al., AAAI 2019):")
    print(f"    X' = σ( Dv^(-½) H W De^(-1) Hᵀ Dv^(-½) X Θ )")
    print()
    print(f"  Forward pass:")
    print(f"    X  [{N:2d} × 64]   node features")
    print(f"    → LearnableHyperedgeConstructor")
    print(f"    H  [{N:2d} × 16]   soft incidence matrix  ← learned, differentiable")
    print(f"    → HypergraphConv(64 → 128)  — spectral message passing over H")
    print(f"    → HypergraphConv(128 → 128) — second round of propagation")
    print(f"    → mean pool over {N} nodes")
    print(f"    g  [128]          graph-level embedding")
    print(f"    → Linear(128 → 58)")
    print(f"    z  [58]           logits over accident categories")
    print()
    print(f"  Output shapes:")
    print(f"    Graph embedding : {list(emb.shape)}")
    print(f"    Logits          : {list(logits.shape)}")
    print(f"    Hard hyperedges : {len(hard_edges)} non-empty (threshold 0.5)")
    print()
    print(f"  Why this beats Lohner et al. (IAVVC 2024):")
    print(f"    • They use a fixed pairwise graph (edges connect exactly 2 nodes)")
    print(f"    • We use learnable hyperedges (edges connect ANY number of nodes)")
    print(f"    • A hyperedge can capture 'car + cyclist + truck all involved'")
    print(f"    • H is learned from data — not hand-tuned distance thresholds")
    print(f"    • End-to-end: ∂loss/∂H propagates back to the MLP that builds H")


def print_pipeline_summary():
    section("FULL PIPELINE SUMMARY")
    print()
    print("  ① Grounding DINO  →  objects + bounding boxes  (zero-shot detection)")
    print("  ② LLaVA-1.5-7B   →  semantic relations         (zero-shot VQA)")
    print("                          ↓")
    print("  ③ LearnableHyperedgeConstructor")
    print("        node features [N×64] → soft incidence matrix H [N×K]")
    print("        H is a differentiable MLP output — trained end-to-end")
    print("                          ↓")
    print("  ④ HypergraphConv × 2")
    print("        spectral message passing: X' = Dv^(-½) H W De^(-1) Hᵀ Dv^(-½) X Θ")
    print("        nodes aggregate from ALL members of each hyperedge at once")
    print("                          ↓")
    print("  ⑤ Mean pool → Linear → accident category (58 classes, MM-AU)")
    print()
    print("  Comparison with prior work:")
    print(f"  {'Method':<35} {'Graph type':<20} {'Categories':<12} {'SGG'}")
    print(f"  {'─'*35} {'─'*20} {'─'*12} {'─'*15}")
    print(f"  {'Lohner et al. IAVVC 2024':<35} {'pairwise, fixed':<20} {'4 (DoTA)':<12} {'rule-based'}")
    print(f"  {'Ours (MM-AU)':<35} {'hypergraph, learned':<20} {'58 (MM-AU)':<12} {'zero-shot'}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MM-AU pipeline demo for meeting")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force the synthetic toy example")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   MM-AU: Scene Graph  →  Situation Hypergraph  →  HGNN      ║")
    print("║   Liu Restrepo — SRP 2026 (Prof. Cheng lab)                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    from mmau_adapter.hypergraph import build_hypergraph

    video_id, scene_graphs = load_scene_graph(args.synthetic)

    print_scene_graph(video_id, scene_graphs)
    hg = build_hypergraph(video_id, scene_graphs, frame_width=1280)
    print_rule_based_hypergraph(hg)
    print_learnable_hypergraph(scene_graphs)
    print_hgnn(scene_graphs)
    print_pipeline_summary()


if __name__ == "__main__":
    main()
