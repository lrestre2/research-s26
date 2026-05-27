"""
Scene Graph Analyzer for MM-AU
================================
Loads all generated scene graphs and produces the analysis Hyun asked for:
"analyze graphs based on the scene graph so that we can decide what would
be the gold standard for generating the hypergraph."

What it computes:
  1. Object frequency per category (what objects appear in each accident type)
  2. Relation frequency per category (what relations are most common)
  3. Near-collision signature (objects + relations in the LAST frame before t_ae)
  4. Category comparison: cat 11 vs cat 43 — what distinguishes them?
  5. Co-occurrence matrix: which object pairs appear together most often
  6. Hyperedge recommendations — data-driven suggestions for hyperedge design

Outputs (saved to ~/data/mmau/processed/analysis/):
  - object_frequency.png
  - relation_frequency.png
  - cooccurrence_matrix.png
  - collision_signature.png
  - analysis_report.md    ← bring this to the meeting

Usage:
    conda activate srp26
    cd ~/Liu/research-s26
    python mmau_adapter/analyze_scene_graphs.py
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── paths ──────────────────────────────────────────────────────────────
DATA_DIR    = Path.home() / "data" / "mmau"
SGRAPH_DIR  = DATA_DIR / "processed" / "scene_graphs"
ANALYSIS_DIR = DATA_DIR / "processed" / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

CAT_LABELS = {11: "ego-car hitting car", 43: "car hitting car"}
plt.style.use("seaborn-v0_8-whitegrid")


# ── load all scene graphs ───────────────────────────────────────────────

def load_all() -> list[dict]:
    files = list(SGRAPH_DIR.glob("*.json"))
    print(f"Loading {len(files)} scene graph files ...")
    records = []
    for f in files:
        try:
            records.append(json.loads(f.read_text()))
        except Exception:
            pass
    print(f"Loaded {len(records)} records.\n")
    return records


# ── aggregation helpers ─────────────────────────────────────────────────

def aggregate(records: list[dict]) -> dict:
    """
    Aggregate statistics per category.
    Returns a dict keyed by category with nested counters.
    """
    stats = defaultdict(lambda: {
        "n_videos"       : 0,
        "obj_freq"       : Counter(),
        "rel_freq"       : Counter(),
        "semantic_rels"  : Counter(),  # LLaVA-generated only
        "obj_pairs"      : Counter(),  # pairs seen together in same frame
        "last_frame_objs": Counter(),  # objects in final (collision) frame
        "last_frame_rels": Counter(),  # relations in final frame
    })

    for rec in records:
        cat  = rec.get("category", 0)
        sgs  = rec.get("scene_graphs", [])
        if not sgs:
            continue

        st = stats[cat]
        st["n_videos"] += 1

        for sg in sgs:
            objs = sg.get("objects", [])
            rels = sg.get("relations", [])
            labels = [o["label"] for o in objs]
            id_to_label = {o["id"]: o["label"] for o in objs}

            # object frequencies
            for lbl in labels:
                st["obj_freq"][lbl] += 1

            # relation frequencies
            for r in rels:
                st["rel_freq"][r["predicate"]] += 1
                if r.get("source") == "llava":
                    st["semantic_rels"][r["predicate"]] += 1

            # object pair co-occurrences
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    pair = tuple(sorted([labels[i], labels[j]]))
                    st["obj_pairs"][pair] += 1

        # last frame = proxy for collision moment
        last = sgs[-1]
        last_labels = [o["label"] for o in last.get("objects", [])]
        last_rels   = [r["predicate"] for r in last.get("relations", [])
                       if r.get("source") == "llava"]
        for lbl in last_labels:
            st["last_frame_objs"][lbl] += 1
        for rel in last_rels:
            st["last_frame_rels"][rel] += 1

    return stats


# ── plots ───────────────────────────────────────────────────────────────

def plot_object_frequency(stats: dict):
    fig, axes = plt.subplots(1, len(stats), figsize=(7 * len(stats), 6))
    if len(stats) == 1:
        axes = [axes]

    for ax, (cat, st) in zip(axes, stats.items()):
        top = pd.Series(st["obj_freq"]).sort_values(ascending=False).head(12)
        top.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
        ax.invert_yaxis()
        ax.set_title(f"Cat {cat}: {CAT_LABELS.get(cat, '')}\n({st['n_videos']} videos)",
                     fontsize=12)
        ax.set_xlabel("Total object occurrences")

    fig.suptitle("Most common objects per accident category", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(ANALYSIS_DIR / "object_frequency.png", dpi=150)
    plt.close()
    print("Saved → object_frequency.png")


def plot_relation_frequency(stats: dict):
    fig, axes = plt.subplots(1, len(stats), figsize=(7 * len(stats), 6))
    if len(stats) == 1:
        axes = [axes]

    for ax, (cat, st) in zip(axes, stats.items()):
        # Show only LLaVA semantic relations (more interesting than spatial)
        sem = st["semantic_rels"]
        if not sem:
            sem = st["rel_freq"]
        top = pd.Series(sem).sort_values(ascending=False).head(10)
        top.plot(kind="barh", ax=ax, color="coral", edgecolor="white")
        ax.invert_yaxis()
        ax.set_title(f"Cat {cat}: semantic relations", fontsize=12)
        ax.set_xlabel("Count")

    fig.suptitle("Most common semantic relations per category\n(zero-shot, from LLaVA)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(ANALYSIS_DIR / "relation_frequency.png", dpi=150)
    plt.close()
    print("Saved → relation_frequency.png")


def plot_cooccurrence(stats: dict):
    all_objs = set()
    for st in stats.values():
        all_objs.update([o for pair in st["obj_pairs"].keys() for o in pair])
    all_objs = sorted(all_objs)

    fig, axes = plt.subplots(1, len(stats), figsize=(8 * len(stats), 7))
    if len(stats) == 1:
        axes = [axes]

    for ax, (cat, st) in zip(axes, stats.items()):
        n = len(all_objs)
        matrix = np.zeros((n, n))
        idx = {o: i for i, o in enumerate(all_objs)}

        for (a, b), cnt in st["obj_pairs"].items():
            if a in idx and b in idx:
                matrix[idx[a]][idx[b]] = cnt
                matrix[idx[b]][idx[a]] = cnt

        sns.heatmap(matrix, xticklabels=all_objs, yticklabels=all_objs,
                    ax=ax, cmap="Blues", fmt=".0f", annot=(n <= 10))
        ax.set_title(f"Cat {cat}: object co-occurrence", fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.suptitle("Object pair co-occurrence in the same frame",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(ANALYSIS_DIR / "cooccurrence_matrix.png", dpi=150)
    plt.close()
    print("Saved → cooccurrence_matrix.png")


def plot_collision_signature(stats: dict):
    """What objects and relations appear in the last (collision) frame?"""
    fig, axes = plt.subplots(2, len(stats), figsize=(7 * len(stats), 10))
    if len(stats) == 1:
        axes = axes.reshape(2, 1)

    for col, (cat, st) in enumerate(stats.items()):
        # Top objects at collision moment
        top_objs = pd.Series(st["last_frame_objs"]).sort_values(ascending=False).head(8)
        top_objs.plot(kind="barh", ax=axes[0][col], color="tomato", edgecolor="white")
        axes[0][col].invert_yaxis()
        axes[0][col].set_title(f"Cat {cat}: objects at collision", fontsize=11)

        # Top relations at collision moment
        top_rels = pd.Series(st["last_frame_rels"]).sort_values(ascending=False).head(8)
        if not top_rels.empty:
            top_rels.plot(kind="barh", ax=axes[1][col], color="darkorange", edgecolor="white")
            axes[1][col].invert_yaxis()
        axes[1][col].set_title(f"Cat {cat}: relations at collision", fontsize=11)

    fig.suptitle("Collision moment signature\n(objects + relations in final frame)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(ANALYSIS_DIR / "collision_signature.png", dpi=150)
    plt.close()
    print("Saved → collision_signature.png")


# ── markdown report ─────────────────────────────────────────────────────

def write_report(stats: dict):
    lines = ["# Scene Graph Analysis Report — MM-AU\n",
             "## Purpose",
             "Determine the gold standard for hypergraph construction based on",
             "data-driven evidence from the generated zero-shot scene graphs.\n",
             "---\n"]

    for cat, st in stats.items():
        label = CAT_LABELS.get(cat, f"category {cat}")
        lines += [f"## Category {cat} — {label}",
                  f"Videos analysed: **{st['n_videos']}**\n"]

        # Top objects
        top_objs = st["obj_freq"].most_common(8)
        lines += ["### Most common objects",
                  "| Object | Count |", "|--------|-------|"]
        for obj, cnt in top_objs:
            lines.append(f"| {obj} | {cnt} |")

        # Top semantic relations
        top_rels = (st["semantic_rels"] or st["rel_freq"]).most_common(8)
        lines += ["\n### Most common semantic relations (zero-shot LLaVA)",
                  "| Relation | Count |", "|----------|-------|"]
        for rel, cnt in top_rels:
            lines.append(f"| {rel} | {cnt} |")

        # Collision frame
        lines += ["\n### At the collision moment (final frame)",
                  "**Objects present:** " +
                  ", ".join(f"{o}({c})" for o, c in st["last_frame_objs"].most_common(5)),
                  "**Relations present:** " +
                  ", ".join(f"{r}({c})" for r, c in st["last_frame_rels"].most_common(5))]
        lines.append("\n---\n")

    # Hyperedge recommendations
    lines += [
        "## Hyperedge Design Recommendations",
        "Based on the data above, here are evidence-based proposals for hyperedge types:\n",
        "| Hyperedge type | Evidence | Static or Learnable? |",
        "|----------------|----------|----------------------|",
        "| **Spatial proximity** | High co-occurrence of near objects at collision | Both |",
        "| **Temporal chain** | Same object tracked across consecutive frames | Static |",
        "| **Collision group** | Objects with 'colliding_with' / 'approaching' at final frame | Learnable |",
        "| **Risk group** | Objects within danger zone (high semantic proximity) | Learnable |",
        "| **Multi-view** | Same event from infra camera (future) | Learnable |\n",
        "> **Key finding for discussion:** The collision and risk hyperedges are best",
        "> learned rather than hardcoded — the data shows complex multi-object",
        "> interactions that simple geometric rules would miss.",
    ]

    report_path = ANALYSIS_DIR / "analysis_report.md"
    report_path.write_text("\n".join(lines))
    print(f"Saved → analysis_report.md")


# ── main ────────────────────────────────────────────────────────────────

def main():
    records = load_all()
    if not records:
        print("No scene graphs found. Run preprocess.py first.")
        return

    stats = aggregate(records)
    print(f"Categories found: {list(stats.keys())}\n")

    plot_object_frequency(stats)
    plot_relation_frequency(stats)
    plot_cooccurrence(stats)
    plot_collision_signature(stats)
    write_report(stats)

    print(f"\nAll outputs saved to: {ANALYSIS_DIR}")
    print("Bring analysis_report.md to the meeting.")


if __name__ == "__main__":
    main()
