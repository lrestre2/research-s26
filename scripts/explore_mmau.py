"""
MM-AU Dataset Explorer
======================
Loads the metadata, prints statistics, and generates plots
for your Friday meeting presentation.

Usage:
    conda activate srp26
    python explore_mmau.py

Outputs (saved to ./outputs/):
    - category_distribution.png
    - temporal_timeline.png
    - weather_conditions.png
    - dataset_stats.txt
"""

import json
import os
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── paths ──────────────────────────────────────────────────────────────
DATA_DIR  = Path.home() / "data" / "mmau"
META_FILE = DATA_DIR / "video_metadata.json"
OUT_DIR   = Path("./outputs")
OUT_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("tab10")

# ── load ────────────────────────────────────────────────────────────────
print("Loading metadata...")
with open(META_FILE) as f:
    raw = json.load(f)

# Normalise: the JSON may be a list or a dict with a key
if isinstance(raw, dict):
    records = list(raw.values())
elif isinstance(raw, list):
    records = raw
else:
    raise ValueError("Unexpected JSON root type")

df = pd.DataFrame(records)
print(f"Loaded {len(df):,} video records.\n")

# ── basic stats ─────────────────────────────────────────────────────────
stats_lines = []

def log(msg=""):
    print(msg)
    stats_lines.append(msg)

log("=" * 60)
log("MM-AU DATASET STATISTICS")
log("=" * 60)
log(f"Total videos      : {len(df):,}")

for col in ["type", "weather", "light", "scenes"]:
    if col in df.columns:
        log(f"Unique {col:<12}: {df[col].nunique()}")

# accident timing columns
for col in ["t_ai", "t_co", "t_ae"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if all(c in df.columns for c in ["t_ai", "t_co", "t_ae"]):
    df["pre_accident_s"]  = df["t_ai"]
    df["accident_dur_s"]  = df["t_ae"] - df["t_ai"]
    log(f"\nAccident timing (seconds):")
    log(f"  Avg pre-accident clip  : {df['pre_accident_s'].mean():.2f}s")
    log(f"  Avg accident duration  : {df['accident_dur_s'].mean():.2f}s")

# train / val / test split (7 : 1.5 : 1.5)
n = len(df)
log(f"\nOfficial split (7:1.5:1.5):")
log(f"  Train : ~{int(n * 7/10):,}")
log(f"  Val   : ~{int(n * 1.5/10):,}")
log(f"  Test  : ~{int(n * 1.5/10):,}")

log()
log("SOTA Baselines on MM-AU:")
log("  Object Detection (mAP50) :")
log("    YOLOv5s          : 0.757 (val) / 0.748 (test)")
log("    DiffusionDet     : 0.731 (val) / 0.733 (test)")
log("  Accident Reason Answering (Accuracy):")
log("    SeViLA           : 89.26% (val) / 89.02% (test)")
log("    CoVGT            : ~80%  (val)")
log()
log("KEY LIMITATION: ALL baselines are single-view (ego / dashcam only).")
log("No model leverages multi-view or relational scene structure.")

stats_path = OUT_DIR / "dataset_stats.txt"
stats_path.write_text("\n".join(stats_lines))
print(f"\nStats saved → {stats_path}")

# ── plot 1: accident category distribution ──────────────────────────────
if "type" in df.columns:
    cat_counts = df["type"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(14, 5))
    cat_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("MM-AU: Video count per accident category (1–58)", fontsize=14)
    ax.set_xlabel("Accident category ID")
    ax.set_ylabel("Number of videos")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "category_distribution.png", dpi=150)
    plt.close()
    print("Saved → category_distribution.png")

# ── plot 2: accident temporal structure ─────────────────────────────────
if "pre_accident_s" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    df["pre_accident_s"].clip(0, 30).hist(bins=40, ax=axes[0], color="coral", edgecolor="white")
    axes[0].set_title("Pre-accident clip length (s)")
    axes[0].set_xlabel("Seconds before accident")

    df["accident_dur_s"].clip(0, 10).hist(bins=40, ax=axes[1], color="steelblue", edgecolor="white")
    axes[1].set_title("Accident event duration (s)")
    axes[1].set_xlabel("Duration (seconds)")

    fig.suptitle("MM-AU: Temporal structure of accident events", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "temporal_timeline.png", dpi=150)
    plt.close()
    print("Saved → temporal_timeline.png")

# ── plot 3: conditions (weather / lighting) ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, col in zip(axes, ["weather", "light"]):
    if col in df.columns:
        counts = df[col].value_counts().head(10)
        counts.plot(kind="barh", ax=ax, color="mediumseagreen", edgecolor="white")
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel("Count")
        ax.invert_yaxis()
    else:
        ax.set_visible(False)

fig.suptitle("MM-AU: Environmental conditions", fontsize=13)
plt.tight_layout()
fig.savefig(OUT_DIR / "weather_conditions.png", dpi=150)
plt.close()
print("Saved → weather_conditions.png")

# ── plot 4: hypergraph motivation diagram ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: flat CNN baseline (single frame, independent)
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
ax.set_title("SOTA: Single-view, flat CNN\n(each frame independent)", fontsize=11, color="gray")
for i, (x, y, lbl) in enumerate([(2,4,"Car"), (5,4,"Ped."), (8,4,"Cyclist"),
                                   (2,2,"Car"), (5,2,"Ped."), (8,2,"Cyclist")]):
    ax.add_patch(plt.Circle((x, y), 0.6, color="steelblue", alpha=0.7))
    ax.text(x, y, lbl, ha="center", va="center", fontsize=8, color="white", fontweight="bold")
ax.text(5, 5.2, "Frame t", ha="center", fontsize=9, style="italic")
ax.text(5, 0.7, "Frame t+1", ha="center", fontsize=9, style="italic")
ax.annotate("", xy=(5, 3.4), xytext=(5, 2.6), arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
ax.text(5.3, 3.0, "No edge\nmodelling", fontsize=7, color="red")

# Right: hypergraph (nodes + hyperedges across frames + views)
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
ax.set_title("Proposed: Situation Hyper-Graph\n(multi-frame + multi-view edges)", fontsize=11, color="darkgreen")
positions = {"Car1":(1.5,4.5), "Ped":(5,4.8), "Cyclist":(8.5,4.5),
             "Car2":(1.5,1.5), "Truck":(5,1.2), "Car3":(8.5,1.5)}
colors = {"Car1":"steelblue","Ped":"coral","Cyclist":"mediumseagreen",
          "Car2":"steelblue","Truck":"mediumpurple","Car3":"steelblue"}
for lbl, (x,y) in positions.items():
    ax.add_patch(plt.Circle((x, y), 0.55, color=colors[lbl], alpha=0.85, zorder=3))
    ax.text(x, y, lbl[:4], ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")

# hyperedge 1: collision group (Car1, Cyclist, Car2)
he1_pts = np.array([[1.5,4.5],[8.5,4.5],[8.5,1.5],[1.5,1.5]])
poly1 = plt.Polygon(he1_pts, closed=True, fill=True, facecolor="orange", alpha=0.12, edgecolor="orange", lw=2, linestyle="--", zorder=1)
ax.add_patch(poly1)
ax.text(5, 3.1, "Collision\nhyperedge", ha="center", fontsize=8, color="darkorange")

# hyperedge 2: temporal link (Ped across frames)
ax.annotate("", xy=(5, 1.75), xytext=(5, 4.25),
            arrowprops=dict(arrowstyle="<->", color="darkgreen", lw=1.8), zorder=4)
ax.text(5.4, 3.0, "Temporal\nedge", fontsize=7, color="darkgreen")

ax.text(5, 5.4, "Frame t (ego-view)", ha="center", fontsize=9, style="italic")
ax.text(5, 0.5, "Frame t+1  ✚  infra-view (proposed)", ha="center", fontsize=9, style="italic", color="darkgreen")

patch1 = mpatches.Patch(color="orange", alpha=0.4, label="Collision hyperedge")
patch2 = mpatches.Patch(color="darkgreen", alpha=0.7, label="Temporal hyperedge")
ax.legend(handles=[patch1, patch2], loc="lower right", fontsize=8)

plt.suptitle("Why Situation Hyper-Graphs improve on SOTA baselines", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "hypergraph_motivation.png", dpi=150)
plt.close()
print("Saved → hypergraph_motivation.png")

print("\n=== All outputs saved to ./outputs/ ===")
print("Open dataset_stats.txt and the PNG files for your meeting.")
