"""
Pipeline Visualizer — annotates real dashcam frames with what the code sees.
=============================================================================
Reads a preprocessed scene graph JSON and the original JPEG frames on disk.
Produces:
  1. annotated_frames.png  — 5 frames side-by-side, bboxes + labels + relations
  2. hypergraph.png        — spatial diagram: nodes = objects, hyperedges = hulls
  3. demo.gif              — animated version (show this at the meeting)

Everything is generated from the saved JSON — models do NOT need to re-run.

Usage (on GPU lab, after quick_demo.py has run):
    PYTHONPATH=. python mmau_adapter/visualize.py
    PYTHONPATH=. python mmau_adapter/visualize.py --video 011460
    PYTHONPATH=. python mmau_adapter/visualize.py --video 011460 --out ~/Desktop/
"""

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont

# ── paths ─────────────────────────────────────────────────────────────────
DATA_DIR   = Path.home() / "data" / "mmau"
CAP_DIR    = DATA_DIR / "CAP-DATA" / "CAP-DATA"
SGRAPH_DIR = DATA_DIR / "processed" / "scene_graphs"
VIZ_DIR    = DATA_DIR / "processed" / "visualizations"

# ── colour palette ────────────────────────────────────────────────────────
CLASS_COLORS = {
    "car":           "#4C9BE8",   # blue
    "truck":         "#F0A500",   # amber
    "bus":           "#9B59B6",   # purple
    "motorcycle":    "#E67E22",   # orange
    "bicycle":       "#27AE60",   # green
    "cyclist":       "#2ECC71",   # light green
    "pedestrian":    "#E74C3C",   # red
    "person":        "#E74C3C",
    "traffic light": "#F1C40F",   # yellow
    "other":         "#95A5A6",   # grey
}
DEFAULT_COLOR = "#95A5A6"

EDGE_COLORS = {
    "spatial":   "#3498DB",   # blue
    "temporal":  "#2ECC71",   # green
    "collision": "#E74C3C",   # red
}

RELATION_COLOR = "#FFFFFF"


# ── load data ─────────────────────────────────────────────────────────────

def find_json(video_id: str | None) -> Path:
    """Return path to a scene graph JSON."""
    files = list(SGRAPH_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No scene graph JSONs in {SGRAPH_DIR}. "
            "Run quick_demo.py or preprocess.py first."
        )
    if video_id:
        matches = [f for f in files if video_id in f.stem]
        if matches:
            return matches[0]
        print(f"  Warning: '{video_id}' not found, using {files[0].name}")
    return files[0]


def find_frame_dir(video_id: str) -> Path | None:
    """Locate the images/ directory for a video by rglob."""
    for p in CAP_DIR.rglob("images"):
        if p.is_dir() and p.parent.name == video_id:
            return p
    return None


def load_frame_image(frame_dir: Path, timestamp_s: float) -> Image.Image | None:
    """Load the JPEG closest to timestamp_s (10 fps assumed)."""
    if frame_dir is None:
        return None
    frame_num = round(timestamp_s * 10)
    candidate = frame_dir / f"{frame_num:06d}.jpg"
    if candidate.exists():
        return Image.open(candidate).convert("RGB")
    # Fall back to sorted list if off-by-one
    jpgs = sorted(frame_dir.glob("*.jpg"))
    if jpgs:
        idx = min(frame_num, len(jpgs) - 1)
        return Image.open(jpgs[idx]).convert("RGB")
    return None


# ── frame annotation (PIL) ────────────────────────────────────────────────

def _box_color(label: str) -> str:
    return CLASS_COLORS.get(label.lower(), DEFAULT_COLOR)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def annotate_frame(
    image: Image.Image,
    frame_data: dict,
    hg_membership: dict[str, str],   # node_id → edge_type (for halo colour)
) -> Image.Image:
    """
    Draw on one frame:
      - Coloured bounding box + label for each detected object
      - Line + predicate text for each LLaVA semantic relation
      - Glow border for objects involved in collision hyperedges
    """
    img = image.copy().resize((640, 360))
    scale_x = 640 / image.width
    scale_y = 360 / image.height
    draw = ImageDraw.Draw(img)

    objects = frame_data.get("objects", [])
    fi = frame_data.get("frame_idx", 0)

    # Try to load a font; fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    # Map obj id → scaled centre
    centres: dict[int, tuple[float, float]] = {}

    for obj in objects:
        x, y, w, h = obj["bbox"]
        x1 = x * scale_x
        y1 = y * scale_y
        x2 = (x + w) * scale_x
        y2 = (y + h) * scale_y
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        centres[obj["id"]] = (cx, cy)

        color = _box_color(obj.get("label", "other"))
        rgb   = _hex_to_rgb(color)

        # Check if this node is in a collision hyperedge → red glow
        vid_id  = frame_data.get("_video_id", "v")
        node_id = f"{vid_id}_f{fi:02d}_o{obj['id']}"
        if hg_membership.get(node_id) == "collision":
            draw.rectangle([x1-4, y1-4, x2+4, y2+4], outline=(231, 76, 60), width=4)

        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=2)

        # Label background + text
        label = f"{obj.get('label','?')} {obj.get('conf', 0):.2f}"
        bbox_txt = draw.textbbox((x1, y1 - 16), label, font=font)
        draw.rectangle(bbox_txt, fill=rgb)
        draw.text((x1, y1 - 16), label, fill="white", font=font)

    # Draw LLaVA semantic relations as arrows.
    # These are zero-shot VLM predictions — shown for reference only.
    # Collision predicates are highlighted in orange to signal they are
    # unverified; true collision groupings are learned by the HGNN.
    COLLISION_PREDS = {"colliding_with", "hitting", "cutting_off", "crashing_into"}
    for rel in frame_data.get("relations", []):
        if rel.get("source") != "llava":
            continue
        s_id = rel.get("subject")
        o_id = rel.get("object")
        pred = rel.get("predicate", "")
        if s_id not in centres or o_id not in centres:
            continue
        sx, sy = centres[s_id]
        ox, oy = centres[o_id]
        mid_x, mid_y = (sx + ox) / 2, (sy + oy) / 2

        # Collision predicates drawn in orange + marked as LLaVA-only
        is_collision = pred in COLLISION_PREDS
        line_color   = (255, 165, 0) if is_collision else (255, 255, 255)
        text_color   = (255, 165, 0) if is_collision else (255, 255, 0)

        draw.line([(sx, sy), (ox, oy)], fill=line_color, width=2)
        dx, dy = ox - sx, oy - sy
        ln = max((dx**2 + dy**2)**0.5, 1)
        ax = ox - 10 * dx / ln
        ay = oy - 10 * dy / ln
        draw.polygon(
            [(ox, oy),
             (ax + 5 * dy / ln, ay - 5 * dx / ln),
             (ax - 5 * dy / ln, ay + 5 * dx / ln)],
            fill=line_color,
        )
        short = pred.replace("_", " ")
        suffix = " (?)" if is_collision else ""
        draw.text((mid_x + 4, mid_y - 8), short + suffix,
                  fill=text_color, font=small_font)

    # Timestamp watermark
    draw.text((8, 8),
              f"t={frame_data.get('timestamp_s', 0):.1f}s  frame {fi}",
              fill="white", font=font)

    return img


# ── hypergraph diagram (matplotlib) ──────────────────────────────────────

def draw_hypergraph_diagram(
    video_id: str,
    scene_graphs: list[dict],
    hyperedges: list,           # list of Hyperedge dataclass instances
    frame_width: int,
    output_path: Path,
):
    """
    Draw all objects from all frames in a 2D spatial layout.
    Hyperedges are shown as coloured convex hulls around their members.
    """
    from scipy.spatial import ConvexHull

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_title(f"Situation Hypergraph — video {video_id}",
                 color="white", fontsize=14, pad=12)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.invert_yaxis()
    ax.set_xlabel("x (normalised)", color="#aaa")
    ax.set_ylabel("y (normalised)", color="#aaa")
    ax.tick_params(colors="#aaa")

    # Collect node positions + metadata
    node_pos:   dict[str, tuple[float, float]] = {}
    node_label: dict[str, str]                 = {}
    all_frames = max(len(scene_graphs), 1)

    for sg in scene_graphs:
        fi = sg.get("frame_idx", 0)
        for obj in sg.get("objects", []):
            nid = f"{video_id}_f{fi:02d}_o{obj['id']}"
            x, y, w, h = obj["bbox"]
            cx = (x + w / 2) / frame_width
            cy = (y + h / 2) / 360          # assume 360px height
            # Slight temporal jitter so overlapping frames don't collapse
            cx += fi * 0.015
            node_pos[nid]   = (cx, cy)
            node_label[nid] = obj.get("label", "?")

    # Draw hyperedge hulls first (so nodes appear on top)
    for edge in hyperedges:
        pts = [node_pos[n] for n in edge.node_ids if n in node_pos]
        if len(pts) < 2:
            continue
        color = EDGE_COLORS.get(edge.edge_type, "#888")
        alpha = 0.18
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if len(pts) == 2:
            ax.plot(xs, ys, color=color, linewidth=2, alpha=0.7,
                    linestyle="--" if edge.edge_type == "temporal" else "-")
        else:
            try:
                arr = np.array(pts)
                hull = ConvexHull(arr)
                hull_pts = arr[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])   # close polygon
                ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                        color=color, alpha=alpha)
                ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                        color=color, linewidth=1.5, alpha=0.6)
            except Exception:
                ax.plot(xs, ys, color=color, linewidth=2, alpha=0.7)

    # Draw nodes
    for nid, (cx, cy) in node_pos.items():
        label = node_label[nid]
        color = _hex_to_rgb(CLASS_COLORS.get(label.lower(), DEFAULT_COLOR))
        color_f = tuple(c / 255 for c in color)
        ax.scatter(cx, cy, color=color_f, s=160, zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.text(cx, cy - 0.035, label, color="white", fontsize=7,
                ha="center", va="bottom", zorder=6)

    # Legend
    legend_entries = [
        mpatches.Patch(color=EDGE_COLORS["spatial"],   label="spatial hyperedge"),
        mpatches.Patch(color=EDGE_COLORS["temporal"],  label="temporal hyperedge"),
        mpatches.Patch(color=EDGE_COLORS["collision"], label="collision hyperedge"),
    ]
    class_entries = [
        mpatches.Patch(
            color=tuple(c / 255 for c in _hex_to_rgb(v)),
            label=k
        )
        for k, v in CLASS_COLORS.items()
        if k in {node_label[n] for n in node_label}
    ]
    ax.legend(handles=legend_entries + class_entries,
              loc="lower right", fontsize=8,
              facecolor="#2a2a3e", labelcolor="white",
              framealpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {output_path}")


# ── animated GIF ──────────────────────────────────────────────────────────

def make_gif(frames: list[Image.Image], output_path: Path, fps: float = 1.0):
    """Save list of PIL Images as an animated GIF."""
    duration_ms = int(1000 / fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"  Saved → {output_path}")


def make_grid(frames: list[Image.Image], output_path: Path):
    """Stitch annotated frames into a single wide PNG."""
    if not frames:
        return
    w, h = frames[0].size
    grid = Image.new("RGB", (w * len(frames), h), (20, 20, 40))
    for i, f in enumerate(frames):
        grid.paste(f, (i * w, 0))
    grid.save(output_path)
    print(f"  Saved → {output_path}")


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None,
                        help="Video ID substring to select (e.g. 011460)")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: ~/data/mmau/processed/visualizations/)")
    parser.add_argument("--fps", type=float, default=0.8,
                        help="GIF playback speed in frames per second (default 0.8)")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else VIZ_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load scene graph JSON ────────────────────────────────────────────
    json_path = find_json(args.video)
    data      = json.loads(json_path.read_text())
    if isinstance(data, dict):
        video_id    = data.get("video_id", json_path.stem)
        scene_graphs = data.get("scene_graphs", [])
    else:
        video_id, scene_graphs = json_path.stem, data

    print(f"\nVideo    : {video_id}")
    print(f"Frames   : {len(scene_graphs)}")

    # ── Build rule-based hypergraph ──────────────────────────────────────
    from mmau_adapter.hypergraph import build_hypergraph
    hg = build_hypergraph(video_id, scene_graphs, frame_width=1280)
    print(f"Nodes    : {len(hg.nodes)}")
    print(f"Hyperedges: {len(hg.hyperedges)}\n")

    # Build node → edge_type membership (collision takes priority for highlight)
    hg_membership: dict[str, str] = {}
    for edge in sorted(hg.hyperedges, key=lambda e: e.edge_type):
        for nid in edge.node_ids:
            hg_membership[nid] = edge.edge_type
    # Override with collision so it wins
    for edge in hg.hyperedges:
        if edge.edge_type == "collision":
            for nid in edge.node_ids:
                hg_membership[nid] = "collision"

    # ── Find frames on disk ──────────────────────────────────────────────
    frame_dir = find_frame_dir(video_id)
    if frame_dir:
        print(f"Frames dir: {frame_dir}")
    else:
        print("  Warning: original frames not found on disk.")
        print("  Annotation will use blank placeholders.\n")

    # ── Frame width for normalization ────────────────────────────────────
    frame_width = 1280
    if frame_dir:
        sample_imgs = list(frame_dir.glob("*.jpg"))
        if sample_imgs:
            w, _ = Image.open(sample_imgs[0]).size
            frame_width = w

    # ── Annotate each frame ──────────────────────────────────────────────
    print("Annotating frames ...")
    annotated: list[Image.Image] = []

    for sg in scene_graphs:
        sg["_video_id"] = video_id    # needed inside annotate_frame
        ts = sg.get("timestamp_s", 0.0)

        if frame_dir:
            raw = load_frame_image(frame_dir, ts)
        else:
            raw = None

        if raw is None:
            raw = Image.new("RGB", (1280, 720), (30, 30, 50))

        annotated.append(annotate_frame(raw, sg, hg_membership))

    # ── Save outputs ─────────────────────────────────────────────────────
    print("\nSaving outputs ...")

    grid_path = out_dir / f"{video_id}_annotated_frames.png"
    make_grid(annotated, grid_path)

    gif_path = out_dir / f"{video_id}_demo.gif"
    make_gif(annotated, gif_path, fps=args.fps)

    hg_path = out_dir / f"{video_id}_hypergraph.png"
    try:
        from scipy.spatial import ConvexHull  # noqa: F401 — check available
        draw_hypergraph_diagram(video_id, scene_graphs, hg.hyperedges,
                                frame_width, hg_path)
    except ImportError:
        print("  scipy not available — skipping hypergraph diagram.")
        print("  Install with: pip install scipy")

    print(f"\nAll outputs in: {out_dir}")
    print("\nCopy to Mac with:")
    print(f"  scp -r trinity@<ip>:{out_dir}/ ~/Downloads/mmau_viz/")


if __name__ == "__main__":
    main()
