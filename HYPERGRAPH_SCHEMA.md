# Hypergraph Schema for MM-AU
### Design document — SRP 2026, Project 2

---

## Why a hypergraph?

A regular graph can only draw an edge between **two** things at a time.
A traffic accident usually involves **more than two** things at once —
a car, a cyclist, a traffic light, a wet road surface, all interacting
simultaneously. A **hyperedge** can connect any number of nodes in one
go, which makes it a natural fit for accident scenes.

Example: "Car A ran a red light and hit Cyclist B, who then fell into Car C."
In a regular graph this needs 3 separate edges: A→B, B→C, A→C.
In a hypergraph it's one collision hyperedge: {A, B, C}.

---

## What a "situation" is

Each accident clip from MM-AU is 16 frames long, sampled uniformly from
the accident window (t_ai → t_ae, roughly 10 seconds).
One **situation hypergraph** represents the entire clip — all objects
across all frames, connected by typed hyperedges.

---

## Nodes

Every detected object in every frame becomes one node.

| Property   | Type    | Description |
|------------|---------|-------------|
| `node_id`  | string  | `"{video_id}_f{frame_idx}_o{obj_id}"` — globally unique |
| `frame_idx`| int     | Which of the 16 frames (0–15) |
| `obj_id`   | int     | YOLO detection index within that frame |
| `obj_class`| string  | One of: car, pedestrian, cyclist, truck, bus, traffic light, motorbike |
| `bbox`     | [x,y,w,h] | Bounding box in pixels |
| `conf`     | float   | YOLO confidence score (0–1) |
| `features` | vector  | Visual embedding from SlowFast (added later; None for now) |

**Example:** A car detected in frame 3 of video ABC123:
```
node_id   = "ABC123_f03_o0"
frame_idx = 3
obj_class = "car"
bbox      = [420, 210, 180, 90]
conf      = 0.91
```

---

## Hyperedge types

### 1. SPATIAL
**What it means:** These objects are physically close to each other in the same frame.
**When it's created:** When the distance between object centres is less than 30% of the frame width.
**Why it matters:** Tells the model which objects are in each other's danger zone at a given moment.

```
edge_type = "spatial"
node_ids  = ["ABC123_f03_o0", "ABC123_f03_o2", "ABC123_f03_o5"]
metadata  = {"frame_idx": 3}
```

### 2. TEMPORAL
**What it means:** This is the same object tracked from one frame to the next.
**When it's created:** When two objects in consecutive frames share the same class and their bounding boxes overlap by more than 40% (IoU > 0.4).
**Why it matters:** Gives the model a sense of trajectory — how each object is moving through the scene over time.

```
edge_type = "temporal"
node_ids  = ["ABC123_f03_o0", "ABC123_f04_o0"]
metadata  = {"iou": 0.72, "frames": [3, 4]}
```

### 3. COLLISION
**What it means:** Two objects of different types whose bounding boxes physically overlap — they are touching or hitting each other.
**When it's created:** IoU > 0.1 between objects of different classes during the accident window.
**Why it matters:** This is the causal edge. It marks the actual moment and participants of the crash. A model that can identify collision edges correctly understands who was involved.

```
edge_type = "collision"
node_ids  = ["ABC123_f07_o0", "ABC123_f07_o3"]
metadata  = {"iou": 0.24, "frame_idx": 7, "classes": ["car", "cyclist"]}
```

### 4. MULTIVIEW *(future — not yet active)*
**What it means:** The same physical object seen from two different cameras simultaneously.
**When it's created:** When we have infrastructure-camera or drone footage alongside the dashcam (planned via CARLA simulation).
**Why it matters:** This is the multi-view extension. A hyperedge across views lets the model reason about the same event from multiple angles simultaneously — something impossible with standard pairwise edges.

```
edge_type = "multiview"
node_ids  = ["ABC123_ego_f07_o0", "ABC123_infra_f07_o2"]
metadata  = {"cameras": ["ego", "infrastructure"], "frame_idx": 7}
```

---

## Full example — one clip

```
Video: ABC123  (ego-car hitting car, category 11)
Frames: 16 frames from t=12.0s to t=22.0s

Nodes (example subset):
  ABC123_f00_o0  car         frame 0
  ABC123_f00_o1  pedestrian  frame 0
  ABC123_f01_o0  car         frame 1   ← same car as f00_o0
  ABC123_f07_o0  car         frame 7   ← ego vehicle
  ABC123_f07_o2  cyclist     frame 7   ← cyclist hit

Hyperedges:
  spatial   {f00_o0, f00_o1}                  — car and pedestrian close in frame 0
  temporal  {f00_o0, f01_o0}                  — car tracked frame 0→1
  temporal  {f06_o0, f07_o0}                  — ego car tracked frame 6→7
  collision {f07_o0, f07_o2}                  — car hits cyclist at frame 7
```

---

## What the model does with the hypergraph

1. Each node gets an initial feature vector (from the visual embedding or bbox features).
2. A **Hypergraph Neural Network (HGNN)** passes messages along hyperedges — each node aggregates information from all nodes in its hyperedges.
3. The updated node features are pooled into a single graph-level representation.
4. This representation is combined with the question encoding (BERT) via cross-attention.
5. The result is projected to an answer — either a category label or a free-text reason.

---

## What we are NOT doing yet (but planned)

| Feature | Status | When |
|---------|--------|------|
| SlowFast visual features | Planned | After baseline confirms the schema works |
| Multi-view (CARLA data) | Planned | Week 4–5 |
| Causal hyperedges (responsibility attribution) | Planned | Ties into Chris's work on Project 1 |
| Per-category accuracy breakdown | **This week** | After baseline runs |
