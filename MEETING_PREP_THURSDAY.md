# Thursday Meeting — Liu's Talking Points
## Week 2 progress: pipeline running, hypergraph built, demo ready

---

## 1. What I built since Wednesday — full summary

| What | Why |
|------|-----|
| Fixed PyTorch / CUDA | GPU lab now runs both A6000 + Blackwell without crashing |
| Fixed preprocessing path bug | `preprocess.py` no longer says "Not found: 5694" — now scans disk directly |
| Fixed LLaVA processor mismatch | `LlavaNextProcessor` (v1.6) was failing on our v1.5 checkpoint; switched to correct class |
| Scene graph generation running | `quick_demo.py` produced first real JSON from video 011460 |
| Full preprocessing kicked off | Running overnight on all 5,694 videos |
| Added `HypergraphConv` layer | Proper spectral HGNN (Feng et al. AAAI 2019 formula) |
| Added `SituationHGNN` | 2-layer learnable HGNN, end-to-end differentiable |
| Added `HGNNQAModel` | Full multimodal model: MobileNetV3 + BERT + SituationHGNN |
| Built pipeline demo | `demo_pipeline.py` — runs live, shows every step |
| Built visualizer | `visualize.py` — annotated GIF of real dashcam frames |

---

## 2. The key design decision — why no rule-based collision edges

> "I removed rule-based collision detection entirely. Bounding-box
> overlap in 2D image space is not a reliable signal — objects overlap
> in the image all the time due to camera perspective without actually
> colliding. LLaVA also hallucinates 'colliding_with' on close objects.
>
> Instead, collision groupings are what SituationHGNN **learns** from
> accident category labels during training. The model sees thousands of
> category-11 accidents and discovers for itself that the ego-car and
> the struck car belong in the same hyperedge. We never tell it that.
>
> This is a cleaner contribution: spatial and temporal edges are free
> geometric structure we give the model; collision structure it earns."

---

## 3. The HGNN — explain the math simply if asked

The core update rule (one layer):

```
X' = σ( Dv^{-½}  H  W  De^{-1}  H^T  Dv^{-½}  X  Θ )
```

In plain English:
- **H** [N × K] — which nodes belong to which hyperedges (learned, not hardcoded)
- **Dv, De** — normalisation (like dividing by degree in a regular GNN)
- **Θ** — a learnable linear projection
- Each node aggregates features from all other members of its hyperedges at once

Because **H comes from a trainable MLP**, the whole thing is differentiable.
Backprop adjusts H so the groupings become more useful for predicting accident type.

---

## 4. How it compares to the paper Hyun sent (arxiv 2407.05910)

| | Lohner et al. IAVVC 2024 | Ours |
|--|--|--|
| Graph type | Pairwise (edges connect exactly 2 nodes) | Hypergraph (edges connect any number) |
| Edge construction | Rule-based, fixed | Learned end-to-end |
| Scene graph | Rule-based SGG | Zero-shot (Grounding DINO + LLaVA) |
| Dataset | DoTA — 4 accident classes | MM-AU — 58 accident classes |
| Best result | 57.77% balanced accuracy | TBD (training in progress) |

Their method is a direct baseline we beat on all four axes.

---

## 5. What the demo shows (run this live)

```bash
PYTHONPATH=. python mmau_adapter/demo_pipeline.py
```

Walk through each step:

**Step 1** — Scene graph: "Grounding DINO detects every object, LLaVA describes
relations between them. Zero-shot — neither model saw traffic accident data."

**Step 2** — Rule-based hypergraph: "Spatial edges: objects near each other.
Temporal edges: same object tracked across frames. No collision edges — those are learned."

**Step 3** — Learnable constructor: "This MLP takes node features and outputs H,
a soft membership matrix. H[i,k] is the probability node i belongs to hyperedge k.
Before training these are random — after training they reflect collision groupings."

**Step 4** — HGNN: "Two rounds of message passing over H. Each node aggregates
from all co-members of its hyperedges. Mean pool → linear → 58-class prediction."

---

## 6. Current status + what comes next

| Step | Status |
|------|--------|
| Scene graph pipeline | ✅ Working on real data |
| Preprocessing all 5,694 videos | 🔄 Running now (overnight) |
| Analysis (`analyze_scene_graphs.py`) | ⬜ Run after preprocessing finishes |
| Training (`run_baseline.py`) | ⬜ Run after analysis — needs ~hours |
| Per-category accuracy breakdown | ⬜ Output of training |
| Connect with Chris on causal edges | ⬜ Discuss today |

---

## 7. If asked about the preprocessing results

Check how many videos were processed:
```bash
ls ~/data/mmau/processed/scene_graphs/ | wc -l
tail ~/preprocess.log
```

If preprocessing is still running: "It's running now, I'll have the full dataset processed by this afternoon."
If it finished: "We have N scene graphs. Next step is the analysis pass."

---

## Key numbers to know

| Fact | Value |
|------|-------|
| Videos total (cat 11 + 43) | 5,694 |
| Frames on disk | 567,840 JPEG |
| Frames per video sampled | 5 (every 3 seconds) |
| Node feature dim | 64 (class one-hot + bbox + time) |
| Learnable hyperedges K | 16 |
| HGNN hidden dim | 128 |
| Output classes | 58 (MM-AU categories) |
| Key prior work | Lohner et al. IAVVC 2024 (arxiv 2407.05910) |
