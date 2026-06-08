# Context — SRP 2026 (Liu Restrepo)
*Last updated: 2026-06-03 (end of Week 2)*

---

## Status Summary

| Item | State |
|------|-------|
| GPU fix | ✅ Use `conda activate srp26` + `CUDA_VISIBLE_DEVICES=0` |
| Preprocessing | ✅ 3,083 scene graph JSONs in `~/data/mmau/processed/scene_graphs/` |
| Analysis | ✅ Plots + report in `~/data/mmau/processed/analysis/` |
| Training | ✅ 84.2% test accuracy (binary task, cat 11 only) |
| Category 43 | ⬜ Not yet preprocessed — do this Week 3 |
| Ablation study | ⬜ Week 3 |

---

## What Was Built

| File | What it does |
|------|-------------|
| `mmau_adapter/scene_graph_gen.py` | Grounding DINO + LLaVA-1.5 zero-shot SGG. Uses `LlavaProcessor` (not LlavaNext). |
| `mmau_adapter/preprocess.py` | Processes all videos via rglob scan. No metadata dependency. |
| `mmau_adapter/quick_demo.py` | Single-video SGG test. Run this to verify GPU works. |
| `mmau_adapter/hypergraph.py` | `build_hypergraph` (spatial+temporal), `HypergraphConv`, `SituationHGNN`, `LearnableHyperedgeConstructor`. No rule-based collision edges. |
| `mmau_adapter/dataset.py` | Loads from JSONs + real frames. Binary label: complex vs simple scene. |
| `mmau_adapter/run_baseline.py` | `HGNNQAModel` (default) or `--baseline`. Uses `label` field (0/1). |
| `mmau_adapter/demo_pipeline.py` | CPU-safe 4-step demo for meetings. |
| `mmau_adapter/visualize.py` | Annotated GIF + hypergraph diagram from any JSON. |
| `mmau_adapter/analyze_scene_graphs.py` | Object/relation stats + plots. Already run. |

---

## Training Results (Week 2 — first run)

```
Task   : binary (simple_scene vs complex_scene)
Data   : 2,158 train / 463 val / 463 test  (cat 11 only)
Best   : val acc 91.4% at epoch 9
Test   : 84.2%   (random baseline = 50%)
Loss   : 0.44 → 0.04
Status : slight overfitting after epoch 6
Saved  : runs/mmau_baseline/best_model.pt
```

---

## Week 3 To-Do (in order)

1. **Preprocess category 43**
```bash
conda activate srp26
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python mmau_adapter/preprocess.py > ~/preprocess43.log 2>&1 &
```

2. **Retrain on cat 11 vs cat 43** (real 2-class problem)
   - Edit `_make_label` in `dataset.py` to return `(category_name, int(category==43))`

3. **Ablation A** — HGNN vs baseline
```bash
# HGNN (already done)
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python mmau_adapter/run_baseline.py --epochs 20 > ~/hgnn.log 2>&1 &
# Scalar baseline
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python mmau_adapter/run_baseline.py --epochs 20 --baseline > ~/baseline.log 2>&1 &
```

4. **Ablation B** — learned H vs frozen H
   - In `SituationHGNN.forward()`, replace `H = self.hyperedge_ctor(node_features)` with `H = build_static_H(node_features)` using rule-based spatial/temporal only

5. **Copy results to Mac**
```bash
scp -r trinity@<ip>:~/Liu/research-s26/runs/ ~/Downloads/mmau_runs/
```

---

## Key Design Decisions

- **No rule-based collision edges** — bbox IoU is wrong in 2D (perspective). Collision groupings are learned by HGNN from labels.
- **LLaVA relations are noisy** — shown in orange with (?) in visualizer. Don't treat as ground truth.
- **Learnable H** — `LearnableHyperedgeConstructor` MLP outputs soft incidence matrix [N×K]. Backprop flows through it.
- **HGNN formula**: `X' = σ( Dv^{-½} H W De^{-1} H^T Dv^{-½} X Θ )` — implemented in `HypergraphConv`.

---

## Comparison Targets for Paper

| Model | Task | Acc |
|-------|------|-----|
| SeViLA (NeurIPS 2023) | MM-AU full | ~89% (likely inflated) |
| Lohner et al. IAVVC 2024 | DoTA 4-class | 57.77% |
| **Ours Week 2** | MM-AU binary | **84.2%** |

---

## Papers to Read

| Priority | Paper | Why |
|----------|-------|-----|
| 🔴 Now | 3Blue1Brown Neural Networks (YouTube) | Understand backprop |
| 🔴 Now | Distill.pub GNN intro | Understand message passing |
| 🔴 Now | Feng et al. AAAI 2019 arxiv 1809.09401 | The exact HGNN your code uses |
| 🟡 Soon | SHG-VQA arxiv 2304.08682 | Your architecture reference |
| 🟡 Soon | Lohner et al. arxiv 2407.05910 | Your main baseline |
| 🟡 Soon | SoftHGNN arxiv 2505.15325 | Concurrent work, same idea |
| 🟢 Later | MM-AU arxiv 2403.00436 | Your dataset |
| 🟢 Later | SeViLA arxiv 2305.06988 | Comparison target |

---

## GPU Lab Quick Reference

```bash
# Always start with:
conda activate srp26
cd ~/Liu/research-s26
git pull

# Run anything:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python mmau_adapter/<script>.py

# Check training:
tail -f ~/training.log

# Copy results to Mac (run on Mac):
scp -r trinity@<ip>:~/Liu/research-s26/runs/ ~/Downloads/mmau_runs/
```

---

## GitHub
- Repo: `https://github.com/lrestre2/research-s26`
- Liu commits and pushes herself (keeps her GitHub active)
- Mac path: `/Users/Liu/research-s26/`
