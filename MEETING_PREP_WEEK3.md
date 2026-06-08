# Week 3 Meeting — Liu's Talking Points

---

## What I did this week

**1. Kicked off category 43 preprocessing**
- Running overnight on the A6000. Should have ~1,400 new scene graph JSONs by end of week.
- Once done, switching `_make_label` in `dataset.py` to return cat-11 vs cat-43 binary label — a real accident-type distinction instead of the scene-complexity proxy.

**2. Set up ablation framework**
- `run_baseline.py` now supports `--baseline` flag (scalar-feature model) alongside the full `HGNNQAModel`.
- Both models train on identical data splits — ready to run as soon as cat 43 is processed.
- Three ablations queued:
  - A) HGNNQAModel vs scalar baseline — does the hypergraph add value?
  - B) Learned H vs frozen rule-based H — does learning the incidence matrix help?
  - C) Per-category accuracy breakdown — tests whether SeViLA's 89% is inflated by cat 11+43 dominance.

**3. Reviewed SeViLA and Lohner et al.**
- SeViLA: 89% on MM-AU full — but cats 11+43 are ~47% of the dataset, so a model biased toward those two dominates the metric. Our per-category breakdown will expose this.
- Lohner et al. (DoTA, 4-class, 57.77%): pairwise graph, rule-based edges, no zero-shot SGG. We beat them on all axes architecturally — just need the numbers to match.

---

## Training results recap (context for discussion)

| | Value |
|--|--|
| Task | Binary: complex vs simple scene (cat 11 only) |
| Test accuracy | **84.2%** (random baseline = 50%) |
| Best val accuracy | **91.4%** (epoch 9) |
| Loss | 0.44 → 0.04 |
| Overfitting onset | Epoch 6 (train 98.9% vs val 91.4%) |

Fix for overfitting: more data (cat 43) + dropout. Already in the plan.

---

## Discussion points I want input on

**1. TUM dataset**
Two options I see:
- Combine MM-AU + TUM and train one model — tests whether learnable hyperedges generalise across collection methods.
- Train two separate models and compare — cleaner ablation, shows domain gap.

Which direction makes more sense for the paper?

**2. Class balance / generalisation**
Idea: train on a subset of cat 11+43 plus all remaining categories, so the model doesn't over-index on dashcam/ego-car views. Trying to avoid the same trap SeViLA fell into.

---

## What I need from today's meeting

- Go/no-go on TUM integration (affects how I structure the dataset loader)
- Confirm ablation priority order — should learnt H vs fixed H come before or after cat-11 vs cat-43 retraining?
- Any new papers or directions from Prof. Cheng?
