# Week 3 Meeting — Liu's Talking Points
*Format: talk through sections in order, use numbers as anchors*

---

## 1. Quick recap — where we left off

Last week I finished the first full end-to-end training run. To summarise what the pipeline actually does before I get to the new stuff:

- **Input**: raw dashcam video frames (JPEG) from MM-AU
- **Step 1 — Zero-shot SGG**: Grounding DINO detects every object per frame with bounding boxes. LLaVA-1.5 reads each frame and outputs spatial relations between objects ("approaching", "cutting_off", etc.). Neither model saw traffic accident data — this is fully zero-shot.
- **Step 2 — Hypergraph construction**: Objects become nodes. We build two types of rule-based hyperedges (spatial proximity, temporal identity across frames) and then pass everything to a learnable MLP that produces a soft incidence matrix H. H[i,k] = how strongly node i belongs to hyperedge k.
- **Step 3 — HGNN**: Two rounds of spectral convolution (Feng et al. AAAI 2019 formula). Nodes aggregate features from all co-members of their hyperedges. Mean pool across nodes → linear head → prediction.
- **Step 4 — Label**: For the sanity check run, we used a binary scene-complexity proxy label (≥3 unique object classes = "complex", else "simple"). This was intentional — category 43 wasn't processed yet so we had only one accident type, which is useless for category classification.

---

## 2. Training results

| Metric | Value |
|--------|-------|
| Task | Binary: complex vs simple scene (cat 11 only) |
| Train / val / test split | 2,158 / 463 / 463 |
| Test accuracy | **84.2%** |
| Random baseline | 50% |
| Best val accuracy | **91.4%** (epoch 9) |
| Loss | 0.44 → 0.04 (dropped ~10×) |
| Overfitting onset | Epoch 6 (train 98.9% vs val 91.4%) |

The loss curve going from 0.44 to 0.04 matters — it means the model is learning real structure, not just memorising. The 7-point gap between train and val accuracy after epoch 6 is classic mild overfitting; the fix is more data, which is exactly what adding category 43 gives us.

At 84.2%, we're already in the same ballpark as SeViLA (89%) — but on a simpler task. Once we switch to real accident-type classification with more data, the comparison becomes meaningful.

---

## 3. What I did this week

**Category 43 preprocessing kicked off**
- The preprocessor already handles both categories and auto-skips videos that are done.
- I launched it on the A6000 this morning — it's running right now in the background.
- Estimated ~1,400 new scene graph JSONs by end of day (6-hour run).
- Once done, the dataset grows from 3,083 to ~4,500 samples, and I swap the label function from scene-complexity to cat-11 vs cat-43.

**Ablation framework is ready to run**
- `run_baseline.py` already supports a `--baseline` flag that switches to a scalar-feature model (no HGNN, just node counts fed into a linear classifier).
- Three ablations queued, in priority order:
  1. **HGNNQAModel vs scalar baseline** — does the hypergraph add value beyond a trivial feature?
  2. **Learned H vs frozen rule-based H** — freeze the hyperedge constructor and use only spatial/temporal edges. Isolates the contribution of learning the incidence matrix.
  3. **Per-category accuracy breakdown** — runs inference per accident category, not just aggregate accuracy.

**SeViLA and Lohner et al. review**
- SeViLA reports 89% on MM-AU full dataset, but categories 11 and 43 together make up ~47% of MM-AU. A model that does well on those two categories gets a very inflated overall number. Our per-category breakdown (ablation 3) will directly test this hypothesis.
- Lohner et al. (DoTA, 4-class, 57.77%): uses pairwise graphs with rule-based fixed edges and no zero-shot SGG. Our architecture beats theirs on every design axis — zero-shot SGG, hypergraph over pairwise, learned H over fixed H. We just need the numbers to catch up.

---

## 4. Discussion point 1 — TUM dataset integration

The TUM traffic dataset is collected differently from MM-AU (fixed infrastructure cameras vs dashcam/ego views). I see two directions and want input on which to pursue:

**Option A — Combine MM-AU + TUM, train one model**
- Tests whether learnable hyperedges generalise across camera perspectives and collection conditions.
- More training data, potentially better generalisation.
- Risk: the domain gap between dashcam and fixed-camera views could hurt rather than help if not handled carefully (separate normalisation, domain tags as input features, etc.).

**Option B — Train two separate models, compare**
- Cleaner ablation — shows the domain gap explicitly.
- Easier to debug if one dataset degrades performance.
- Could extend to a third run that combines both, making it a three-way comparison.

Either way, the dataset loader in `dataset.py` needs to be updated to support a second data source. I can do that once we agree on direction.

---

## 5. Discussion point 2 — class balance and generalisation

SeViLA's inflated 89% is a warning sign about class imbalance. I want to avoid the same problem. Proposed training setup:

- Use only a **subset** of category 11 and 43 (e.g. 30% of each), balanced with samples drawn equally from the remaining 56 categories.
- This forces the model to learn from diverse accident types and camera views rather than overfitting to the two dominant classes.
- Downside: less data for the main classes. May need to tune the balance ratio.

This could be a meaningful contribution in the paper — showing that our model generalises across categories while SeViLA doesn't, even if our aggregate accuracy is lower. Happy to hear if there's a preferred evaluation protocol for this.

---

## 6. Immediate next steps (this week)

1. **Wait for cat 43 preprocessing to finish** — check with `tail ~/preprocess43.log`
2. **Edit `_make_label` in `dataset.py`** — one-line change to return `(category_name, int(category==43))`
3. **Retrain on cat 11 vs cat 43** — same script, new label
4. **Run ablation A** — HGNN vs scalar baseline, same train/val/test split
5. **Run ablation B** — frozen H vs learned H

---

## 7. What I need from today's meeting

- Go/no-go on TUM integration, and if yes, Option A or B?
- Confirm ablation priority — run A and B before or after switching to cat-11 vs cat-43?
- Any guidance on evaluation protocol to avoid the SeViLA class-imbalance trap?
- Any new papers or directions from Prof. Cheng?
