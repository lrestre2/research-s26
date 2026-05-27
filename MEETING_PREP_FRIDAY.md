# Friday Meeting — Liu's Talking Points
## MM-AU Dataset Analysis (5–8 min)

> Show the plots on screen as you speak. Order: category_distribution → weather_conditions → temporal_timeline → hypergraph_motivation.

---

## 1. What is the MM-AU dataset?

> "MM-AU is a large-scale **ego-view** traffic accident video dataset, published as a CVPR 2024 Highlight.
> It contains **11,730 dashcam videos** with **2.23 million annotated bounding boxes** and
> **58,650 accident reason QA pairs** across **58 accident categories**.
>
> Each video carries three layers of annotation: the accident reason, a prevention measure,
> and a category label — all time-stamped to the moment of the crash.
> Seven object classes are annotated: cars, pedestrians, cyclists, trucks, buses,
> traffic lights, and motorbikes.
>
> The official split is 7 : 1.5 : 1.5 → roughly 8,200 train / 1,760 val / 1,760 test."

---

## 2. What the data actually looks like — three things I found

### 2a. The dataset is severely class-imbalanced
*(show category_distribution.png)*

> "Looking at the category distribution across all 58 accident types, two categories
> completely dominate: **category 11 has ~3,300 videos** and **category 43 has ~2,350** —
> together that's nearly **half the entire dataset**.
> Meanwhile most of the other 56 categories have fewer than 200 clips each, and many
> have under 50.
>
> This is a real problem for SOTA models. SeViLA reports 89% accuracy overall —
> but that number is almost certainly inflated by how well it performs on categories
> 11 and 43. For rare accident types — the ones that are actually hardest to handle
> in the real world — the true accuracy is likely much lower. We don't know by how much
> because the paper doesn't break accuracy down per category."

### 2b. The dataset barely covers adverse conditions
*(show weather_conditions.png)*

> "The weather and lighting distributions are almost binary.
> Weather code 1 — which corresponds to clear conditions — accounts for roughly
> **~10,000 of 11,730 videos, about 85% of the dataset**.
> Weather codes 2, 3, and 4 (rain, fog, snow) together make up the remaining 15%.
> Lighting is even more skewed: code 1 (daytime) is again ~90% of the dataset.
>
> The practical implication: any model trained on MM-AU will almost certainly fail
> in rain, fog, or nighttime — exactly the conditions where accidents are most dangerous
> and prediction matters most. This is a generalisation gap that the current SOTA
> completely ignores."

### 2c. The temporal annotation window is fixed, not real
*(show temporal_timeline.png)*

> "The temporal plots reveal something interesting about the annotation scheme.
> The accident duration distribution spikes at exactly 10 seconds for essentially
> every video — that's a fixed annotation window, not a measured event length.
> The pre-accident clip shows two clusters: a small group of very short clips (1–2s)
> and the rest piling up at the 30-second boundary, meaning most videos have
> more than 30 seconds of footage before the crash point.
>
> This matters for our approach: the 10-second accident window is where we want
> to build the hypergraph. We have a well-defined temporal segment to work with."

---

## 3. SOTA baselines — and their limits

> "Two tasks are benchmarked:
>
> **Object Detection (mAP50):**
> - YOLOv5s: 0.757 val / 0.748 test
> - DiffusionDet: 0.731 val / 0.733 test
> - Performance degrades significantly inside the accident window itself —
>   occlusion, motion blur, and rare multi-agent configurations break these models.
>
> **Accident Reason Answering (accuracy):**
> - SeViLA: 89.26% val / 89.02% test
> - CoVGT: ~80% val
>
> **The single biggest limitation across all of these:** every baseline treats each
> video as an independent stream from one camera. There is no modelling of
> *how objects relate to each other*, no reasoning about *who caused what*, and
> critically — **no multi-view setting at all**.
> The 89% headline number hides failures on rare, complex, multi-agent scenarios
> — which are exactly the cases we care about most."

---

## 4. How our approach addresses this
*(show hypergraph_motivation.png)*

> "The diagram shows the contrast. On the left, SOTA: each detected object is
> an independent node, frames are processed one at a time, nothing is connected.
>
> On the right, our proposal: a **situation hypergraph** where objects are nodes
> and interactions are hyperedges. A hyperedge is powerful because it can connect
> *more than two nodes at once* — so a 'collision event' can be a single hyperedge
> linking Car1, Cyclist, and Car2 simultaneously rather than needing pairwise edges
> between every pair.
>
> Three limitations we directly address:
>
> **1 — Relational structure.** Hyperedges capture multi-agent interactions that
> pairwise graphs and flat CNNs simply cannot express. A cyclist swerving *causes*
> the downstream collision — that causal chain is a hyperedge.
>
> **2 — Single-view blind spots.** Ego-view dashcams miss what's happening to the
> side, from behind, or from above. Adding an infrastructure camera or drone view
> gives us new nodes, and hyperedges can naturally connect the same physical event
> seen from different angles. That's the multi-view extension.
>
> **3 — Condition imbalance.** Because hypergraph structure is relational rather than
> appearance-based, it should generalise better to rain and nighttime — the model
> reasons about *who hit whom* rather than *what the scene looks like*."

---

## 5. Proposed direction

> "Concretely, here is what I want to do:
>
> 1. Start with the **SHG-VQA** codebase (CVPR 2023, github.com/aurooj/SHG-VQA)
>    which already implements situation hypergraph construction for Video QA.
> 2. Adapt it to MM-AU annotations — the format is compatible: we have object
>    bounding boxes, temporal windows, and QA pairs.
> 3. First experiment: **reproduce SHG-VQA on MM-AU in ego-view only** and evaluate
>    per-category accuracy to confirm the imbalance hypothesis.
> 4. Second experiment: augment with a simulated second view using the **CARLA
>    driving simulator**, which can render synchronised multi-camera scenes,
>    and extend the hyperedge construction across views.
>
> Baseline to beat: SeViLA at 89.02% overall — but more importantly,
> we want to improve on the *rare category* accuracy, which is the real challenge."

---

## Key numbers — know these cold

| Metric | Value |
|---|---|
| Total videos | 11,730 |
| Object boxes | 2.23M |
| QA pairs | 58,650 |
| Accident categories | 58 |
| Two dominant categories (11 + 43) | ~47% of all videos |
| Clear weather videos | ~85% |
| Daytime videos | ~90% |
| Best OD mAP50 | 0.757 (YOLOv5s) |
| Best ArA accuracy | 89.26% (SeViLA) |
| Dataset split | 7 : 1.5 : 1.5 |
| All baselines | Ego-view only |

---

## If Prof. Cheng asks about next steps

> "I want to run SHG-VQA on MM-AU as a baseline first — the GPU lab has a
> 96 GB RTX PRO 6000 which is more than enough. That gives us a concrete number
> to improve on. Then we decide whether the multi-view augmentation comes from
> CARLA simulation or from a separate real dataset."

## If asked about the class imbalance

> "I'd suggest we either subsample the dominant categories or use a weighted loss
> during training so the model can't coast on categories 11 and 43. Evaluating
> per-category accuracy will be the first thing I check."

## If asked about the diffusion course

> "I completed Lecture 3 of the MIT Diffusion Course.
> The forward process adds Gaussian noise step by step using a schedule β_t.
> The reverse process is a neural network that learns to undo that noise.
> The loss is the ELBO — it decomposes into a reconstruction term and KL divergence
> at each step, which in practice simplifies to predicting the noise ε directly.
> Connection to our work: the Graph Diffusion Model (NeurIPS 2024) applies this
> exact denoising framework on graph-structured data — directly relevant to Project 1."
