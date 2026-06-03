# Context — SRP 2026 (Liu Restrepo)
*Last updated: 2026-05-28 (Week 2 — meeting prep)*

---

## What this file is for
This is YOUR context file — a plain-English record of what has been done, what is broken, and what needs doing next. Update it when things change. Claude reads `CLAUDE.md` for its own memory; this one is for you.

---

## The Two Projects

### Project 1 — Traffic Congestion Prediction
- **Goal**: predict traffic congestion using causal + diffusion-based graph models
- **Dataset**: PEMS (traffic sensors), METR-LA
- **Your role**: learning/support. Chris is doing the main work here.
- **Key references**: CausalGRIT, Dynamic Causal Graph CN, Logic-Diffusion paper
- **Status**: Not touched yet — Chris is handling Algorithm Analysis + Implementation Strategy

### Project 2 — Scene/Accident Detection (Video QA)
- **Goal**: understand traffic accidents from dashcam video using hypergraph learning
- **Dataset**: MM-AU (CVPR 2024) — 11,730 ego-view dashcam videos, 58 accident categories
- **Your role**: This is YOUR project. Full ownership.
- **Key reference**: SHG-VQA (CVPR 2023) — "Learning Situation Hyper-Graphs for Video Question Answering"
- **Status**: Pipeline built. Blocked on one GPU fix. See below.

---

## What Was Built (Week 1–2)

| Component | File | What it does | Status |
|-----------|------|-------------|--------|
| Scene graph generator | `mmau_adapter/scene_graph_gen.py` | Grounding DINO detects objects; LLaVA-1.5 describes relations between them. Zero-shot — no traffic-specific training. | ✅ Built |
| Quick demo | `mmau_adapter/quick_demo.py` | Runs the pipeline on 1 video (5 frames) to prove it works. The safe script to test first. | ✅ Built, needs GPU fix |
| Full preprocessor | `mmau_adapter/preprocess.py` | Runs scene graph generation on all 5,694 videos. Saves one JSON per video. **Path bug fixed** — now scans disk directly, no metadata dependency. | ✅ Built, needs GPU fix |
| Scene graph analyser | `mmau_adapter/analyze_scene_graphs.py` | Loads all JSONs, makes plots, writes `analysis_report.md`. Run this after preprocessing. | ✅ Built |
| Hypergraph schema + HGNN | `mmau_adapter/hypergraph.py` | Nodes = detected objects. Rule-based hyperedges (spatial/temporal/collision) + **HypergraphConv layer** (spectral HGNN, Feng et al. 2019) + **SituationHGNN** (2-layer HGNN + classifier, end-to-end differentiable). | ✅ Built |
| Dataset loader | `mmau_adapter/dataset.py` | Wraps scene graph JSONs as a PyTorch Dataset (train/val/test split). | ✅ Built |
| Baseline model | `mmau_adapter/run_baseline.py` | **HGNNQAModel** (MobileNetV3 + BERT + SituationHGNN) as default. LightHGQA (scalar node-count) with `--baseline` flag. | ✅ Built |
| **Meeting demo** | `mmau_adapter/demo_pipeline.py` | **NEW.** Runs CPU-only. Shows full pipeline: scene graph → rule-based HG → learnable H → HGNN. Uses synthetic data if no JSONs preprocessed yet. | ✅ Built |
| Full pipeline script | `scripts/run_pipeline.sh` | Runs everything in order: setup → check data → preprocess → analyse → train. | ✅ Built |

---

## The Data

**Location on GPU lab:**
```
~/data/mmau/
├── CAP-DATA/CAP-DATA/          ← actual accident videos (pre-extracted as JPEGs)
│   ├── 11/11/011460/images/    ← example: category 11, video 011460
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...  (301 frames)
│   └── 43/43/.../images/
├── video_metadata.json         ← metadata for all videos
└── processed/
    └── scene_graphs/           ← OUTPUT goes here (currently empty)
```

- **Downloaded**: ~149 GB, categories 11 + 43 only
- **Frames on disk**: 567,840 JPEGs
- **Videos**: 5,694
- **Format**: 10 fps pre-extracted JPEGs (no video decoding needed)

---

## What is Broken / Blocked Right Now

### 🔴 Blocker 1: PyTorch doesn't support the RTX PRO 6000 (Blackwell GPU)

**What happens**: Any PyTorch CUDA code crashes with:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Why**: The RTX PRO 6000 is "Blackwell" architecture (sm_120). Current PyTorch only supports up to sm_90. Even with `CUDA_VISIBLE_DEVICES=0` (A6000), PyTorch still detects the Blackwell GPU at startup and crashes.

**Fix** (one command, run on GPU lab):
```bash
conda activate srp26
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

This installs PyTorch Nightly which supports Blackwell. Takes ~5 minutes.

### ✅ Bug 2 FIXED: preprocess.py now scans disk directly

`find_frame_dir()` is gone. The new `_scan_all_image_dirs()` function does a single `rglob("images")` pass at startup and iterates over found directories. Metadata is looked up by folder name (best-effort) rather than the other way around. Will no longer print "Not found: 5694".

---

## Immediate Next Steps (In Order)

### Step 1 — Fix PyTorch on the GPU lab ← DO THIS FIRST
```bash
ssh trinity@<gpu-lab-ip>
conda activate srp26
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Step 2 — Test with quick_demo.py
```bash
cd ~/Liu/research-s26
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python mmau_adapter/quick_demo.py
```
This should run Grounding DINO + LLaVA on 5 frames of video `011460` and save a JSON. If this works, the pipeline is proven.

### Step 3 — ✅ DONE: preprocess.py path bug fixed (see above)

### Step 4 — Run full preprocessing overnight
```bash
nohup bash scripts/run_pipeline.sh > ~/pipeline.log 2>&1 &
tail -f ~/pipeline.log
```
This runs for ~hours on all 5,694 videos. Let it run overnight.

### Step 5 — Analyse the scene graphs
```bash
PYTHONPATH=. python mmau_adapter/analyze_scene_graphs.py
```
Produces `~/data/mmau/processed/analysis/analysis_report.md` + 4 plots.

### Step 6 — Copy results to your Mac, review
```bash
# On your Mac:
scp -r trinity@<ip>:~/data/mmau/processed/analysis/ ~/Downloads/mmau_analysis/
```

### Step 7 — Bring analysis to Friday meeting
The analysis report tells you: which object types appear most in accidents, which relations are most common, what the collision "signature" looks like. Use that to justify the hyperedge design.

---

## How the Pipeline Works (Plain English)

```
Raw videos (JPEGs on disk)
        ↓
  preprocess.py
  For each video → pick 5 evenly-spaced frames
  For each frame:
    1. Grounding DINO → "I see: car, cyclist, truck at these bounding boxes"
    2. LLaVA-1.5 → "car is approaching cyclist, truck is stopped_in_front_of car"
  Saves one JSON per video with 5 scene graphs inside
        ↓
  analyze_scene_graphs.py
  Loads all JSONs → counts, plots, recommendations
        ↓
  hypergraph.py / run_baseline.py
  Converts scene graphs → hypergraphs → trains accident classifier
```

---

## For Thursday's Meeting

**Run this on the GPU lab** (after GPU fix):
```bash
conda activate srp26
PYTHONPATH=. python mmau_adapter/demo_pipeline.py   # uses real data if available
```

**Or run this on your Mac right now** (CPU, no data needed — for rehearsal):
```bash
PYTHONPATH=. python3 mmau_adapter/demo_pipeline.py --synthetic
```

**Key paper to cite**: Lohner et al., "Enhancing Vision-Language Models with Scene Graphs for Traffic Accident Understanding", IEEE IAVVC 2024 (Best Paper Runner-Up). arxiv 2407.05910.
- They get 57.77% on DoTA (4 classes), pairwise graphs, rule-based SGG
- We extend: hypergraph, learned H, zero-shot SGG, MM-AU (58 classes)

**HGNN formulation to know**:
```
X' = σ( Dv^{-1/2}  H  W  De^{-1}  H^T  Dv^{-1/2}  X  Θ )
```
- H [N×K]: incidence matrix, learned by an MLP (not hard-coded)
- Dv, De: diagonal degree matrices (normalisation)
- Θ: linear projection (the "weights" of the layer)
- With soft H from backprop, entire pipeline is end-to-end differentiable

---

## Key Design Decisions (for meetings)

**Why zero-shot?** — Neither Grounding DINO nor LLaVA was trained on traffic accidents, yet they generalise because they understand general visual concepts. This is the contribution: no labelled traffic data needed for scene graph generation.

**Why hypergraphs instead of regular graphs?** — A regular graph edge connects exactly 2 nodes. A hyperedge connects ANY number of nodes — e.g. "car + cyclist + truck are all in the same near-collision cluster". This captures multi-body interactions that pairwise edges miss.

**Why learnable hyperedges?** — Instead of hardcoding "objects within 30% of frame width get a spatial hyperedge", a small neural network learns which objects to group together from data. Hyun specifically asked for this.

**Connection to Project 1** — Chris is building causal edges for traffic prediction. Our collision hyperedges are the same idea applied to accident videos. Worth aligning with Chris.

---

## Team & Meetings

| Person | Role | How they relate to your work |
|--------|------|------------------------------|
| Prof. Cheng | PI | Final authority. Be able to explain everything simply. |
| Hyun Lee | PhD coordinator | Sends tasks, reviews outputs. His format specs are in the code. |
| Chris (Yi) | Other student | Project 1 (congestion). His causal edges ↔ your collision hyperedges. |

**Meeting schedule**: Mon/Wed/Fri 10am, ~30–60 min each.

---

## Diffusion Course (Don't Forget)

- MIT course: https://diffusion.csail.mit.edu/2026/index.html#lectures
- Team is on **Lecture 3 + Lab 1**
- You need to **send Lecture 3 notes to Hyun** — this is overdue
- Do 1 lecture per day, send notes immediately after

---

## Git Workflow

Your code lives at: `https://github.com/lrestre2/research-s26`

```bash
# On Mac — make changes, push
cd ~/Liu/research-s26
git add mmau_adapter/some_file.py
git commit -m "fix: description of what you changed"
git push

# On GPU lab — pull and run
cd ~/Liu/research-s26
git pull
conda activate srp26
PYTHONPATH=. python mmau_adapter/quick_demo.py
```

---

## Files You Should Know About

```
research-s26/
├── CLAUDE.md                        ← Claude's compressed memory (don't edit manually)
├── CONTEXT.md                       ← This file (YOUR notes)
├── MEETING_PREP_WEDNESDAY.md        ← Talking points for Wednesday meeting
├── MEETING_PREP_FRIDAY.md           ← Talking points for Friday meeting
├── HYPERGRAPH_SCHEMA.md             ← Full design doc for the hypergraph
├── mmau_adapter/
│   ├── scene_graph_gen.py           ← Grounding DINO + LLaVA zero-shot SGG
│   ├── quick_demo.py                ← Test script: 1 video, 5 frames
│   ├── preprocess.py                ← Full preprocessing: all 5,694 videos
│   ├── analyze_scene_graphs.py      ← Analysis + plots + report
│   ├── hypergraph.py                ← Hypergraph + learnable constructor
│   ├── dataset.py                   ← PyTorch Dataset wrapper
│   └── run_baseline.py              ← Baseline classifier model
└── scripts/
    ├── run_pipeline.sh              ← Master pipeline (run overnight)
    └── inspect_hf_repo.py           ← HuggingFace repo explorer (one-off tool)
```

## Training results
```

(srp26)  🐍 srp26  trinity@trinity-gpulab  ~/Liu/research-s26  ↱ main  tail -f ~/training.log

cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
cls.seq_relationship.bias                  | UNEXPECTED |  | 
cls.predictions.bias                       | UNEXPECTED |  | 
cls.seq_relationship.weight                | UNEXPECTED |  | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Epoch 1/10: 100%|██████████| 270/270 [00:16<00:00, 16.45it/s]
Epoch 01 | loss 0.4385 | train acc 0.820 | val acc 0.842
  ↑ New best: 0.842
Epoch 2/10: 100%|██████████| 270/270 [00:15<00:00, 17.61it/s]
Epoch 02 | loss 0.3251 | train acc 0.858 | val acc 0.864
  ↑ New best: 0.864
Epoch 3/10: 100%|██████████| 270/270 [00:15<00:00, 17.33it/s]
Epoch 03 | loss 0.2290 | train acc 0.901 | val acc 0.877
  ↑ New best: 0.877
Epoch 4/10: 100%|██████████| 270/270 [00:15<00:00, 17.18it/s]
Epoch 04 | loss 0.1686 | train acc 0.931 | val acc 0.875
Epoch 5/10: 100%|██████████| 270/270 [00:15<00:00, 17.40it/s]
Epoch 05 | loss 0.1255 | train acc 0.951 | val acc 0.883
  ↑ New best: 0.883
Epoch 6/10: 100%|██████████| 270/270 [00:15<00:00, 17.73it/s]
Epoch 06 | loss 0.0798 | train acc 0.972 | val acc 0.890
  ↑ New best: 0.890
Epoch 7/10: 100%|██████████| 270/270 [00:14<00:00, 18.06it/s]
Epoch 07 | loss 0.0681 | train acc 0.975 | val acc 0.894
  ↑ New best: 0.894
Epoch 8/10: 100%|██████████| 270/270 [00:15<00:00, 17.24it/s]
Epoch 08 | loss 0.0616 | train acc 0.978 | val acc 0.870
Epoch 9/10: 100%|██████████| 270/270 [00:15<00:00, 17.11it/s]
Epoch 09 | loss 0.0405 | train acc 0.989 | val acc 0.914
  ↑ New best: 0.914
Epoch 10/10: 100%|██████████| 270/270 [00:15<00:00, 17.68it/s]
Epoch 10 | loss 0.0457 | train acc 0.985 | val acc 0.862

=== Test Set Evaluation ===
Test accuracy: 0.8423 (84.23%)                       
(Task: simple_scene vs complex_scene — random baseline = 50%)

Results saved → /home/trinity/Liu/research-s26/runs/mmau_baseline/results.json
[1]  + 2169398 done       nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python  --epochs 10 --batch 8 >
```