# CLAUDE.md — Summer Research Programme 2026

## Who I am working with
**Liu** (Restrepo, Liu) — Trinity College undergrad, SRP 2026 under **Prof. Cheng**. ML newbie, fast learner. Explain from first principles but stay concise. No assumed prior knowledge.

## Team
- **Prof. Cheng** — PI
- **Hyun Lee** — PhD/postdoc coordinator, sends task assignments via email
- **Chris (Yi, Christopher)** — other student, working on Project 1 (causal traffic prediction)

## Two Projects
### Project 1 — Traffic Congestion Prediction (Chris's lead)
- Datasets: PEMS, METR-LA
- Approach: logic-diffusion models + causal inference
- Refs: CausalGRIT, Dynamic Causal Graph CN, arxiv 2602.05549, IEEE 10422482, arxiv 2402.02518
- Diffusion course: MIT https://diffusion.csail.mit.edu/2026/index.html#lectures (team on Lecture 4+)

### Project 2 — Scene/Accident Detection (Liu's lead)
- Dataset: **MM-AU** (CVPR 2024 Highlight) — 11,730 ego-view dashcam accident videos
- Approach: zero-shot scene graph generation → situation hypergraph → HGNN for accident QA
- Key refs: SHG-VQA (CVPR 2023, arxiv 2304.08682), MM-AU paper (arxiv 2403.00436)

## Current State — Week 2 (2026-05-27)
Meetings: Mon/Wed/Fri 10am

### GPU Lab
- SSH: `trinity@<ip>`, conda env `srp26`
- GPU 0: RTX A6000 48GB sm_86 ✅ works with current PyTorch
- GPU 1: RTX PRO 6000 Blackwell 96GB sm_120 ❌ needs PyTorch nightly
- **BLOCKER**: PyTorch doesn't support Blackwell sm_120. Fix: `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128`. For now use `CUDA_VISIBLE_DEVICES=0`.
- Repo on GPU lab: `~/Liu/research-s26/` — always `git pull` before running

### Dataset on GPU Lab
- Path: `~/data/mmau/`
- Format: pre-extracted JPEGs (NOT video files)
- Structure: `~/data/mmau/CAP-DATA/CAP-DATA/{cat}/{cat}/{video_id}/images/*.jpg`
- Downloaded: categories 11 (ego-car hitting car, ~3300 videos) + 43 (car hitting car, ~2350 videos)
- Total: 567,840 frames, ~149GB
- Metadata: `~/data/mmau/video_metadata.json`
- **Known issue**: metadata `video_name` field does NOT match folder names on disk. Use `rglob("images")` to find frame dirs (as in `quick_demo.py`), NOT direct path lookup.

### GitHub Repo
- `github.com/lrestre2/research-s26`
- Mac: `/Users/Liu/research-s26/`
- Workflow: edit on Mac → `git push` → on GPU lab `git pull`

## What's Been Built (mmau_adapter/)
| File | Status | What it does |
|------|--------|------|
| `scene_graph_gen.py` | ✅ built, ❌ blocked by GPU | Grounding DINO (open-vocab detection) + LLaVA-1.5-7B (semantic relations). Zero-shot. |
| `preprocess.py` | ✅ built, ⚠️ find_frame_dir broken | Loads JPEGs, 5 frames/video, saves one JSON per video with scene graphs |
| `quick_demo.py` | ✅ built, ❌ blocked by GPU | Single-video demo using rglob. Use this first after fixing PyTorch. |
| `analyze_scene_graphs.py` | ✅ built, not yet run | Object/relation frequency, co-occurrence, collision signatures, outputs analysis_report.md |
| `hypergraph.py` | ✅ built | Node/Hyperedge schema + build_hypergraph() + LearnableHyperedgeConstructor (nn.Module) |
| `dataset.py` | ✅ built | PyTorch Dataset class bridging MM-AU → SHG-VQA format |
| `run_baseline.py` | ✅ built, not yet run | Training + per-category accuracy. Got 0 samples last run (preprocess hadn't run yet). |

## Hyun's Tasks (assigned Wed 2026-05-21, status as of 2026-05-27)
1. ✅ Zero-shot SGG — built (Grounding DINO + LLaVA)
2. ✅ JSON format — one file/video, scene graphs per 3s frame
3. ⬜ Analyse scene graphs → gold standard for hyperedge design (blocked on GPU fix)
4. ✅ Learnable hyperedges — LearnableHyperedgeConstructor in hypergraph.py

## Immediate Next Steps
1. Fix PyTorch for Blackwell: `pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128`
2. Run: `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python mmau_adapter/quick_demo.py`
3. Fix `find_frame_dir` in preprocess.py to use rglob (like quick_demo.py does)
4. Run full preprocess: `PYTHONPATH=. python mmau_adapter/preprocess.py`
5. Run analysis: `PYTHONPATH=. python mmau_adapter/analyze_scene_graphs.py`
6. Copy `~/data/mmau/processed/analysis/` to Mac, review analysis_report.md
7. Bring analysis findings to Friday meeting — this is the "gold standard" discussion

## Key Design Decisions (for context)
- **Why zero-shot**: avoid needing traffic-specific training data for SGG
- **Why 5 frames/3s**: Hyun's spec from Wednesday meeting
- **Why learnable hyperedges**: Hyun said "learnable sounds better than static" — don't hardcode rules
- **Hyperedge types**: spatial (proximity), temporal (same object across frames), collision (bbox overlap, different classes), multiview (future — CARLA simulation)
- **Category 11** = ego-car hitting car (~3300 videos), **Category 43** = car hitting car (~2350 videos)
- These two = ~47% of dataset. SeViLA's 89% accuracy likely inflated by these dominant classes.

## Working Style
- Concise, actionable. Liu picks things up fast.
- Always `git push` after changes so GPU lab can `git pull`.
- Prefer fixing and running over long explanations.
