# CLAUDE.md — SRP 2026 (Liu Restrepo)

## Who I am working with
**Liu** — Trinity College undergrad, SRP 2026 under **Prof. Cheng**. ML newbie, fast learner. First principles but concise. No assumed prior knowledge.

## Team
- **Prof. Cheng** — PI
- **Hyun Lee** — PhD coordinator, assigns tasks by email
- **Chris (Yi)** — other student, Project 1 (causal traffic prediction)

## Two Projects
### Project 1 — Traffic Congestion Prediction (Chris's lead)
- Datasets: PEMS, METR-LA. Approach: logic-diffusion + causal inference.
- Diffusion course: https://diffusion.csail.mit.edu/2026/index.html#lectures

### Project 2 — Scene/Accident Detection (Liu's lead)
- Dataset: MM-AU (CVPR 2024) — 11,730 dashcam videos, 58 accident categories
- Approach: zero-shot SGG (Grounding DINO + LLaVA-1.5) → situation hypergraph → HGNN
- Key refs: SHG-VQA arxiv 2304.08682, MM-AU arxiv 2403.00436, Lohner et al. arxiv 2407.05910

## Current State — End of Week 2 (2026-06-03)
Meetings: Mon/Wed/Fri 10am

### GPU Lab
- SSH: `trinity@<ip>`, **always use `conda activate srp26`**
- GPU 0: A6000 48GB ✅ | GPU 1: RTX PRO 6000 Blackwell ⚠️
- **Always run with**: `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python ...`
- Repo: `~/Liu/research-s26/` — always `git pull` before running

### Dataset
- Path: `~/data/mmau/CAP-DATA/CAP-DATA/{cat}/{cat}/{video_id}/images/*.jpg`
- Categories on disk: 11 (ego-car hitting car) + 43 (car hitting car), ~149GB
- **3,083 scene graph JSONs generated**: `~/data/mmau/processed/scene_graphs/`
- Analysis plots + report: `~/data/mmau/processed/analysis/`

## What's Built (all in mmau_adapter/)
| File | Status |
|------|--------|
| `scene_graph_gen.py` | ✅ Working. Uses `LlavaProcessor` (not LlavaNext — that breaks). |
| `preprocess.py` | ✅ Fixed. Scans disk via rglob, no metadata dependency. |
| `quick_demo.py` | ✅ Working. Tests SGG on 1 video. |
| `hypergraph.py` | ✅ Full. `build_hypergraph` (spatial+temporal only, NO rule-based collision), `LearnableHyperedgeConstructor`, `HypergraphConv` (Feng et al. AAAI 2019), `SituationHGNN`. |
| `dataset.py` | ✅ Fixed. Loads from JSONs directly, finds frames via rglob, binary label from scene complexity. |
| `run_baseline.py` | ✅ Working. `HGNNQAModel` (MobileNetV3+BERT+HGNN) default, `--baseline` for scalar model. |
| `demo_pipeline.py` | ✅ CPU-safe meeting demo. |
| `visualize.py` | ✅ Generates annotated GIF + hypergraph diagram from a JSON. |
| `analyze_scene_graphs.py` | ✅ Run, produced plots + report. |

## Training Results (Week 2)
- Task: binary scene complexity (simple vs complex, random baseline = 50%)
- Best val acc: **91.4%** (epoch 9) | Test acc: **84.2%**
- Loss: 0.44 → 0.04 over 10 epochs — model learned real structure
- Checkpoint: `runs/mmau_baseline/best_model.pt`
- Slight overfitting after epoch 6 — needs more data or regularisation

## Key Design Decisions
- **No rule-based collision edges**: bbox IoU in 2D is unreliable (perspective). Collision groupings are LEARNED by the HGNN from accident labels.
- **LLaVA "colliding_with" not trusted**: shown in orange in visualizer with (?) marker.
- **Learnable H**: `LearnableHyperedgeConstructor` MLP outputs soft incidence matrix. End-to-end differentiable.
- **Binary task for now**: only cat 11 processed. Switch to cat 11 vs 43 classification in Week 3.

## Next Steps (Week 3)
1. Preprocess category 43: `nohup env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python mmau_adapter/preprocess.py > ~/preprocess43.log 2>&1 &`
2. Retrain on cat 11 vs cat 43 (real 2-class task)
3. Ablation: `HGNNQAModel` vs `--baseline` (LightHGQA) — does HGNN beat scalar?
4. Ablation: learned H vs frozen rule-based H — does learning H help?

## Comparison Targets
| Model | Dataset | Accuracy |
|-------|---------|----------|
| SeViLA (NeurIPS 2023) | MM-AU | 89% (likely inflated by cat 11+43 dominance) |
| Lohner et al. IAVVC 2024 | DoTA (4 classes) | 57.77% |
| **Ours (binary, Week 2)** | **MM-AU cat 11** | **84.2%** |

## Relevant New Papers
- SoftHGNN (2025) arxiv 2505.15325 — concurrent, learnable soft hyperedges for vision
- SMA-Hyper (2024) arxiv 2407.17642 — hypergraph for traffic accident prediction
- "Language + vision for road safety" (2025) arxiv 2501.10604 — MLLM baseline

## Working Style
- Concise, actionable. Liu picks things up fast.
- Never commit for Liu — she commits/pushes herself so her GitHub looks active.
- Always `CUDA_VISIBLE_DEVICES=0` + `conda activate srp26` on GPU lab.
