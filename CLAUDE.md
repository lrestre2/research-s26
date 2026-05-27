# CLAUDE.md — Summer Research Programme 2026

## Who I am working with
**Liu** (Restrepo, Liu) — undergraduate student at Trinity College, working in the Summer Research Programme 2026 under **Prof. Cheng**. Liu is new to ML/AI and hypergraphs, but learns and adapts very fast. Frame explanations from first principles; no assumed prior knowledge of graph theory or diffusion models.

## The Research Programme
10-week SRP with Prof. Cheng on **two interconnected projects**, both traffic-domain:

### Project 1 — Traffic Congestion Prediction
- Use graph/diffusion-based models on traffic sensor datasets (PEMS, METR-LA)
- Causal inference is a key angle: CausalGRIT and Dynamic Causal Graph CN are reference papers
- Logic-diffusion modelling is the framework being developed
- Key reference papers: [arxiv 2602.05549](https://arxiv.org/pdf/2602.05549), [IEEE 10422482](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10422482), [arxiv 2402.02518](https://arxiv.org/pdf/2402.02518)

### Project 2 — Scene/Accident Detection (Video QA)
- Dataset: **MMAU** (from LOTVS-MMAU) — multi-view traffic accident understanding
- Approach: hypergraph learning, multi-view settings preferred over ego-view only
- Key reference: *Learning Situation Hyper-Graphs for Video Question Answering* (CVPR 2023)

## Team
- **Prof. Cheng** — PI
- **Hyun Lee** — likely a PhD student/postdoc coordinating the team, sends task assignments
- **Christopher Yi (Chris)** — the other student, working on causal algorithm analysis for Project 1

## Current State (Week 1, as of 2026-05-22)
- Diffusion model study: MIT course at https://diffusion.csail.mit.edu/2026/index.html#lectures
  - Team is on Lecture 3 + Lab 1
  - Liu still needs to complete lectures and submit notes to Hyun Lee
- **Meeting: Friday 2026-05-23 at 10am** — Liu must present MMAU dataset analysis

## Liu's Immediate Tasks (due Friday 2026-05-23)
1. Download MMAU dataset (LOTVS-MMAU GitHub)
2. Review dataset + associated papers
3. Present: SOTA limitations + how graph-learning addresses them
4. Explore multi-view settings (not just ego-view)
5. Complete MIT Diffusion Lecture 3, send notes to Hyun Lee

## Compute Resources
- **Local**: MacBook Air 2020 M1 — fine for reading, light prototyping, data exploration
- **GPU Lab** (SSH): 
  - GPU 0: NVIDIA RTX A6000 — 48 GB VRAM, 300W TDP
  - GPU 1: NVIDIA RTX PRO 6000 Black — 96 GB VRAM, 600W TDP
  - CUDA 12.8, Driver 570.211.01
  - Liu has never used a GPU lab before

## Key Datasets
- **PEMS** (traffic sensors, Project 1): https://www.kaggle.com/datasets/elmahy/pems-dataset
- **MMAU** (accident video QA, Project 2): LOTVS-MMAU GitHub

## Working Style Notes
- Liu learns fast — give concise, actionable guidance; don't over-explain basics once understood
- Liu is excited and motivated; keep momentum with clear next steps
- Prefer practical action plans over theoretical deep-dives until context demands it
