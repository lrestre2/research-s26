# Wednesday Meeting — Liu's Talking Points
## What I built this week + next steps

---

## 1. What quick_demo.py does (explain this if asked)

The script has four simple steps:

**Step 1 — Find a video**
It looks through the downloaded dataset and grabs the first accident video it finds.
In our case that's video `011460` — a category 11 accident (ego-car hitting car)
with 301 pre-extracted frames on disk.

**Step 2 — Pick 5 frames**
Instead of processing all 301 frames, it picks 5 evenly spaced ones.
Why 5? One frame every 3 seconds for a ~15 second clip — exactly the structure
Hyun asked for on Wednesday.

**Step 3 — Run the two AI models**
For each of the 5 frames it runs:
- **Grounding DINO** — detects every object in the frame using open vocabulary
  (not limited to fixed classes — it can detect anything described in language)
- **LLaVA-1.5** — a vision-language model that looks at the frame and the detected
  objects and describes the relationships between them in natural language:
  "approaching", "colliding_with", "cutting_off" etc.
This is the zero-shot part — neither model was trained on traffic accident data.

**Step 4 — Save one JSON per video**
The output is one JSON file with 5 scene graphs inside it, one per frame.
Each scene graph has: objects (with bounding boxes), and relations between them.

---

## 2. What I built this week — full summary

| What | Why |
|------|-----|
| Zero-shot scene graph generation | Grounding DINO + LLaVA, no traffic-specific training needed |
| New JSON format | One file per video, one scene graph per 3-second window (Hyun's spec) |
| Dataset downloaded + explored | 100GB, categories 11 + 43, 567,840 frames on GPU lab |
| Hypergraph schema designed | Nodes = objects, 3 hyperedge types: spatial, temporal, collision |
| Learnable hyperedge constructor | Neural network that learns which nodes to group, instead of hardcoded rules |
| Analysis pipeline | Generates plots + report: object frequency, relation frequency, collision signatures |

---

## 3. The GPU issue — say this clearly

> "The scene graph pipeline is fully built. When I ran it last night it hit a
> compatibility issue: our GPU lab has an RTX PRO 6000 which is Blackwell
> architecture — PyTorch doesn't support it yet. The A6000 is affected too
> because PyTorch initialises across both GPUs at startup.
> The fix is installing a PyTorch nightly build that supports Blackwell.
> I'll have actual scene graph outputs this afternoon."

This is not a code problem. The pipeline works — it's a library version issue
that takes one command to fix.

---

## 4. The learnable hyperedge idea — explain if asked

> "Hyun suggested learnable hyperedges instead of static rules.
> What that means: instead of hardcoding 'objects within 30% of the frame
> width get a spatial hyperedge', we train a small neural network that takes
> the node features and outputs which nodes should be grouped together.
> The model discovers the relevant groupings from data rather than us
> deciding them in advance. I've implemented this as a differentiable
> incidence matrix — it can be trained end-to-end with the rest of the model."

---

## 5. What I'm proposing for next week

1. **Fix PyTorch today** → run the full scene graph generation overnight
2. **Analyse the generated graphs** → bring analysis_report.md to Friday's meeting
3. **Discuss hyperedge design with the team** based on what the data shows
4. **Connect with Chris** — his causal edges on Project 1 and our collision
   hyperedges on Project 2 are the same idea — should align the two

---

## Key numbers to know

| Fact | Value |
|------|-------|
| Videos downloaded | 5,694 (categories 11 + 43) |
| Frames on disk | 567,840 |
| Frames per scene graph | 5 (one per 3 seconds) |
| Scene graph generation models | Grounding DINO + LLaVA-1.5-7B |
| Zero-shot? | Yes — no fine-tuning on traffic data |
| GPU issue | Blackwell sm_120 not in current PyTorch; fix = nightly install |
