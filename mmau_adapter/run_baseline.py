"""
SHG-VQA Baseline on MM-AU
==========================
Trains the SHG-VQA model on MM-AU (categories 11 + 43) and evaluates it,
reporting both overall accuracy and per-category accuracy.

The per-category breakdown is the key output — it lets us confirm the
class-imbalance hypothesis: that the ~89% SeViLA headline number hides
poor performance on rare accident types.

Usage:
    conda activate srp26
    cd ~/Liu/research-s26
    python mmau_adapter/run_baseline.py [--epochs 10] [--batch 8] [--eval-only]

Outputs:
    runs/mmau_baseline/
    ├── best_model.pt           — best checkpoint by val accuracy
    ├── training_log.json       — loss + accuracy per epoch
    └── per_category_accuracy.json — accuracy broken down by accident type
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Make sure the repo root is on the path so mmau_adapter is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# Add SHG-VQA to path
RESEARCH_DIR = Path(__file__).resolve().parent.parent
SHGVQA_DIR   = RESEARCH_DIR / "SHG-VQA" / "AGQA"
sys.path.insert(0, str(SHGVQA_DIR))

from mmau_adapter.dataset import MMAUDataset, collate_fn
from mmau_adapter.hypergraph import (
    build_hypergraph,
    SituationHGNN,
    node_features_from_sg,
)

# ── config ─────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
RUN_DIR    = RESEARCH_DIR / "runs" / "mmau_baseline"
CATEGORIES = [11, 43]    # downloaded categories


# ── simple baseline model ───────────────────────────────────────────────
# Full SHG-VQA requires pre-extracted SlowFast features which take days
# to compute. This lightweight stand-in lets us get a real accuracy number
# fast using the same frame tensors + BERT question encoder + scene graph
# node features. Swap in the full SHG-VQA model later.

class LightHGQA(nn.Module):
    """
    Lightweight Video QA model that uses:
      - Mean-pooled frame features (from a frozen ResNet)
      - BERT-encoded question
      - Scene graph node counts as auxiliary features
      - Concatenation → 2-layer MLP → answer logits

    This is a fast baseline, NOT the full SHG-VQA hypergraph model.
    The output of this script is the per-category accuracy number we
    want to confirm the imbalance hypothesis.
    """

    def __init__(self, n_answers: int, hidden: int = 512):
        super().__init__()
        # Visual encoder: lightweight MobileNetV3 to avoid OOM
        import torchvision.models as M
        backbone  = M.mobilenet_v3_small(weights=M.MobileNet_V3_Small_Weights.DEFAULT)
        self.vis  = nn.Sequential(*list(backbone.children())[:-1])  # remove classifier
        vis_dim   = 576

        # Question encoder
        self.bert      = BertModel.from_pretrained("bert-base-uncased")
        self.bert_proj = nn.Linear(768, hidden)

        # Visual projection
        self.vis_proj = nn.Linear(vis_dim, hidden)

        # Scene graph: just use node count as a scalar feature for now
        self.sg_proj = nn.Linear(1, hidden // 8)

        # Classifier
        clf_in = hidden + hidden + hidden // 8
        self.classifier = nn.Sequential(
            nn.Linear(clf_in, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_answers),
        )

    def forward(self, frames, input_ids, attention_mask, sg_node_counts):
        B, T, C, H, W = frames.shape

        # Visual: encode each frame, mean-pool over time
        frames_flat = frames.view(B * T, C, H, W)
        vis_feat    = self.vis(frames_flat)           # [B*T, vis_dim, 1, 1]
        vis_feat    = vis_feat.view(B, T, -1).mean(1) # [B, vis_dim]
        vis_feat    = self.vis_proj(vis_feat)          # [B, hidden]

        # Question: [CLS] token from BERT
        q_out  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        q_feat = self.bert_proj(q_out.last_hidden_state[:, 0])   # [B, hidden]

        # Scene graph: node count scalar
        sg_feat = self.sg_proj(sg_node_counts.unsqueeze(1).float())  # [B, hidden//8]

        # Classify
        x = torch.cat([vis_feat, q_feat, sg_feat], dim=-1)
        return self.classifier(x)


# ── upgraded model: multimodal HGNN ─────────────────────────────────────
#
# HGNNQAModel replaces LightHGQA as the main model.
# The key difference: instead of a scalar node-count feature, we pass all
# node features through SituationHGNN, which does two rounds of spectral
# hypergraph convolution over a *learned* incidence matrix H.
#
# Concretely, for each sample in the batch we:
#   1. Build node feature matrix X from the scene graph (CPU, fast).
#   2. Run SituationHGNN(X):
#        - LearnableHyperedgeConstructor → H [N, K]
#        - HypergraphConv × 2           → [N, hidden]
#        - mean pool                    → [hidden]
#   3. Concatenate with visual (MobileNetV3) and text (BERT) features.
#   4. Classify with a 2-layer MLP.
#
# The HGNN introduces ~500k additional parameters on top of LightHGQA
# but provides a richer, structured representation of the accident scene.

class HGNNQAModel(nn.Module):
    """
    Multimodal accident classifier with a learnable situation hypergraph.

    Modalities:
        visual : MobileNetV3-small, mean-pooled over T keyframes
        text   : BERT [CLS] token for the question
        graph  : SituationHGNN over the scene graph node features

    Args:
        n_answers     : size of the answer vocabulary
        node_feat_dim : dimension of per-node features fed to the HGNN (default 64)
        hidden        : shared hidden dimension for all projection heads
        n_hg_edges    : number of learnable hyperedges K
    """

    def __init__(
        self,
        n_answers     : int,
        node_feat_dim : int = 64,
        hidden        : int = 256,
        n_hg_edges    : int = 16,
    ):
        super().__init__()

        # ── Visual encoder (frozen MobileNetV3-small) ────────────────────
        import torchvision.models as M
        backbone  = M.mobilenet_v3_small(weights=M.MobileNet_V3_Small_Weights.DEFAULT)
        self.vis  = nn.Sequential(*list(backbone.children())[:-1])
        vis_dim   = 576
        self.vis_proj = nn.Linear(vis_dim, hidden)

        # ── Text encoder (BERT) ──────────────────────────────────────────
        from transformers import BertModel
        self.bert      = BertModel.from_pretrained("bert-base-uncased")
        self.bert_proj = nn.Linear(768, hidden)

        # ── Situation Hypergraph Neural Network ──────────────────────────
        self.hgnn = SituationHGNN(
            node_feat_dim=node_feat_dim,
            hidden_dim=hidden,
            n_classes=hidden,      # output is an embedding, not logits here
            n_edges=n_hg_edges,
        )
        # Override the HGNN head so it outputs an embedding vector
        self.hgnn.head = nn.Identity()   # pooled [hidden] passes straight through

        # ── Fusion + classifier ──────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_answers),
        )

    def forward(
        self,
        frames        : "torch.Tensor",      # [B, T, C, H, W]
        input_ids     : "torch.Tensor",      # [B, seq_len]
        attention_mask: "torch.Tensor",      # [B, seq_len]
        scene_graphs  : "list[list[dict]]",  # B scene graph dicts (variable N)
    ) -> "torch.Tensor":                     # [B, n_answers]

        B, T, C, H, W = frames.shape

        # ── Visual features ──────────────────────────────────────────────
        vis_out  = self.vis(frames.view(B * T, C, H, W))      # [B*T, 576, 1, 1]
        vis_feat = vis_out.view(B, T, -1).mean(dim=1)         # [B, 576]
        vis_feat = self.vis_proj(vis_feat)                     # [B, hidden]

        # ── Text features ────────────────────────────────────────────────
        bert_out  = self.bert(input_ids=input_ids,
                              attention_mask=attention_mask)
        txt_feat  = self.bert_proj(
            bert_out.last_hidden_state[:, 0])                 # [B, hidden]

        # ── Hypergraph features (variable-size, process one-by-one) ─────
        hg_feats = []
        for sg in scene_graphs:
            node_feats = node_features_from_sg(
                sg if isinstance(sg, list) else sg.get("scene_graphs", []),
                feat_dim=64,
            ).to(frames.device)                               # [N, 64]
            emb = self.hgnn(node_feats)                       # [hidden]
            hg_feats.append(emb)
        hg_feat = torch.stack(hg_feats)                       # [B, hidden]

        # ── Fuse modalities and classify ─────────────────────────────────
        fused = torch.cat([vis_feat, txt_feat, hg_feat], dim=-1)  # [B, hidden*3]
        return self.classifier(fused)                         # [B, n_answers]


# ── helpers ─────────────────────────────────────────────────────────────

def build_answer_vocab(records):
    """Map unique answer strings to integer indices."""
    answers = set()
    for r in records:
        ans = r.get("causes") or r.get("texts") or "unknown"
        answers.add(ans)
    vocab = {a: i for i, a in enumerate(sorted(answers))}
    return vocab


def get_sg_node_count(scene_graph: list[dict]) -> int:
    """Total detected objects across all frames."""
    return sum(len(f.get("objects", [])) for f in scene_graph)


def evaluate(model, loader, answer_vocab, device):
    """Return overall accuracy and per-category accuracy dict."""
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    idx_to_ans = {v: k for k, v in answer_vocab.items()}

    correct = total = 0
    cat_correct = defaultdict(int)
    cat_total   = defaultdict(int)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            frames    = batch["frames"].to(device)
            questions = batch["question"]
            answers   = batch["answer"]
            categories = batch["category"].tolist()
            sgs       = batch["scene_graph"]

            # Tokenise questions
            enc = tokenizer(questions, return_tensors="pt",
                            padding=True, truncation=True, max_length=32)
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # Scene graph node counts
            sg_counts = torch.tensor([get_sg_node_count(sg) for sg in sgs],
                                     device=device)

            logits = model(frames, input_ids, attention_mask, sg_counts)
            preds  = logits.argmax(dim=-1).cpu().tolist()

            for pred, ans, cat in zip(preds, answers, categories):
                predicted_ans = idx_to_ans.get(pred, "")
                is_correct    = int(predicted_ans == ans)
                correct      += is_correct
                total        += 1
                cat_correct[cat] += is_correct
                cat_total[cat]   += 1

    overall = correct / total if total else 0
    per_cat = {cat: cat_correct[cat] / cat_total[cat]
               for cat in cat_total}
    return overall, per_cat


# ── main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int,   default=10)
    parser.add_argument("--batch",     type=int,   default=8)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--baseline",  action="store_true",
                        help="Use LightHGQA (scalar node-count) instead of HGNNQAModel")
    args = parser.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device  : {DEVICE}")
    print(f"Run dir : {RUN_DIR}\n")

    # Datasets
    print("Loading datasets...")
    train_ds = MMAUDataset(split="train",  categories=CATEGORIES)
    val_ds   = MMAUDataset(split="val",    categories=CATEGORIES)
    test_ds  = MMAUDataset(split="test",   categories=CATEGORIES)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}\n")

    if len(train_ds) == 0:
        print("No preprocessed data found. Run preprocess.py first.")
        return

    # Build answer vocabulary from train set
    answer_vocab = build_answer_vocab(
        [train_ds.records[i] for i in range(len(train_ds))]
    )
    n_answers = len(answer_vocab)
    print(f"Answer vocabulary size: {n_answers}\n")
    (RUN_DIR / "answer_vocab.json").write_text(json.dumps(answer_vocab, indent=2))

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  collate_fn=collate_fn, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, collate_fn=collate_fn, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch,
                              shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Model — HGNNQAModel by default, LightHGQA with --baseline flag
    if args.baseline:
        print("Model: LightHGQA (scalar node-count baseline)")
        model = LightHGQA(n_answers=n_answers).to(DEVICE)
    else:
        print("Model: HGNNQAModel (learnable situation hypergraph)")
        model = HGNNQAModel(n_answers=n_answers).to(DEVICE)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if args.eval_only:
        ckpt = RUN_DIR / "best_model.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            acc, per_cat = evaluate(model, test_loader, answer_vocab, DEVICE)
            print(f"Test accuracy : {acc:.4f} ({acc*100:.2f}%)")
            print("Per-category  :")
            for cat, a in sorted(per_cat.items()):
                label = {11: "ego-car hitting car", 43: "car hitting car"}.get(cat, f"cat {cat}")
                print(f"  {label:<25}: {a:.4f} ({a*100:.1f}%)")
        else:
            print("No checkpoint found. Train first.")
        return

    # Training loop
    log = []
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            frames    = batch["frames"].to(DEVICE)
            questions = batch["question"]
            answers   = batch["answer"]
            sgs       = batch["scene_graph"]

            # Map answers to indices (unknown → 0)
            labels = torch.tensor(
                [answer_vocab.get(a, 0) for a in answers], device=DEVICE
            )

            enc = tokenizer(questions, return_tensors="pt",
                            padding=True, truncation=True, max_length=32)
            input_ids      = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)

            optimizer.zero_grad()
            if args.baseline:
                sg_counts = torch.tensor([get_sg_node_count(sg) for sg in sgs],
                                         device=DEVICE)
                logits = model(frames, input_ids, attention_mask, sg_counts)
            else:
                logits = model(frames, input_ids, attention_mask, sgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_acc, val_per_cat = evaluate(model, val_loader, answer_vocab, DEVICE)

        print(f"Epoch {epoch:02d} | loss {avg_loss:.4f} | val acc {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), RUN_DIR / "best_model.pt")
            print(f"  ↑ New best val accuracy: {best_val:.4f}")

        log.append({"epoch": epoch, "loss": avg_loss, "val_acc": val_acc,
                    "val_per_cat": val_per_cat})

    (RUN_DIR / "training_log.json").write_text(json.dumps(log, indent=2))

    # Final evaluation on test set
    print("\n=== Test Set Evaluation ===")
    model.load_state_dict(torch.load(RUN_DIR / "best_model.pt", map_location=DEVICE))
    test_acc, test_per_cat = evaluate(model, test_loader, answer_vocab, DEVICE)

    print(f"Overall accuracy  : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("Per-category breakdown:")
    cat_labels = {11: "ego-car hitting car", 43: "car hitting car"}
    for cat, acc in sorted(test_per_cat.items()):
        label = cat_labels.get(cat, f"category {cat}")
        print(f"  {label:<25}: {acc:.4f} ({acc*100:.1f}%)")

    results = {"overall": test_acc, "per_category": test_per_cat}
    (RUN_DIR / "per_category_accuracy.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {RUN_DIR}/per_category_accuracy.json")


if __name__ == "__main__":
    main()
