"""
Hard-negative composition ablation (brief section 8): score-only /
"no hard negatives" (random negatives only) vs "specificity hard negatives
only" (same-subject/diff-relation + same-relation/diff-subject, the two
categories EVOKE's paper specifically flags as the under-covered case) vs the
full 8-category mix already trained (akew_verifier_ckpt).

Trains two new small verifiers on CounterFact+WikiUpdate (same scope as the
ORIGINAL v1 training, not the MQuAKE-augmented v2 -- isolating the hard-
negative-composition question from the separate MQuAKE-generalization
question), evaluates all three on the SAME held-out val split for a fair
three-way comparison.
"""
import sys, json, random
sys.path.insert(0, "src")
from akew_data import load_akew
from akew_splits import subject_disjoint_split, assert_subject_disjoint
from akew_verifier_train import TRAIN_DATASETS, MODES
from akew_verifier_eval import confusion_at_threshold, auroc_auprc, pick_threshold
from sentence_transformers import CrossEncoder, InputExample
import torch
import numpy as np


def _card_text(card):
    return card.canonical_fact_text or card.raw_evidence_text or ""


def assemble_random_negatives(cards, golds, seed=0, neg_per_pos=4):
    """'No hard negatives' baseline: for each positive, sample NEG_PER_POS
    uniformly random OTHER cards as negatives -- no structure, no targeting."""
    rng = random.Random(seed)
    rows = []
    all_ids = [c.edit_id for c in cards]
    by_id = {c.edit_id: c for c in cards}
    for c in cards:
        g = golds.get(c.edit_id)
        q = g.eval_question if (g and g.eval_question) else _card_text(c)
        if not q or not _card_text(c):
            continue
        rows.append((q, _card_text(c), 1, "positive"))
        others = [i for i in all_ids if i != c.edit_id]
        picks = rng.sample(others, min(neg_per_pos, len(others)))
        for pid in picks:
            rows.append((q, _card_text(by_id[pid]), 0, "random"))
    return rows


def assemble_specificity_only(cards, golds, seed=0, neg_per_pos=4):
    """'Specificity hard negatives only' -- categories 1+2 exclusively
    (same_subject_diff_relation, same_relation_diff_subject), no neighbor/
    stale/sibling/prefixed/ranked-below categories."""
    from akew_hard_negatives import build_specificity_negatives
    rng = random.Random(seed)
    negs = build_specificity_negatives(cards, golds)
    negs_by_source = {}
    for n in negs:
        negs_by_source.setdefault(n.source_edit_id, []).append(n)
    rows = []
    for c in cards:
        g = golds.get(c.edit_id)
        q = g.eval_question if (g and g.eval_question) else _card_text(c)
        if not q or not _card_text(c):
            continue
        rows.append((q, _card_text(c), 1, "positive"))
        cand = negs_by_source.get(c.edit_id, [])
        rng.shuffle(cand)
        for n in cand[:neg_per_pos]:
            if n.candidate_text:
                rows.append((q, n.candidate_text, 0, n.negative_type))
    return rows


def build_variant_data(assembler_fn):
    train_rows, val_rows = [], []
    for ds in TRAIN_DATASETS:
        for mode in MODES:
            cards, golds, _groups = load_akew(ds, mode)
            tr, va, te = subject_disjoint_split(cards, train_frac=0.7, val_frac=0.15, seed=0)
            violations = assert_subject_disjoint(tr, va, te)
            assert not violations
            train_rows += assembler_fn(tr, golds, seed=0)
            val_rows += assembler_fn(va, golds, seed=0)
    return train_rows, val_rows


def train_and_eval(name, train_rows, val_rows, output_path):
    train_examples = [InputExample(texts=[q, c], label=float(l)) for q, c, l, _cat in train_rows]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1, device=device)
    train_dl = torch.utils.data.DataLoader(train_examples, shuffle=True, batch_size=32)
    model.fit(train_dataloader=train_dl, epochs=2, warmup_steps=int(0.1 * len(train_dl)),
              show_progress_bar=False)
    model.save(output_path)

    val_pairs = [[q, c] for q, c, l, _cat in val_rows]
    val_labels = [l for q, c, l, _cat in val_rows]
    scores = model.predict(val_pairs, convert_to_numpy=True, show_progress_bar=False)
    scores = 1 / (1 + np.exp(-scores))
    thresh, _ = pick_threshold(scores, val_labels)
    auroc, auprc = auroc_auprc(scores, val_labels)
    conf = confusion_at_threshold(scores, val_labels, thresh)
    return {"variant": name, "n_train": len(train_rows), "n_val": len(val_rows),
            "threshold": round(thresh, 3), "auroc": auroc, "auprc": auprc, **conf}


if __name__ == "__main__":
    results = {}

    print("=== Variant: no hard negatives (random only) ===", file=sys.stderr)
    tr, va = build_variant_data(assemble_random_negatives)
    results["no_hard_negatives_random_only"] = train_and_eval(
        "no_hard_negatives_random_only", tr, va, "outputs/akew_verifier_ablation_random")

    print("=== Variant: specificity hard negatives only ===", file=sys.stderr)
    tr, va = build_variant_data(assemble_specificity_only)
    results["specificity_only"] = train_and_eval(
        "specificity_only", tr, va, "outputs/akew_verifier_ablation_specificity")

    print("<<<JSON>>>")
    print(json.dumps(results, indent=2))
    print("<<<END>>>")
