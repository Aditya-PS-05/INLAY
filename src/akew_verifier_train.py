"""
Train the scope-verifier cross-encoder (brief section 4, stage 2).

Protocol (brief section 9): calibrate on CounterFact + WikiUpdate, subject-
disjoint train/val split within each. MQuAKE-CF is held out ENTIRELY as a
zero-shot test set -- never seen during training or threshold calibration --
matching the brief's suggested alternative validation experiment.

Positive examples: (query, own_card_text, label=1), one per card per input
mode. Negative examples: the 8 hard-negative categories from
akew_hard_negatives.py, capped per positive to keep the class balance
reasonable and to spread coverage across categories rather than letting the
numerically dominant categories (same_relation_diff_subject, ranked_below_correct)
drown out the rarer, more diagnostic ones (same_subject_diff_relation,
stale_object_same_slot).
"""
import sys, json, random
sys.path.insert(0, "src")
from akew_data import load_akew
from akew_retrieval import DenseCardIndex
from akew_hard_negatives import build_all_hard_negatives
from akew_splits import subject_disjoint_split, assert_subject_disjoint
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
import torch

TRAIN_DATASETS = ["CounterFact", "WikiUpdate"]   # MQuAKE-CF held out entirely
MODES = ["structured", "unstructured", "extracted"]
NEG_PER_POS = 4       # cap: spread across categories rather than let volume dominate
SEED = 0


def _card_text(card):
    return card.canonical_fact_text or card.raw_evidence_text or ""


def load_raw_triplets(dataset_name):
    import os
    _here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(_here, "..", "..", "AKEW", "repo", "datasets", f"{dataset_name}.json")
    raw = json.load(open(path))
    out = {}
    if dataset_name == "MQuAKE-CF":
        for i, rec in enumerate(raw):
            for j, rr in enumerate(rec["requested_rewrite"]):
                out[f"mquake_{i}_edit{j}"] = rr.get("unsfact_triplets_GPT", [])
    else:
        for rec in raw:
            eid = f"{dataset_name}_{rec.get('case_id')}"
            out[eid] = rec["requested_rewrite"].get("unsfact_triplets_GPT", [])
    return out


def assemble_examples(cards, golds, dataset_name, seed=SEED):
    """Returns list of (query_text, candidate_text, label, category) for one
    dataset x mode's cards. category is 'positive' for label=1 rows."""
    rng = random.Random(seed)
    index = DenseCardIndex()
    index.build(cards)
    raw_triplets = load_raw_triplets(dataset_name) if any(c.input_mode == "extracted" for c in cards) else None
    negs = build_all_hard_negatives(cards, golds, index, dataset_name, raw_triplets)

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
        cand_negs = negs_by_source.get(c.edit_id, [])
        # spread across categories: shuffle within each category, then round-robin
        by_cat = {}
        for n in cand_negs:
            by_cat.setdefault(n.negative_type, []).append(n)
        for v in by_cat.values():
            rng.shuffle(v)
        picked = []
        cats_cycle = list(by_cat.keys())
        i = 0
        while len(picked) < NEG_PER_POS and cats_cycle:
            cat = cats_cycle[i % len(cats_cycle)]
            if by_cat[cat]:
                picked.append(by_cat[cat].pop())
            else:
                cats_cycle.remove(cat)
                if not cats_cycle:
                    break
                continue
            i += 1
        for n in picked:
            if n.candidate_text:
                rows.append((q, n.candidate_text, 0, n.negative_type))
    return rows


def build_split_examples(dataset_names, modes, seed=SEED):
    """Loads each dataset x mode, subject-disjoint splits it, pools train/val
    across datasets and modes. Returns (train_rows, val_rows, split_report)."""
    train_rows, val_rows = [], []
    report = {}
    for ds in dataset_names:
        for mode in modes:
            cards, golds, _groups = load_akew(ds, mode)
            tr, va, te = subject_disjoint_split(cards, train_frac=0.7, val_frac=0.15, seed=seed)
            violations = assert_subject_disjoint(tr, va, te)
            assert not violations, f"{ds}/{mode}: subject leakage across splits: {violations[:5]}"
            tr_rows = assemble_examples(tr, golds, ds, seed=seed)
            va_rows = assemble_examples(va, golds, ds, seed=seed)
            train_rows += tr_rows
            val_rows += va_rows
            report[f"{ds}/{mode}"] = {"n_cards_train": len(tr), "n_cards_val": len(va),
                                       "n_rows_train": len(tr_rows), "n_rows_val": len(va_rows)}
    return train_rows, val_rows, report


if __name__ == "__main__":
    print("Assembling training data (CounterFact + WikiUpdate, subject-disjoint, MQuAKE-CF held out entirely)...")
    train_rows, val_rows, report = build_split_examples(TRAIN_DATASETS, MODES)
    print(json.dumps(report, indent=2))
    print(f"\ntotal train rows: {len(train_rows)} (pos={sum(1 for r in train_rows if r[2]==1)}, "
          f"neg={sum(1 for r in train_rows if r[2]==0)})")
    print(f"total val rows:   {len(val_rows)} (pos={sum(1 for r in val_rows if r[2]==1)}, "
          f"neg={sum(1 for r in val_rows if r[2]==0)})")

    from sentence_transformers import InputExample
    train_examples = [InputExample(texts=[q, c], label=float(l)) for q, c, l, _cat in train_rows]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1, device=device)

    train_dl = torch.utils.data.DataLoader(train_examples, shuffle=True, batch_size=32)
    val_evaluator = CEBinaryClassificationEvaluator(
        [[q, c] for q, c, l, _cat in val_rows],
        [l for q, c, l, _cat in val_rows],
        name="akew_val",
    )

    print("\nTraining...")
    model.fit(train_dataloader=train_dl, evaluator=val_evaluator, epochs=2,
              warmup_steps=int(0.1 * len(train_dl)), output_path="outputs/akew_verifier_ckpt",
              show_progress_bar=False)

    # save_best_model=True (fit()'s default) did NOT persist actual model files in
    # this sentence-transformers version (5.6.1, Trainer-backed CrossEncoder) --
    # verified by direct inspection: only an eval/ CSV subfolder existed after fit()
    # returned, no config.json/model.safetensors anywhere. Explicit, version-
    # independent save as the real persistence step, not relying on fit()'s internals.
    model.save("outputs/akew_verifier_ckpt")
    import os
    saved_files = os.listdir("outputs/akew_verifier_ckpt")
    assert "config.json" in saved_files or any(f.endswith(".safetensors") for f in saved_files), \
        f"model.save() did not produce expected files: {saved_files}"

    print(f"\nTraining complete. Checkpoint saved to outputs/akew_verifier_ckpt: {saved_files}")
    print("<<<TRAIN_DONE>>>")
