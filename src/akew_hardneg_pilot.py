"""Sanity-check pilot for hard-negative construction: builds all 8 categories
on a real dataset and reports non-empty counts per category, plus prints one
worked example per category so the construction can be eyeballed, not just
counted.

Usage: python akew_hardneg_pilot.py <dataset> <input_mode>
"""
import sys, json
sys.path.insert(0, "src")
from akew_data import load_akew
from akew_retrieval import DenseCardIndex
from akew_hard_negatives import build_all_hard_negatives, CATEGORIES

DATASET = sys.argv[1] if len(sys.argv) > 1 else "WikiUpdate"
MODE = sys.argv[2] if len(sys.argv) > 2 else "extracted"

cards, golds, groups = load_akew(DATASET, MODE)
index = DenseCardIndex()
index.build(cards)

# raw triplets, for the same-document-sibling category (extracted mode only) --
# read directly from the AKEW json here (evaluator/training-data-construction
# context, not ingestion), same pattern as akew_hard_negatives' own docstring
# describes for stale_object_same_slot.
raw_triplets_by_edit = {}
if MODE == "extracted":
    import os
    _here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(_here, "..", "..", "AKEW", "repo", "datasets", f"{DATASET}.json")
    raw = json.load(open(path))
    if DATASET == "MQuAKE-CF":
        for i, rec in enumerate(raw):
            for j, rr in enumerate(rec["requested_rewrite"]):
                raw_triplets_by_edit[f"mquake_{i}_edit{j}"] = rr.get("unsfact_triplets_GPT", [])
    else:
        for rec in raw:
            eid = f"{DATASET}_{rec.get('case_id')}"
            raw_triplets_by_edit[eid] = rec["requested_rewrite"].get("unsfact_triplets_GPT", [])

negs = build_all_hard_negatives(cards, golds, index, DATASET, raw_triplets_by_edit)

by_cat = {}
for n in negs:
    by_cat.setdefault(n.negative_type, []).append(n)

print(f"=== {DATASET} / {MODE}: {len(cards)} cards, {len(negs)} total hard negatives ===\n")
for cat in CATEGORIES:
    items = by_cat.get(cat, [])
    print(f"{cat:32s} n={len(items):5d}", "  <-- ZERO, check construction" if not items else "")
    if items:
        ex = items[0]
        print(f"    example query: {ex.query_text[:100]!r}")
        print(f"    wrong card:    {ex.wrong_card_id[:80]!r}  (true edit: {ex.source_edit_id})")
        print(f"    candidate_text: {ex.candidate_text[:100]!r}")
        empty_ct = sum(1 for x in items if not x.candidate_text)
        if empty_ct:
            print(f"    WARNING: {empty_ct}/{len(items)} examples in this category have empty candidate_text")
