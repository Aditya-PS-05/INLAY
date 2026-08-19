"""AKEW retrieval-only pilot: Recall@1/@5/MRR for dense retrieval alone, before
any router or generation is involved. This isolates retrieval quality from
downstream routing/generation failures (brief section 6's oracle-diagnostics
principle, applied one stage earlier than the brief's own oracle list covers).

Usage: python akew_retrieval_pilot.py <dataset> <input_mode> [limit]
"""
import sys, json, random
sys.path.insert(0, "src")
from akew_data import load_akew
from akew_retrieval import DenseCardIndex, recall_and_mrr

DATASET = sys.argv[1] if len(sys.argv) > 1 else "CounterFact"
MODE = sys.argv[2] if len(sys.argv) > 2 else "structured"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 100

random.seed(0)
cards, golds, groups = load_akew(DATASET, MODE)
if LIMIT and LIMIT < len(cards):
    idx = sorted(random.sample(range(len(cards)), LIMIT))
    cards = [cards[i] for i in idx]

index = DenseCardIndex()
index.build(cards)

# Build eval queries: the AKEW `question` field (gold, eval-only -- read here
# ONLY as the evaluator, never during ingestion/retrieval-index construction,
# which is why this script imports akew_data's already-separated GoldRecord
# rather than reaching back into raw AKEW json).
eval_pairs = []
for c in cards:
    g = golds.get(c.edit_id)
    if g and g.eval_question:
        eval_pairs.append((g.eval_question, c.edit_id))

metrics = recall_and_mrr(index, eval_pairs, topk=5)
out = {"dataset": DATASET, "input_mode": MODE, "n_cards": len(cards),
       "n_eval_pairs": len(eval_pairs), **metrics}
print("<<<JSON>>>")
print(json.dumps(out))
print("<<<END>>>")
