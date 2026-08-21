"""
Generate OUTCOME labels for the decision-theoretic head (v2).

WHY THIS EXISTS — the diagnosis, reached twice independently:

The v1 reliability head predicts P(top-1 retrieval is correct) and does it well
(OOD AUROC 0.956 on a dataset it never saw). But two separate experiments then
showed that is not the quantity the router needs:

  1. The three-way policy (akew_reliability_head_results.md): using the
     MAGNITUDE of predicted unreliability to pick REJECT vs REASON failed on
     the cell it was designed to fix (43.13% -> 41.25%) and was catastrophic on
     MQuAKE-CF (80.95% -> 58.73%).
  2. The multi-hop per-hop gate (akew_multihop_results.md): swapping the raw
     verifier threshold for the head changed nothing (53.95% -> 53.39%, two
     examples), because the fallback already makes both branches survivable.

Both failures have one cause: `P(retrieval correct)` and `P(this action yields
a correct answer)` are different predicates, and they come apart in both
directions. A wrong retrieval can still be answered correctly from parametric
knowledge, so declining loses a point that was available. A right retrieval can
still be misread, so trusting it loses one too. Tellingly, the fixed router's
verifier-threshold REJECT set is a WORSE predictor of retrieval correctness yet
a BETTER-chosen action set on WikiUpdate -- which is exactly what a target
mismatch looks like from the outside.

WHAT THIS SCRIPT DOES

For every training query it actually RUNS each candidate action and scores the
result, producing one binary outcome label per action:

    REJECT  -> answer_no_context(...)        -> hit?
    DIRECT  -> answer_hard_playback(...)     -> hit?   (structured mode only)
    REASON  -> answer_contextual(...)        -> hit?

Together with the same 15 features the v1 head already uses, that gives a
supervised dataset for "which action pays off here", which is the question the
router is actually asking. This is why it was not built first: it costs up to
three generations per training query instead of zero.

DIRECT is emitted as null outside structured mode, deliberately rather than as
a zero: akew_router gates DIRECT on input_mode == "structured" (the router bug
found in akew_fullpipeline_results.md), so outside structured mode DIRECT is
not a legal action rather than a bad one, and labelling it 0 would teach the
head to avoid something it was never allowed to choose.

Held out: MQuAKE-CF entirely, mirroring the v1 protocol so the OOD claim stays
genuinely out-of-distribution.

Usage: AKEW_MODEL=... python akew_outcome_labels.py <dataset> <mode> [limit]
Writes outputs/outcome_labels_<dataset>_<mode>.json
"""
import sys, os, json, random, pathlib

sys.path.insert(0, "src")
from akew_data import load_akew
from akew_splits import subject_disjoint_split
from akew_retrieval import DenseCardIndex
from akew_reliability import extract_features, score_candidates, FEATURE_NAMES
from akew_answering import (answer_no_context, answer_hard_playback,
                            answer_contextual, is_hit)
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

DATASET = sys.argv[1] if len(sys.argv) > 1 else "CounterFact"
MODE = sys.argv[2] if len(sys.argv) > 2 else "structured"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 250
SPLIT = os.environ.get("AKEW_SPLIT", "train")
MODEL_NAME = os.environ.get("AKEW_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
VERIFIER_PATH = "outputs/akew_verifier_ckpt_v2"
TOPK = 5              # must match the v1 head's training distribution
DIRECT_THRESHOLD = 0.85
OUT = f"outputs/outcome_labels_{DATASET}_{MODE}_{SPLIT}.json"

cards, golds, _groups = load_akew(DATASET, MODE)
tr, va, te = subject_disjoint_split(cards, train_frac=0.7, val_frac=0.15, seed=0)
subset = {"train": tr, "val": va, "test": te}[SPLIT]
random.seed(0)
if LIMIT and LIMIT < len(subset):
    subset = random.sample(subset, LIMIT)

index = DenseCardIndex()
index.build(cards)
verifier = CrossEncoder(VERIFIER_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device).eval()

card_by_id = {c.edit_id: c for c in cards}
rows = []

for n_done, c in enumerate(subset):
    g = golds.get(c.edit_id)
    if not g or not g.eval_question or not g.target_new:
        continue
    q = g.eval_question

    cands = index.query(q, topk=TOPK)
    if not cands:
        continue
    vscores = score_candidates(verifier, q, cands)
    vec, feats = extract_features(q, cands, vscores, DIRECT_THRESHOLD)
    top_card = cands[0][0]

    # --- run each candidate action for real and score the result -----------
    ans_reject = answer_no_context(model, tok, q, device)
    y_reject = int(is_hit(ans_reject, g))

    ans_reason = answer_contextual(model, tok, q, top_card, device)
    y_reason = int(is_hit(ans_reason, g))

    if MODE == "structured":
        ans_direct = answer_hard_playback(top_card, golds.get(top_card.edit_id))
        y_direct = int(is_hit(ans_direct, g))
    else:
        y_direct = None      # not a legal action here; see module docstring

    rows.append({
        "edit_id": c.edit_id,
        "dataset": DATASET, "mode": MODE,
        "features": [float(x) for x in vec],
        "retrieval_correct": int(top_card.edit_id == c.edit_id),
        "y_reject": y_reject,
        "y_reason": y_reason,
        "y_direct": y_direct,
    })
    if (n_done + 1) % 25 == 0:
        print(f"  {n_done+1}/{len(subset)}", file=sys.stderr)

pathlib.Path("outputs").mkdir(exist_ok=True)
payload = {
    "dataset": DATASET, "mode": MODE, "split": SPLIT, "model": MODEL_NAME,
    "feature_names": FEATURE_NAMES, "topk": TOPK, "n": len(rows),
    "rows": rows,
}
pathlib.Path(OUT).write_text(json.dumps(payload))


def rate(key):
    vals = [r[key] for r in rows if r[key] is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


# Printed because it is the first real look at whether the premise holds: if
# the best action were the same everywhere, a decision-theoretic head would
# have nothing to learn and the whole v2 would be pointless.
best_counts = {"REJECT": 0, "REASON": 0, "DIRECT": 0, "TIE_ALL_WRONG": 0, "TIE_MULTI": 0}
for r in rows:
    opts = {"REJECT": r["y_reject"], "REASON": r["y_reason"]}
    if r["y_direct"] is not None:
        opts["DIRECT"] = r["y_direct"]
    winners = [k for k, v in opts.items() if v == 1]
    if not winners:
        best_counts["TIE_ALL_WRONG"] += 1
    elif len(winners) == 1:
        best_counts[winners[0]] += 1
    else:
        best_counts["TIE_MULTI"] += 1

print("<<<JSON>>>")
print(json.dumps({
    "dataset": DATASET, "mode": MODE, "split": SPLIT, "n": len(rows), "out": OUT,
    "action_success_rates": {"reject": rate("y_reject"), "reason": rate("y_reason"),
                             "direct": rate("y_direct")},
    "retrieval_correct_rate": rate("retrieval_correct"),
    "uniquely_best_action": best_counts,
}, indent=2))
print("<<<END>>>")
