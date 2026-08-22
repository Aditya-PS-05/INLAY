"""
Out-of-scope (negative) extension of akew_outcome_labels.py.

WHY THIS EXISTS -- akew_routing_headroom_results.md's own "the experiment this
demands" section: every prior headroom measurement asks eval questions against
an index that DOES contain the corresponding edit, so REJECT (abstention) is
never the objectively correct action -- it wins on 19/1689 queries and never
uniquely. That is a property of the query population, not of routing, and it
caps the measurable headroom of ANY router at 0.00 by construction.

This script constructs the missing condition. For a fraction of queries
(`oos_frac`, seeded per-query so the assignment is reproducible), the query's
OWN edit card is removed from the retrieval index before the per-action
executor runs -- so retrieval returns some OTHER, wrong card. For those
queries, no edit genuinely applies, and the objectively correct behaviour is
"answer from parametric knowledge" (REJECT), scored against the PRE-edit
answer (`target_true`/`aliases_true`), not the post-edit one. For the
untouched (positive) fraction, everything is identical to
akew_outcome_labels.py: the edit is fully present in the index and correctness
is scored against `target_new`/`aliases_new` as before.

The output schema is intentionally IDENTICAL to akew_outcome_labels.py's
(same top-level keys, same per-row y_reject/y_reason/y_direct fields) plus one
extra per-row field, `is_negative`, purely for bookkeeping -- akew_headroom.py
needs zero modification to consume this file: it only ever reads
y_reject/y_reason/y_direct, and on a negative row those already encode "was
the objectively-correct target hit", because the substitution happened at
LABELING time, not at analysis time.

Implementation note on removing a card from an already-built index: the
DenseCardIndex has no exclusion-aware query() method, so we over-fetch
(TOPK + OOS_BUFFER) then drop the excluded edit_id in Python before truncating
back to TOPK. This reuses the single index built over ALL cards rather than
rebuilding a leave-one-out index per query (which would be a full re-encode
per query and is unnecessary: dropping one row post-hoc from a top-k list is
exactly equivalent to that row never having been in the index, as long as the
buffer is large enough that filtering doesn't starve the candidate list --
checked explicitly below and counted if it ever does).

Usage: AKEW_MODEL=... python akew_outcome_labels_oos.py <dataset> <mode> [limit] [oos_frac]
Writes outputs/outcome_labels_oos_<dataset>_<mode>_<split>_f<oos_frac>.json
"""
import sys, os, json, random, pathlib, dataclasses

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
OOS_FRAC = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
SPLIT = os.environ.get("AKEW_SPLIT", "train")
MODEL_NAME = os.environ.get("AKEW_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
VERIFIER_PATH = "outputs/akew_verifier_ckpt_v2"
TOPK = 5              # must match the v1 head's training distribution
OOS_BUFFER = 10        # extra candidates fetched so filtering out one card still leaves >=TOPK
DIRECT_THRESHOLD = 0.85
OUT = f"outputs/outcome_labels_oos_{DATASET}_{MODE}_{SPLIT}_f{OOS_FRAC:.2f}.json"

cards, golds, _groups = load_akew(DATASET, MODE)
tr, va, te = subject_disjoint_split(cards, train_frac=0.7, val_frac=0.15, seed=0)
subset = {"train": tr, "val": va, "test": te}[SPLIT]
random.seed(0)
if LIMIT and LIMIT < len(subset):
    subset = random.sample(subset, LIMIT)

# Deterministic per-query negative assignment, independent of the sampling
# above (a different seed stream) so oos_frac genuinely controls the negative
# rate rather than being coupled to the sampling order.
neg_rng = random.Random(1234)
is_negative_map = {c.edit_id: (neg_rng.random() < OOS_FRAC) for c in subset}

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
n_starved = 0            # negative queries where filtering left < TOPK candidates
n_no_target_true = 0     # negative queries skipped: no target_true to score against

for n_done, c in enumerate(subset):
    g = golds.get(c.edit_id)
    if not g or not g.eval_question or not g.target_new:
        continue
    q = g.eval_question
    is_neg = is_negative_map[c.edit_id]

    if is_neg and (not g.target_true or not str(g.target_true).strip()):
        n_no_target_true += 1
        continue

    cands_raw = index.query(q, topk=TOPK + OOS_BUFFER if is_neg else TOPK)
    if is_neg:
        cands = [(cc, s) for cc, s in cands_raw if cc.edit_id != c.edit_id][:TOPK]
        if len(cands) < TOPK:
            n_starved += 1
        if not cands:
            continue
    else:
        cands = cands_raw[:TOPK]
    if not cands:
        continue

    vscores = score_candidates(verifier, q, cands)
    vec, feats = extract_features(q, cands, vscores, DIRECT_THRESHOLD)
    top_card = cands[0][0]

    # scoring target: post-edit (target_new) for positives, pre-edit
    # (target_true) for negatives -- the substitution happens HERE, once, so
    # every downstream consumer (including akew_headroom.py, unmodified) sees
    # ordinary y_reject/y_reason/y_direct columns that already encode the
    # objectively-correct answer for this query's condition.
    score_gold = g if not is_neg else dataclasses.replace(
        g, target_new=g.target_true, aliases_new=(g.aliases_true or []))

    # --- run each candidate action for real and score the result -----------
    ans_reject = answer_no_context(model, tok, q, device)
    y_reject = int(is_hit(ans_reject, score_gold))

    ans_reason = answer_contextual(model, tok, q, top_card, device)
    y_reason = int(is_hit(ans_reason, score_gold))

    if MODE == "structured":
        ans_direct = answer_hard_playback(top_card, golds.get(top_card.edit_id))
        y_direct = int(is_hit(ans_direct, score_gold))
    else:
        y_direct = None      # not a legal action here; see module docstring

    rows.append({
        "edit_id": c.edit_id,
        "dataset": DATASET, "mode": MODE,
        "features": [float(x) for x in vec],
        "retrieval_correct": int(top_card.edit_id == c.edit_id),
        "is_negative": bool(is_neg),
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
    "oos_frac": OOS_FRAC, "n_starved": n_starved, "n_no_target_true": n_no_target_true,
    "rows": rows,
}
pathlib.Path(OUT).write_text(json.dumps(payload))
if not pathlib.Path(OUT).exists():
    raise IOError(f"wrote {OUT} but it does not exist after write")


def rate(key, pred=None):
    vals = [r[key] for r in rows if r[key] is not None and (pred is None or pred(r))]
    return round(sum(vals) / len(vals), 4) if vals else None


n_pos = sum(1 for r in rows if not r["is_negative"])
n_neg = sum(1 for r in rows if r["is_negative"])

print("<<<JSON>>>")
print(json.dumps({
    "dataset": DATASET, "mode": MODE, "split": SPLIT, "n": len(rows), "out": OUT,
    "oos_frac_requested": OOS_FRAC, "n_positive": n_pos, "n_negative": n_neg,
    "n_starved_negative_candidates": n_starved, "n_skipped_no_target_true": n_no_target_true,
    "action_success_rates_overall": {"reject": rate("y_reject"), "reason": rate("y_reason"),
                                     "direct": rate("y_direct")},
    "action_success_rates_positive": {"reject": rate("y_reject", lambda r: not r["is_negative"]),
                                       "reason": rate("y_reason", lambda r: not r["is_negative"]),
                                       "direct": rate("y_direct", lambda r: not r["is_negative"])},
    "action_success_rates_negative": {"reject": rate("y_reject", lambda r: r["is_negative"]),
                                       "reason": rate("y_reason", lambda r: r["is_negative"]),
                                       "direct": rate("y_direct", lambda r: r["is_negative"])},
}, indent=2))
print("<<<END>>>")
