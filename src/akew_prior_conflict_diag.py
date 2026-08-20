"""
Diagnostic for the unexplained 7B scale anomaly on WikiUpdate unstructured.

THE ANOMALY (akew_reliability_head_results.md): WikiUpdate unstructured is
4.38 points WORSE at 7B than at 1.5B for every routing condition (43.75% ->
39.37% on fixed gating), while CounterFact unstructured moves the expected
direction over the same scale jump (87.07% -> 89.12%). The reliability head
neither causes nor addresses it -- it is present in the fixed baseline -- so
it is an open anomaly, not a property of the method.

THE HYPOTHESIS BEING TESTED: WikiUpdate's defining property is stale-vs-current
entity collision -- its edits concern real-world facts whose PREVIOUS value the
model has also seen in pretraining. A larger model has stronger parametric
priors over exactly those entities, so it may override the injected edit with
what it already "knows" more often than a smaller model does.

HOW "STRONG PRIOR" IS OPERATIONALISED, WITHOUT CIRCULARITY: ask the model the
eval question with NO context at all. If it spontaneously produces
`target_true` (the pre-edit value) unprompted, it demonstrably holds that
prior. This is measured per-model, so the 7B and 1.5B runs each get their own
prior labels rather than sharing one model's -- the hypothesis is precisely
that the larger model holds MORE such priors, so assuming a shared label set
would assume away the thing under test.

WHAT WOULD CONFIRM vs REFUTE, stated before running:
  CONFIRM  - 7B holds strong priors on more examples than 1.5B, AND
             the accuracy drop concentrates in the strong-prior subset, AND
             7B reverts to target_true (answers with the OLD value despite
             correct evidence in context) more often than 1.5B.
  REFUTE   - the drop is spread evenly across prior/no-prior subsets, or the
             revert rates are comparable, in which case the collision account
             is wrong and the anomaly needs a different explanation.

A partial pattern is reported as partial. This is a diagnostic, not a place to
find support for a story already written.

Usage: AKEW_MODEL=<hf-model> python akew_prior_conflict_diag.py [dataset] [mode] [limit]
"""
import sys, os, json, random

sys.path.insert(0, "src")
from akew_data import load_akew, GoldRecord
from akew_splits import subject_disjoint_split
from akew_retrieval import DenseCardIndex
from akew_answering import answer_no_context, answer_contextual, is_hit
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

DATASET = sys.argv[1] if len(sys.argv) > 1 else "WikiUpdate"
MODE = sys.argv[2] if len(sys.argv) > 2 else "unstructured"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 200
MODEL_NAME = os.environ.get("AKEW_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

cards, golds, _groups = load_akew(DATASET, MODE)
_tr, _va, test = subject_disjoint_split(cards, train_frac=0.7, val_frac=0.15, seed=0)
random.seed(0)
if LIMIT and LIMIT < len(test):
    test = random.sample(test, LIMIT)

index = DenseCardIndex()
index.build(cards)

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device).eval()


def hits_old_value(text, g):
    """Reuse is_hit's matching (diacritic/case-insensitive substring, alias
    aware) against the PRE-EDIT value by building a GoldRecord whose
    target_new slot carries target_true. Avoids a second, subtly-different
    matcher drifting from the one every other number in this project uses."""
    if not g.target_true:
        return False
    shim = GoldRecord(edit_id=g.edit_id, target_new=g.target_true, target_true=None,
                      aliases_new=list(g.aliases_true or []), aliases_true=[])
    return is_hit(text, shim)


rows = []
for c in test:
    g = golds.get(c.edit_id)
    if not g or not g.eval_question or not g.target_new:
        continue
    q = g.eval_question

    # 1. Closed-book: does this model hold the pre-edit value unprompted?
    no_ctx = answer_no_context(model, tok, q, device)
    strong_prior = hits_old_value(no_ctx, g)

    # 2. With the retrieved evidence in context (the always-REASON condition,
    #    chosen because it is the one path with no gating confound -- every
    #    example gets evidence, so any failure is the model declining to use
    #    it rather than the router withholding it).
    top1 = index.query(q, topk=1)
    ctx_ans = answer_contextual(model, tok, q, top1[0][0], device) if top1 \
        else answer_no_context(model, tok, q, device)

    rows.append({
        "edit_id": c.edit_id,
        "strong_prior": bool(strong_prior),
        "retrieval_correct": bool(top1 and top1[0][0].edit_id == c.edit_id),
        "hit_new": bool(is_hit(ctx_ans, g)),          # answered the EDITED value
        "reverted_to_old": bool(hits_old_value(ctx_ans, g)),  # answered the PRE-EDIT value
        "no_ctx_hit_new": bool(is_hit(no_ctx, g)),
    })

n = len(rows)
prior_rows = [r for r in rows if r["strong_prior"]]
noprior_rows = [r for r in rows if not r["strong_prior"]]


def rate(subset, key):
    return round(sum(r[key] for r in subset) / len(subset), 4) if subset else None


out = {
    "dataset": DATASET, "input_mode": MODE, "model": MODEL_NAME, "n": n,
    "strong_prior_rate": round(len(prior_rows) / n, 4) if n else None,
    "n_strong_prior": len(prior_rows),
    "n_no_prior": len(noprior_rows),
    "accuracy_overall": rate(rows, "hit_new"),
    "accuracy_strong_prior_subset": rate(prior_rows, "hit_new"),
    "accuracy_no_prior_subset": rate(noprior_rows, "hit_new"),
    "reverted_to_old_overall": rate(rows, "reverted_to_old"),
    "reverted_to_old_strong_prior": rate(prior_rows, "reverted_to_old"),
    "reverted_to_old_no_prior": rate(noprior_rows, "reverted_to_old"),
    "retrieval_correct_rate": rate(rows, "retrieval_correct"),
    # Per-example rows so the two model scales can be compared with the same
    # paired machinery used everywhere else (akew_stats.py).
    "per_example": rows,
}
print("<<<JSON>>>")
print(json.dumps(out, indent=2))
print("<<<END>>>")
