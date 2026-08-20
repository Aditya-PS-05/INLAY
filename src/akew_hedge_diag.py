"""
Second diagnostic for the 7B WikiUpdate scale anomaly, after the FIRST
hypothesis was refuted.

WHAT WAS REFUTED (akew_prior_conflict_diag.py): the idea that a larger model's
stronger parametric priors cause it to override injected edits. Refuted on its
own pre-stated criteria -- the rate of reverting to the pre-edit value went
DOWN at 7B (2.50% -> 1.87%), the loss sat in the NO-strong-prior subset, and
the CounterFact control has 4x the strong-prior rate while improving at 7B.

WHAT THE REFUTATION POINTS AT INSTEAD: at 7B the model is not answering the
OLD value -- it reverts less -- so it is failing some other way. The one
observation already on record in this project is that the 7B model hedges on
weak evidence ("Based on the given evidence, we cannot determine who the head
of state...", akew_multihop_results.md). WikiUpdate has ~28% wrong retrievals
to hedge about; CounterFact has ~1%. That predicts hedging rather than priors
drives the gap, and it explains why the degradation is dataset-specific.

WHAT WOULD CONFIRM vs REFUTE, again stated before running:
  CONFIRM  - 7B produces markedly more non-answers/hedges than 1.5B on
             WikiUpdate, AND the excess concentrates on examples where
             retrieval is WRONG (the evidence genuinely is unhelpful), AND
             the CounterFact control shows a much smaller hedge gap.
  REFUTE   - hedge rates are comparable across scales, or the excess is
             spread evenly across correct/incorrect retrievals, in which case
             this account fails too and the anomaly remains open.

Hedging is detected by surface cues rather than a model judge, deliberately:
the cue list is fixed, inspectable, and reported alongside raw sample outputs
so the reader can check the classifier rather than trust it. Cue matching is
crude and will both over- and under-count; the sample dump exists so that
error is visible rather than hidden behind a single rate.

Usage: AKEW_MODEL=<hf-model> python akew_hedge_diag.py [dataset] [mode] [limit]
"""
import sys, os, json, random

sys.path.insert(0, "src")
from akew_data import load_akew
from akew_splits import subject_disjoint_split
from akew_retrieval import DenseCardIndex
from akew_answering import answer_contextual, answer_no_context, is_hit
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

DATASET = sys.argv[1] if len(sys.argv) > 1 else "WikiUpdate"
MODE = sys.argv[2] if len(sys.argv) > 2 else "unstructured"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 200
MODEL_NAME = os.environ.get("AKEW_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# Surface cues for "declined to commit to an answer".
#
# REWRITTEN after inspecting the first run's sample dump, which showed the
# original literal-string list silently UNDER-counting: it had "does not
# contain / provide / mention / specify" but not "does not INCLUDE", and
# "does not include" turned out to be this model's most common phrasing --
# three of seven sampled failures used it and were scored as non-hedges. A
# rate computed from that list was not measuring what it claimed to.
#
# Replaced with pattern families rather than a longer literal list, since the
# same drift would recur with the next unlisted verb. The sample dump below
# is the check on THIS version -- it exists so the classifier stays auditable
# rather than trusted.
import re as _re

HEDGE_PATTERNS = [
    _re.compile(p) for p in [
        r"(does|do|did)\s+not\s+(contain|include|provide|specify|mention|have|appear|state|indicate|give)",
        r"(doesn't|don't|didn't)\s+(contain|include|provide|specify|mention|have|appear|state|indicate|give)",
        r"\b(no|insufficient|not enough|lacks?)\s+(information|details?|evidence|mention|data)",
        r"(cannot|can't|could not|couldn't|unable to)\s+(be\s+)?(determine|answer|find|tell|say|identify|confirm)",
        r"\bnot\s+(specified|mentioned|provided|included|available|clear|stated|given)",
        r"\bthere\s+is\s+no\s+(information|mention|evidence|indication|detail)",
        r"\b(unclear|not possible to)\b",
        r"\bevidence\s+(provided|given)\s+does\s+not\b",
    ]
]

# Counted and reported SEPARATELY, not folded into the hedge rate: the model
# asserting a negative ("X is not currently a member of Y") is a different
# behaviour from declining for lack of evidence, even though both produce a
# miss. Merging them would blur the very distinction under test.
NEGATIVE_ASSERTION = _re.compile(r"\bis\s+not\s+(currently\s+)?(a\s+)?(member|part|player|the)\b")


def is_hedge(text):
    t = (text or "").lower()
    return any(p.search(t) for p in HEDGE_PATTERNS)


def is_negative_assertion(text):
    return bool(NEGATIVE_ASSERTION.search((text or "").lower()))


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

rows, samples = [], []
for c in test:
    g = golds.get(c.edit_id)
    if not g or not g.eval_question or not g.target_new:
        continue
    q = g.eval_question
    top1 = index.query(q, topk=1)
    ans = answer_contextual(model, tok, q, top1[0][0], device) if top1 \
        else answer_no_context(model, tok, q, device)
    retrieval_ok = bool(top1 and top1[0][0].edit_id == c.edit_id)
    hedged = is_hedge(ans)
    neg = is_negative_assertion(ans)
    rows.append({
        "edit_id": c.edit_id, "retrieval_correct": retrieval_ok,
        "hedged": hedged, "negative_assertion": neg,
        "declined": bool(hedged or neg),
        "hit": bool(is_hit(ans, g)),
        "answer_len_words": len(ans.split()),
    })
    if len(samples) < 16 and not is_hit(ans, g):
        samples.append({"q": q, "gold": g.target_new, "answer": ans,
                        "retrieval_correct": retrieval_ok, "hedged": hedged,
                        "negative_assertion": neg})

n = len(rows)
ok_rows = [r for r in rows if r["retrieval_correct"]]
bad_rows = [r for r in rows if not r["retrieval_correct"]]


def rate(subset, key):
    return round(sum(r[key] for r in subset) / len(subset), 4) if subset else None


def mean(subset, key):
    return round(sum(r[key] for r in subset) / len(subset), 2) if subset else None


out = {
    "dataset": DATASET, "input_mode": MODE, "model": MODEL_NAME, "n": n,
    "accuracy": rate(rows, "hit"),
    "hedge_rate_overall": rate(rows, "hedged"),
    "hedge_rate_retrieval_correct": rate(ok_rows, "hedged"),
    "hedge_rate_retrieval_wrong": rate(bad_rows, "hedged"),
    "negative_assertion_rate": rate(rows, "negative_assertion"),
    # hedge OR negative assertion -- both are "did not commit to an answer",
    # reported alongside the narrow hedge rate rather than replacing it.
    "declined_rate_overall": rate(rows, "declined"),
    "declined_rate_retrieval_correct": rate(ok_rows, "declined"),
    "declined_rate_retrieval_wrong": rate(bad_rows, "declined"),
    "n_retrieval_correct": len(ok_rows),
    "n_retrieval_wrong": len(bad_rows),
    "mean_answer_words": mean(rows, "answer_len_words"),
    "mean_answer_words_retrieval_wrong": mean(bad_rows, "answer_len_words"),
    "sample_failures": samples,
    "per_example": rows,
}
print("<<<JSON>>>")
print(json.dumps(out, indent=2))
print("<<<END>>>")
