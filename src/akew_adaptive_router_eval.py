"""
Evaluates the reliability-head-driven ADAPTIVE router against the two
existing configurations, on the identical test split, in a single pass.

Conditions (all three scored on exactly the same queries, same model, same
decoding, so the comparison is like-for-like rather than three separate runs
stitched together):

  routed_fixed     the current shipped router: REJECT/DIRECT gates on fixed
                   thresholds. Helps on CounterFact/WikiUpdate, HURTS on
                   MQuAKE-CF in every mode tested.
  always_reason    both gates disabled unconditionally -- the manual bypass
                   that akew_fullpipeline_results.md concluded is the right
                   configuration for MQuAKE-CF specifically, but which gives
                   up REJECT's real measured value on WikiUpdate and DIRECT's
                   on structured CounterFact.
  routed_adaptive  the new method: the reliability head predicts whether the
                   router's own gating signal is trustworthy for THIS query,
                   and bypasses both gates when it is not.

WIN CONDITION, stated in advance so the result cannot be reinterpreted after
the fact: adaptive must (a) recover most of the MQuAKE-CF gap where fixed
gating loses badly, AND (b) not give up fixed gating's advantage on
CounterFact structured (100% vs always-REASON's 97.96%) or WikiUpdate. A
single configuration that beats or matches the better of the two existing
ones on every dataset/mode, with no per-dataset hand-tuning, is the claim.
Anything less is reported as a partial result, not rounded up.

A fourth condition, routed_threeway, is scored alongside whenever a
reject_floor is supplied. It exists because of the single cell the binary
policy lost (WikiUpdate unstructured): there the head's DETECTION was its
best anywhere (95.6% recall of bad retrievals) while the hard-coded response
was wrong, since MQuAKE-CF wants REASON on an unreliable retrieval and
WikiUpdate wants REJECT. The three-way policy uses the MAGNITUDE of predicted
unreliability to choose between those, rather than treating "unreliable" as
one undifferentiated bucket.

Usage: python akew_adaptive_router_eval.py <dataset> <input_mode> [limit] [bypass_threshold] [reject_floor]
"""
import sys, os, json, random

sys.path.insert(0, "src")
from akew_data import load_akew
from akew_splits import subject_disjoint_split
from akew_retrieval import DenseCardIndex
from akew_router import AkewRouter
from akew_reliability import ReliabilityHead
from akew_answering import answer_no_context, answer_hard_playback, answer_contextual, is_hit
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

DATASET = sys.argv[1] if len(sys.argv) > 1 else "MQuAKE-CF"
MODE = sys.argv[2] if len(sys.argv) > 2 else "structured"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 200
BYPASS_THRESHOLD = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
REJECT_FLOOR = float(sys.argv[5]) if len(sys.argv) > 5 else None
# Overridable so the head can be validated at the project's 7B scale without
# a second script. The head itself predicts RETRIEVAL correctness, which does
# not depend on the generator at all -- so the head transfers unchanged and
# only the downstream answering accuracy should move. That is a clean,
# falsifiable prediction, and the 7B run below is what tests it.
MODEL_NAME = os.environ.get("AKEW_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
VERIFIER_PATH = "outputs/akew_verifier_ckpt_v2"
HEAD_PATH = "outputs/akew_reliability_head.json"

cards, golds, _groups = load_akew(DATASET, MODE)
_tr, _va, test = subject_disjoint_split(cards, train_frac=0.7, val_frac=0.15, seed=0)
random.seed(0)
if LIMIT and LIMIT < len(test):
    test = random.sample(test, LIMIT)

index = DenseCardIndex()
index.build(cards)
verifier = CrossEncoder(VERIFIER_PATH)
head = ReliabilityHead.load(HEAD_PATH)

router_fixed = AkewRouter(index, verifier)
router_adaptive = AkewRouter(index, verifier, reliability_head=head,
                             bypass_threshold=BYPASS_THRESHOLD)
router_threeway = (AkewRouter(index, verifier, reliability_head=head,
                              bypass_threshold=BYPASS_THRESHOLD,
                              reject_floor=REJECT_FLOOR)
                   if REJECT_FLOOR is not None else None)

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device).eval()

card_by_id = {c.edit_id: c for c in cards}


def answer_for(decision, q):
    """The answering strategy each routing decision implies -- identical
    mapping to akew_fullpipeline_eval.py's, so the numbers here are directly
    comparable to every routed/always-REASON figure already reported."""
    if decision.decision == "REJECT":
        return answer_no_context(model, tok, q, device)
    routed_card = card_by_id.get(decision.card_id)
    if decision.decision == "DIRECT":
        return answer_hard_playback(routed_card, golds.get(decision.card_id)) if routed_card else ""
    return answer_contextual(model, tok, q, routed_card, device) if routed_card \
        else answer_no_context(model, tok, q, device)


fixed_hits, adaptive_hits, always_hits, threeway_hits = [], [], [], []
fixed_counts = {"REJECT": 0, "DIRECT": 0, "REASON": 0}
adaptive_counts = {"REJECT": 0, "DIRECT": 0, "REASON": 0}
threeway_counts = {"REJECT": 0, "DIRECT": 0, "REASON": 0}
bypass_fired = 0
reliabilities = []
retrieval_correct = []
# Does the head fire the bypass on exactly the queries where retrieval is
# actually wrong? Aggregate accuracy alone cannot answer that -- a head that
# bypassed at random could land the same headline number.
bypass_and_retrieval_wrong = bypass_and_retrieval_right = 0
nobypass_and_retrieval_wrong = nobypass_and_retrieval_right = 0
examples_log = []

for c in test:
    g = golds.get(c.edit_id)
    if not g or not g.eval_question or not g.target_new:
        continue
    q = g.eval_question

    d_fixed = router_fixed.route(q)
    d_adapt = router_adaptive.route(q)
    fixed_counts[d_fixed.decision] += 1
    adaptive_counts[d_adapt.decision] += 1

    is_correct_retrieval = (d_fixed.card_id == c.edit_id)
    retrieval_correct.append(is_correct_retrieval)
    if d_adapt.predicted_reliability is not None:
        reliabilities.append(d_adapt.predicted_reliability)
    did_bypass = (d_adapt.reason == "low_predicted_reliability_bypass")
    if did_bypass:
        bypass_fired += 1
        if is_correct_retrieval:
            bypass_and_retrieval_right += 1
        else:
            bypass_and_retrieval_wrong += 1
    else:
        if is_correct_retrieval:
            nobypass_and_retrieval_right += 1
        else:
            nobypass_and_retrieval_wrong += 1

    ans_fixed = answer_for(d_fixed, q)
    # Identical decision AND identical card => identical answering call, so
    # reuse rather than paying for a second deterministic generation.
    if (d_adapt.decision == d_fixed.decision) and (d_adapt.card_id == d_fixed.card_id):
        ans_adapt = ans_fixed
    else:
        ans_adapt = answer_for(d_adapt, q)

    if router_threeway is not None:
        d_three = router_threeway.route(q)
        threeway_counts[d_three.decision] += 1
        # Same reuse discipline: identical decision on the identical card is
        # the identical deterministic call, so it is not paid for twice.
        if (d_three.decision == d_adapt.decision) and (d_three.card_id == d_adapt.card_id):
            ans_three = ans_adapt
        elif (d_three.decision == d_fixed.decision) and (d_three.card_id == d_fixed.card_id):
            ans_three = ans_fixed
        else:
            ans_three = answer_for(d_three, q)
        threeway_hits.append(is_hit(ans_three, g))

    top1 = index.query(q, topk=1)
    ans_always = answer_contextual(model, tok, q, top1[0][0], device) if top1 \
        else answer_no_context(model, tok, q, device)

    fixed_hits.append(is_hit(ans_fixed, g))
    adaptive_hits.append(is_hit(ans_adapt, g))
    always_hits.append(is_hit(ans_always, g))

    if len(examples_log) < 6:
        examples_log.append({
            "query": q, "gold": g.target_new,
            "retrieval_correct": is_correct_retrieval,
            "predicted_reliability": (round(d_adapt.predicted_reliability, 4)
                                      if d_adapt.predicted_reliability is not None else None),
            "fixed_decision": d_fixed.decision, "fixed_hit": fixed_hits[-1],
            "adaptive_decision": d_adapt.decision, "adaptive_reason": d_adapt.reason,
            "adaptive_hit": adaptive_hits[-1],
            "always_reason_hit": always_hits[-1],
        })

n = len(fixed_hits)


def acc(hits):
    return round(sum(hits) / len(hits), 4) if hits else None


out = {
    "dataset": DATASET, "input_mode": MODE, "n": n, "model": MODEL_NAME,
    "bypass_threshold": BYPASS_THRESHOLD, "reject_floor": REJECT_FLOOR, "head": HEAD_PATH,
    "accuracy": {
        "routed_fixed": acc(fixed_hits),
        "routed_adaptive": acc(adaptive_hits),
        "routed_threeway": acc(threeway_hits) if threeway_hits else None,
        "always_reason": acc(always_hits),
    },
    "router_decisions": {"fixed": fixed_counts, "adaptive": adaptive_counts,
                         "threeway": threeway_counts if threeway_hits else None},
    "bypass_fired": bypass_fired,
    "bypass_rate": round(bypass_fired / n, 4) if n else None,
    "mean_predicted_reliability": round(sum(reliabilities) / len(reliabilities), 4)
    if reliabilities else None,
    "actual_retrieval_correctness": round(sum(retrieval_correct) / n, 4) if n else None,
    # The mechanism check: is the bypass FIRING WHERE IT SHOULD, or just
    # firing often enough to look right in aggregate?
    "bypass_targeting": {
        "bypass_and_retrieval_wrong": bypass_and_retrieval_wrong,
        "bypass_and_retrieval_right": bypass_and_retrieval_right,
        "nobypass_and_retrieval_wrong": nobypass_and_retrieval_wrong,
        "nobypass_and_retrieval_right": nobypass_and_retrieval_right,
        "precision_of_bypass": round(bypass_and_retrieval_wrong / bypass_fired, 4)
        if bypass_fired else None,
        "recall_of_bad_retrievals": round(
            bypass_and_retrieval_wrong / (bypass_and_retrieval_wrong + nobypass_and_retrieval_wrong), 4)
        if (bypass_and_retrieval_wrong + nobypass_and_retrieval_wrong) else None,
    },
    # Per-example hit vectors, so the comparative claims can be tested with a
    # PAIRED test rather than by eyeballing two aggregate percentages. Every
    # condition is scored on the identical queries in the same pass, so the
    # pairing is real and McNemar/paired-bootstrap are the right tools --
    # an unpaired comparison of two accuracies computed on the same items
    # would understate significance and is simply the wrong test.
    "per_example_hits": {
        "fixed": [int(h) for h in fixed_hits],
        "adaptive": [int(h) for h in adaptive_hits],
        "always_reason": [int(h) for h in always_hits],
        "threeway": [int(h) for h in threeway_hits] if threeway_hits else None,
    },
    "sample_examples": examples_log,
}
print("<<<JSON>>>")
print(json.dumps(out, indent=2))
print("<<<END>>>")
