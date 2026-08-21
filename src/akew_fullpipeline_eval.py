"""
The full non-oracle pipeline test: real retrieval -> real (v2) verifier ->
router decision (REJECT/DIRECT/REASON) -> answering strategy -> scored
against gold. Every answering-strategy result reported so far in this project
(akew_answering_results.md, akew_multihop_results.md) used ORACLE retrieval
-- the true card handed directly, deliberately isolating the answering-
strategy question from retrieval/routing noise. This is the first test of
whether the SYSTEM works end to end, not just whether contextual generation
beats playback when evidence is already known to be correct.

Compares three conditions per query:
  oracle_contextual:  true card + contextual generation (the known upper
                      bound from the Section 5 pilots, reprinted here for
                      direct comparison, not rerun)
  routed:             router.route() decides REJECT/DIRECT/REASON using REAL
                      retrieval + the v2 verifier; answering strategy follows
                      the decision (REJECT->no_context, DIRECT->hard_playback
                      on the routed card, REASON->contextual_generation on
                      the routed card)
  always_reason:      bypass the router's REJECT/DIRECT logic entirely, always
                      retrieve top-1 and run contextual_generation regardless
                      of verifier confidence -- tests whether the router's
                      gating is actually earning its complexity, or whether
                      blanket contextual generation does just as well.

Usage: python akew_fullpipeline_eval.py <dataset> <input_mode> [limit]
"""
import sys, os, json, random
sys.path.insert(0, "src")
from akew_data import load_akew
from akew_splits import subject_disjoint_split
from akew_retrieval import DenseCardIndex
from akew_router import AkewRouter
from akew_answering import answer_no_context, answer_hard_playback, answer_contextual, is_hit
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

DATASET = sys.argv[1] if len(sys.argv) > 1 else "CounterFact"
MODE = sys.argv[2] if len(sys.argv) > 2 else "unstructured"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 200
MODEL_NAME = sys.argv[4] if len(sys.argv) > 4 else "Qwen/Qwen2.5-1.5B-Instruct"
# akew_fullpipeline_results.md: MQuAKE-CF structured (82.54% live retrieval
# correctness) is the one dataset where the default 0.85 direct_threshold
# makes DIRECT fire on unreliable retrievals and lose to always-REASON
# (93.65% vs 96.83%). Optional override to test whether a stricter,
# dataset-aware threshold (rather than retraining anything) recovers the gap.
DIRECT_THRESHOLD = float(sys.argv[5]) if len(sys.argv) > 5 else 0.85
VERIFIER_PATH = "outputs/akew_verifier_ckpt_v2"
# The adaptive router is the validated-best configuration
# (akew_reliability_head_results.md: dominates both fixed policies, and makes
# byte-identical decisions on the cells where fixed gating already works, so
# it cannot degrade them). It is enabled here when the head artifact exists,
# rather than being made the AkewRouter constructor default: a library class
# should not silently load a file from disk, and an explicit opt-in at the
# call site keeps it obvious which configuration produced a given number.
# Set AKEW_RELIABILITY_HEAD="" to force the legacy fixed-gate behaviour.
HEAD_PATH = os.environ.get("AKEW_RELIABILITY_HEAD", "outputs/akew_reliability_head.json")

cards, golds, _groups = load_akew(DATASET, MODE)
_tr, _va, test = subject_disjoint_split(cards, train_frac=0.7, val_frac=0.15, seed=0)
random.seed(0)
if LIMIT and LIMIT < len(test):
    test = random.sample(test, LIMIT)

index = DenseCardIndex()
index.build(cards)   # full card pool, including OTHER groups' cards -- realistic retrieval difficulty
verifier = CrossEncoder(VERIFIER_PATH)
_head = None
if HEAD_PATH and os.path.exists(HEAD_PATH):
    from akew_reliability import ReliabilityHead
    _head = ReliabilityHead.load(HEAD_PATH)
    print(f"[pipeline] adaptive router ENABLED (head: {HEAD_PATH})", file=sys.stderr)
else:
    print(f"[pipeline] adaptive router DISABLED -- running legacy fixed gates "
          f"(no head at {HEAD_PATH!r})", file=sys.stderr)
router = AkewRouter(index, verifier, direct_threshold=DIRECT_THRESHOLD, reliability_head=_head)

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device).eval()

routed_hits, always_reason_hits = [], []
decision_counts = {"REJECT": 0, "DIRECT": 0, "REASON": 0}
correct_card_retrieved = []
examples_log = []

for c in test:
    g = golds.get(c.edit_id)
    if not g or not g.eval_question or not g.target_new:
        continue
    q = g.eval_question

    decision = router.route(q)
    decision_counts[decision.decision] += 1
    correct_card_retrieved.append(decision.card_id == c.edit_id)

    if decision.decision == "REJECT":
        ans = answer_no_context(model, tok, q, device)
    elif decision.decision == "DIRECT":
        routed_card = next((cc for cc in cards if cc.edit_id == decision.card_id), None)
        ans = answer_hard_playback(routed_card, golds.get(decision.card_id)) if routed_card else ""
    else:  # REASON
        routed_card = next((cc for cc in cards if cc.edit_id == decision.card_id), None)
        ans = answer_contextual(model, tok, q, routed_card, device) if routed_card else answer_no_context(model, tok, q, device)
    routed_hits.append(is_hit(ans, g))

    top1 = index.query(q, topk=1)
    if top1:
        always_ans = answer_contextual(model, tok, q, top1[0][0], device)
    else:
        always_ans = answer_no_context(model, tok, q, device)
    always_reason_hits.append(is_hit(always_ans, g))

    if len(examples_log) < 8:
        examples_log.append({
            "query": q, "gold": g.target_new, "router_decision": decision.decision,
            "verifier_score": decision.verifier_score, "retrieved_correct_card": decision.card_id == c.edit_id,
            "routed_answer": ans, "routed_hit": routed_hits[-1],
            "always_reason_answer": always_ans, "always_reason_hit": always_reason_hits[-1],
        })

n = len(routed_hits)
out = {
    "dataset": DATASET, "input_mode": MODE, "n": n, "model": MODEL_NAME, "verifier": "v2",
    "direct_threshold": DIRECT_THRESHOLD,
    # Stamped into every result so a number can never be read without knowing
    # which router produced it -- the two configurations differ by up to 15.9
    # points on MQuAKE-CF, so an unlabelled accuracy is ambiguous.
    "adaptive_router": bool(_head),
    "reliability_head": HEAD_PATH if _head else None,
    "accuracy": {
        "routed_full_pipeline": round(sum(routed_hits) / n, 4) if n else None,
        "always_reason_no_routing": round(sum(always_reason_hits) / n, 4) if n else None,
    },
    "router_decision_distribution": decision_counts,
    "retrieval_correctness_rate": round(sum(correct_card_retrieved) / n, 4) if n else None,
    "sample_examples": examples_log,
}
print("<<<JSON>>>")
print(json.dumps(out, indent=2))
print("<<<END>>>")
