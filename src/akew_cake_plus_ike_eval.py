"""
Tests folding IKE's demonstration mechanism into CAKE's own REASON path,
directly following the finding in akew_baseline_comparison_results.md (plain
IKE beat CAKE-without-demonstrations 90.48% to 87.07%). Reuses the full
router (retrieval + v2 verifier + REJECT/DIRECT/REASON) but the REASON path
now includes 2 IKE-style demonstrations, on top of the retrieval/
verification/REJECT machinery plain IKE structurally lacks.

Usage: python akew_cake_plus_ike_eval.py <dataset> <input_mode> [limit]
"""
import sys, json, random
sys.path.insert(0, "src")
from akew_data import load_akew
from akew_splits import subject_disjoint_split
from akew_retrieval import DenseCardIndex
from akew_router import AkewRouter
from akew_answering import answer_contextual, answer_no_context, answer_hard_playback, is_hit
from akew_baseline_ike import build_demonstrations
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

DATASET = sys.argv[1] if len(sys.argv) > 1 else "CounterFact"
MODE = sys.argv[2] if len(sys.argv) > 2 else "unstructured"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 150
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
VERIFIER_PATH = "outputs/akew_verifier_ckpt_v2"

cards, golds, _groups = load_akew(DATASET, MODE)
_tr, _va, test = subject_disjoint_split(cards, train_frac=0.7, val_frac=0.15, seed=0)
random.seed(0)
if LIMIT and LIMIT < len(test):
    test = random.sample(test, LIMIT)

index = DenseCardIndex()
index.build(cards)
verifier = CrossEncoder(VERIFIER_PATH)
router = AkewRouter(index, verifier)

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device).eval()

hits, hits_no_demo = [], []
decision_counts = {"REJECT": 0, "DIRECT": 0, "REASON": 0}
examples_log = []

for c in test:
    g = golds.get(c.edit_id)
    if not g or not g.eval_question or not g.target_new:
        continue
    q = g.eval_question
    decision = router.route(q)
    decision_counts[decision.decision] += 1

    routed_card = next((cc for cc in cards if cc.edit_id == decision.card_id), None) if decision.card_id else None

    if decision.decision == "REJECT":
        ans = answer_no_context(model, tok, q, device)
        ans_no_demo = ans
    elif decision.decision == "DIRECT":
        ans = answer_hard_playback(routed_card, golds.get(decision.card_id))
        ans_no_demo = ans
    else:  # REASON
        demos = build_demonstrations(index, decision.card_id, n=2, seed=0)
        ans = answer_contextual(model, tok, q, routed_card, device, demonstrations=demos) if routed_card else answer_no_context(model, tok, q, device)
        ans_no_demo = answer_contextual(model, tok, q, routed_card, device) if routed_card else answer_no_context(model, tok, q, device)

    hits.append(is_hit(ans, g))
    hits_no_demo.append(is_hit(ans_no_demo, g))

    if len(examples_log) < 6:
        examples_log.append({"query": q, "gold": g.target_new, "decision": decision.decision,
                             "with_demo": ans, "with_demo_hit": hits[-1],
                             "no_demo": ans_no_demo, "no_demo_hit": hits_no_demo[-1]})

n = len(hits)
out = {"dataset": DATASET, "input_mode": MODE, "n": n, "model": MODEL_NAME,
       "accuracy": {
           "cake_routed_plus_ike_demos": round(sum(hits) / n, 4) if n else None,
           "cake_routed_no_demos": round(sum(hits_no_demo) / n, 4) if n else None,
       },
       "router_decision_distribution": decision_counts,
       "sample_examples": examples_log}
print("<<<JSON>>>")
print(json.dumps(out, indent=2))
print("<<<END>>>")
