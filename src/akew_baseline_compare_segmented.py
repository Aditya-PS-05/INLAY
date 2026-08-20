"""
Follow-up to akew_baseline_compare.py's WikiUpdate unstructured result (plain
IKE 46.88% beating CAKE routed 43.75%), which looked like it contradicted
akew_cake_plus_ike_eval.py's finding that demonstrations HURT when folded
into CAKE's own REASON path (43.75% -> 40.62%). Both use the identical
random-demonstration mechanism (akew_baseline_ike.build_demonstrations,
n=2, seed=0), so the difference must be structural, not mechanism-level:
CAKE+IKE only adds demos to REASON-routed queries (leaving REJECT queries
untouched, answer_no_context in both conditions); plain IKE retrieves+demos
on EVERY query unconditionally, including the ones CAKE's router would
REJECT. This script segments plain CAKE/RAG/IKE accuracy by the router's
own REJECT vs non-REJECT decision on the identical test split, to test the
hypothesis directly rather than inferring it.

Usage: python akew_baseline_compare_segmented.py <dataset> <input_mode> [limit]
"""
import sys, json, random
sys.path.insert(0, "src")
from akew_data import load_akew
from akew_splits import subject_disjoint_split
from akew_retrieval import DenseCardIndex
from akew_router import AkewRouter
from akew_answering import answer_contextual, answer_no_context, is_hit
from akew_baseline_ike import answer_ike
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

DATASET = sys.argv[1] if len(sys.argv) > 1 else "WikiUpdate"
MODE = sys.argv[2] if len(sys.argv) > 2 else "unstructured"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 160
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

rows = []
for c in test:
    g = golds.get(c.edit_id)
    if not g or not g.eval_question or not g.target_new:
        continue
    q = g.eval_question

    decision = router.route(q)
    is_reject = (decision.decision == "REJECT")

    if is_reject:
        cake_ans = answer_no_context(model, tok, q, device)
    else:
        routed_card = next((cc for cc in cards if cc.edit_id == decision.card_id), None)
        if decision.decision == "DIRECT":
            from akew_answering import answer_hard_playback
            cake_ans = answer_hard_playback(routed_card, golds.get(decision.card_id))
        else:
            cake_ans = answer_contextual(model, tok, q, routed_card, device) if routed_card else ""
    cake_hit = is_hit(cake_ans, g)

    top1 = index.query(q, topk=1)
    if top1:
        rag_ans = answer_contextual(model, tok, q, top1[0][0], device)
        ike_ans = answer_ike(model, tok, q, top1[0][0], index, device)
    else:
        rag_ans = answer_no_context(model, tok, q, device)
        ike_ans = rag_ans
    rag_hit = is_hit(rag_ans, g)
    ike_hit = is_hit(ike_ans, g)

    rows.append({"is_reject": is_reject, "cake_hit": cake_hit, "rag_hit": rag_hit, "ike_hit": ike_hit})

def seg_acc(subset, key):
    if not subset:
        return None
    return round(sum(r[key] for r in subset) / len(subset), 4)

reject_rows = [r for r in rows if r["is_reject"]]
nonreject_rows = [r for r in rows if not r["is_reject"]]

out = {
    "dataset": DATASET, "input_mode": MODE, "n": len(rows), "model": MODEL_NAME,
    "n_reject": len(reject_rows), "n_nonreject": len(nonreject_rows),
    "overall": {
        "cake_routed": seg_acc(rows, "cake_hit"),
        "plain_rag": seg_acc(rows, "rag_hit"),
        "plain_ike": seg_acc(rows, "ike_hit"),
    },
    "reject_subset": {
        "cake_routed": seg_acc(reject_rows, "cake_hit"),
        "plain_rag": seg_acc(reject_rows, "rag_hit"),
        "plain_ike": seg_acc(reject_rows, "ike_hit"),
    },
    "nonreject_subset": {
        "cake_routed": seg_acc(nonreject_rows, "cake_hit"),
        "plain_rag": seg_acc(nonreject_rows, "rag_hit"),
        "plain_ike": seg_acc(nonreject_rows, "ike_hit"),
    },
}
print("<<<JSON>>>")
print(json.dumps(out, indent=2))
print("<<<END>>>")
