"""
Compares three methods on the same sample, real (non-oracle) retrieval
throughout: INLAY's routed pipeline (retrieval + v2 verifier + router),
plain RAG (retrieve top-1, inject as context, generate -- no demonstrations,
no gating), and IKE (RAG + demonstration examples showing the override
pattern). Brief section 6, methods 2-8 (INLAY variants, already covered by
akew_fullpipeline_eval.py's routed condition) vs methods 9 (RAG) and 10
(IKE) -- weight-editing methods (1, 3-8 partially, 11) remain out of scope,
see akew_fullpipeline_results.md's scope note.

Usage: python akew_baseline_compare.py <dataset> <input_mode> [limit]
"""
import sys, json, random
sys.path.insert(0, "src")
from akew_data import load_akew
from akew_splits import subject_disjoint_split
from akew_retrieval import DenseCardIndex
from akew_router import AkewRouter
from akew_answering import answer_contextual, is_hit
from akew_baseline_ike import answer_ike
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


inlay_hits, rag_hits, ike_hits = [], [], []
examples_log = []

for c in test:
    g = golds.get(c.edit_id)
    if not g or not g.eval_question or not g.target_new:
        continue
    q = g.eval_question

    decision = router.route(q)
    if decision.decision == "REJECT":
        from akew_answering import answer_no_context
        inlay_ans = answer_no_context(model, tok, q, device)
    else:
        routed_card = next((cc for cc in cards if cc.edit_id == decision.card_id), None)
        if decision.decision == "DIRECT":
            from akew_answering import answer_hard_playback
            inlay_ans = answer_hard_playback(routed_card, golds.get(decision.card_id))
        else:
            inlay_ans = answer_contextual(model, tok, q, routed_card, device) if routed_card else ""
    inlay_hits.append(is_hit(inlay_ans, g))

    top1 = index.query(q, topk=1)
    if top1:
        rag_ans = answer_contextual(model, tok, q, top1[0][0], device)
        ike_ans = answer_ike(model, tok, q, top1[0][0], index, device)
    else:
        from akew_answering import answer_no_context
        rag_ans = answer_no_context(model, tok, q, device)
        ike_ans = rag_ans
    rag_hits.append(is_hit(rag_ans, g))
    ike_hits.append(is_hit(ike_ans, g))

    if len(examples_log) < 6:
        examples_log.append({"query": q, "gold": g.target_new,
                             "inlay_routed": inlay_ans, "inlay_hit": inlay_hits[-1],
                             "rag": rag_ans, "rag_hit": rag_hits[-1],
                             "ike": ike_ans, "ike_hit": ike_hits[-1]})

n = len(inlay_hits)
out = {"dataset": DATASET, "input_mode": MODE, "n": n, "model": MODEL_NAME,
       "accuracy": {
           "inlay_routed_pipeline": round(sum(inlay_hits) / n, 4) if n else None,
           "plain_rag": round(sum(rag_hits) / n, 4) if n else None,
           "ike_with_demonstrations": round(sum(ike_hits) / n, 4) if n else None,
       },
       "sample_examples": examples_log}
print("<<<JSON>>>")
print(json.dumps(out, indent=2))
print("<<<END>>>")
