"""
Multi-hop pilot: compares the iterative hop-by-hop loop (akew_multihop.py)
against a naive single-shot baseline that just retrieves+answers the composed
multi-hop question directly, with no decomposition at all -- the
'contextual_generation' strategy from akew_answering.py applied blindly to
a question it was never designed to handle in one retrieval. This isolates
whether decomposition itself is doing real work, not just generation quality.

Usage: python akew_multihop_pilot.py [limit]
"""
import sys, json, random
sys.path.insert(0, "src")
from akew_data import load_akew
from akew_retrieval import DenseCardIndex
from akew_multihop import load_mquake_raw, answer_multihop_group
from akew_answering import answer_contextual, is_hit
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 80
MODEL_NAME = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-1.5B-Instruct"
VERIFIER_PATH = sys.argv[3] if len(sys.argv) > 3 else "outputs/akew_verifier_ckpt_v2"
MODE = "structured"

cards, golds, groups = load_akew("MQuAKE-CF", MODE)
index = DenseCardIndex()
index.build(cards)
verifier = CrossEncoder(VERIFIER_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device).eval()

raw = load_mquake_raw()
random.seed(0)
sample = random.sample(list(enumerate(raw)), min(LIMIT, len(raw)))

multihop_hits, naive_hits = [], []
examples_log = []

for i, rec in sample:
    result = answer_multihop_group(rec, index, verifier, model, tok, device)
    if result is None:
        continue
    multihop_hits.append(result["hit"])

    # naive baseline: pick the FIRST multi-hop question phrasing, retrieve once
    # against the whole card pool, generate once, no decomposition at all
    composed_q = rec.get("questions", [None])[0]
    if not composed_q:
        continue
    naive_ans = None
    cands = index.query(composed_q, topk=1)
    if not cands:
        naive_hits.append(False)
    else:
        top_card, _score = cands[0]
        naive_ans = answer_contextual(model, tok, composed_q, top_card, device)
        from akew_data import GoldRecord
        gold_obj = GoldRecord(edit_id="", target_new=rec.get("new_answer"), target_true=None,
                              aliases_new=rec.get("new_answer_alias", []), aliases_true=[])
        naive_hits.append(is_hit(naive_ans, gold_obj))

    if len(examples_log) < 5:
        examples_log.append({
            "case_id": rec.get("case_id"), "composed_question": composed_q,
            "gold_final": rec.get("new_answer"),
            "multihop_final_answer": result["final_answer"], "multihop_hit": result["hit"],
            "n_hops_attempted": result["n_hops_attempted"], "n_hops_total": result["n_hops_total"],
            "stopped_early": result["stopped_early"],
            "naive_answer": naive_ans,
            "naive_hit": naive_hits[-1] if naive_hits else None,
        })

out = {"n": len(multihop_hits), "model": MODEL_NAME, "verifier": VERIFIER_PATH,
       "accuracy": {
           "iterative_multihop": round(sum(multihop_hits) / len(multihop_hits), 4) if multihop_hits else None,
           "naive_single_shot": round(sum(naive_hits) / len(naive_hits), 4) if naive_hits else None,
       },
       "sample_examples": examples_log}
print("<<<JSON>>>")
print(json.dumps(out, indent=2))
print("<<<END>>>")
