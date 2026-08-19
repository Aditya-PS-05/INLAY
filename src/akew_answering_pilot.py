"""
Oracle-evidence answering pilot (brief section 5 oracle diagnostic #1: 'gold
card supplied directly to the generator'). Isolates the answering-STRATEGY
question (no_context vs hard_playback vs contextual_generation) from
retrieval/routing noise, by giving every strategy the TRUE card directly.
This is deliberately the first, cleanest experiment: if contextual generation
doesn't beat hard playback even with perfect retrieval, that's the more
important finding to have before adding retrieval noise on top.

Uses the SAME subject-disjoint test split the verifier's own training never
touched (akew_splits.subject_disjoint_split's third return value), for
consistency with section 4's leakage discipline even though nothing here is
trained.

Usage: python akew_answering_pilot.py <dataset> <input_mode> [limit]
"""
import sys, json, random
sys.path.insert(0, "src")
from akew_data import load_akew
from akew_splits import subject_disjoint_split
from akew_answering import answer_no_context, answer_hard_playback, answer_contextual, is_hit
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

DATASET = sys.argv[1] if len(sys.argv) > 1 else "CounterFact"
MODE = sys.argv[2] if len(sys.argv) > 2 else "structured"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 150
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

cards, golds, _groups = load_akew(DATASET, MODE)
_tr, _va, test = subject_disjoint_split(cards, train_frac=0.7, val_frac=0.15, seed=0)
random.seed(0)
if LIMIT and LIMIT < len(test):
    test = random.sample(test, LIMIT)

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device).eval()

results = {"no_context": [], "hard_playback": [], "contextual_generation": []}
examples_log = []

for c in test:
    g = golds.get(c.edit_id)
    if not g or not g.eval_question or not g.target_new:
        continue
    q = g.eval_question

    a_nc = answer_no_context(model, tok, q, device)
    a_hp = answer_hard_playback(c, g)
    a_ctx = answer_contextual(model, tok, q, c, device)

    hit_nc, hit_hp, hit_ctx = is_hit(a_nc, g), is_hit(a_hp, g), is_hit(a_ctx, g)
    results["no_context"].append(hit_nc)
    results["hard_playback"].append(hit_hp)
    results["contextual_generation"].append(hit_ctx)

    if len(examples_log) < 5:
        examples_log.append({"query": q, "gold": g.target_new,
                             "no_context": a_nc, "hard_playback": a_hp, "contextual_generation": a_ctx,
                             "hits": {"no_context": hit_nc, "hard_playback": hit_hp, "contextual_generation": hit_ctx}})

out = {"dataset": DATASET, "input_mode": MODE, "n": len(results["no_context"]), "model": MODEL_NAME,
       "accuracy": {k: round(sum(v) / len(v), 4) if v else None for k, v in results.items()},
       "sample_examples": examples_log}
print("<<<JSON>>>")
print(json.dumps(out, indent=2))
print("<<<END>>>")
