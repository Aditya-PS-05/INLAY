"""
CounterFact single-edit eval for CAKE v2 (semantic key), same token-accuracy
metric as eval_cf.py so numbers compare directly with base/RAG/ROME/MEMIT/CAKEv1.

Sweeps the firing gate and reports the full ES/PS/NS curve so the efficacy vs
locality trade-off is explicit; also prints the best-by-score operating point.
Usage: python eval_cf_semkey.py <model_path> <N> [key_mode]   key_mode=prompt|subject|both
Emits one JSON blob between <<<JSON>>> markers.
"""
import sys, time, json, random, torch
sys.path.insert(0, "src")
from gpt2_memory_semkey import GPT2WithSemanticMemory

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
KEY_MODE = sys.argv[3] if len(sys.argv) > 3 else "prompt"
GATES = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
CAKE_LAYER, CAKE_ALPHA = 24, 10.0

random.seed(0)
CF = json.load(open("data/counterfact.json"))
recs = random.sample(CF, N)

g = GPT2WithSemanticMemory(MODEL, layer=CAKE_LAYER, alpha=CAKE_ALPHA,
                           n_slots_per_subkey=4096, key_mode=KEY_MODE)
tok = g.tok

def tgt_ids(s): return tok(" " + s.strip(), return_tensors="pt").input_ids[0]

@torch.no_grad()
def token_acc_scored(prompt, target, subject, gate):
    """token accuracy of target after prompt under the gate."""
    pids = tok(prompt, return_tensors="pt").input_ids[0]
    tids = tgt_ids(target)
    full = torch.cat([pids, tids]).unsqueeze(0).to(DEV)
    logits, sid, score = g.gated_logits(full, len(pids), gate, subject)
    start = len(pids) - 1
    preds = logits[start:start+len(tids)].argmax(-1).cpu()
    return float((preds == tids).float().mean())

@torch.no_grad()
def pred_first(prompt, subject, gate):
    pids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    logits, sid, score = g.gated_logits(pids, pids.shape[1], gate, subject)
    return int(logits[-1].argmax())

# base (memory empty) neighborhood predictions, per record, computed once
# by using gate=inf so nothing fires
def record_prompts(r):
    rw = r["requested_rewrite"]
    return rw["prompt"].format(rw["subject"]), rw["target_new"]["str"], rw["target_true"]["str"], \
           rw["subject"], r["paraphrase_prompts"], r["neighborhood_prompts"]

sweep = {}
for gate in GATES:
    ES = PS = NS = 0.0; nP = nN = 0
    for r in recs:
        p, tn, tt, subj, paras, neigh = record_prompts(r)
        base_pred = {npr: pred_first(npr, subj, 99.0) for npr in neigh}  # nothing fires
        g.mem.clear_all(); g.write_chunk(p, tn, subject=subj)
        ES += token_acc_scored(p, tn, subj, gate)
        for pp in paras:
            PS += token_acc_scored(pp, tn, subj, gate); nP += 1
        for npr in neigh:
            NS += float(pred_first(npr, subj, gate) == base_pred[npr]); nN += 1
    es, ps, ns = ES/N, PS/nP, NS/nN
    def hm(*xs):
        xs=[max(x,1e-6) for x in xs]; return len(xs)/sum(1/x for x in xs)
    sweep[gate] = {"ES": round(es,4), "PS": round(ps,4), "NS": round(ns,4), "score_hm": round(hm(es,ps,ns),4)}

best = max(sweep.items(), key=lambda kv: kv[1]["score_hm"])
result = {"method": f"CAKE-semkey({KEY_MODE})", "model": MODEL, "n": N,
          "metric": "EasyEdit native token-accuracy (ES/PS/NS)", "layer": CAKE_LAYER, "alpha": CAKE_ALPHA,
          "sweep": sweep, "best_gate": best[0], **best[1]}
print("<<<JSON>>>"); print(json.dumps(result)); print("<<<END>>>")
