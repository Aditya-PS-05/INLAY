"""
CAKE v2 semantic-key CounterFact eval with an HONEST gate-selection protocol:
select the firing gate on a TUNE split, report ES/PS/NS on a disjoint TEST split.
This removes the "gate fit on its own eval set" concern. prompt-key mode only
(the deployable one: key derives from the query text, no subject annotation needed).
Usage: python eval_cf_semkey_split.py <model_path> <N_total>
Emits one JSON blob between <<<JSON>>> markers.
"""
import sys, time, json, random, torch
sys.path.insert(0, "src")
from gpt2_memory_semkey import GPT2WithSemanticMemory

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
NTOT = int(sys.argv[2]) if len(sys.argv) > 2 else 200
GATES = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
CAKE_LAYER, CAKE_ALPHA = 24, 10.0

random.seed(0)
CF = json.load(open("data/counterfact.json"))
recs = random.sample(CF, NTOT)
half = NTOT // 2
tune, test = recs[:half], recs[half:]

g = GPT2WithSemanticMemory(MODEL, layer=CAKE_LAYER, alpha=CAKE_ALPHA,
                           n_slots_per_subkey=4096, key_mode="prompt")
tok = g.tok
def tgt_ids(s): return tok(" " + s.strip(), return_tensors="pt").input_ids[0]

@torch.no_grad()
def token_acc(prompt, target, gate):
    pids = tok(prompt, return_tensors="pt").input_ids[0]
    tids = tgt_ids(target)
    full = torch.cat([pids, tids]).unsqueeze(0).to(DEV)
    logits, _, _ = g.gated_logits(full, len(pids), gate)
    start = len(pids) - 1
    preds = logits[start:start+len(tids)].argmax(-1).cpu()
    return float((preds == tids).float().mean())

@torch.no_grad()
def pred_first(prompt, gate):
    pids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    logits, _, _ = g.gated_logits(pids, pids.shape[1], gate)
    return int(logits[-1].argmax())

def rp(r):
    rw = r["requested_rewrite"]
    return rw["prompt"].format(rw["subject"]), rw["target_new"]["str"], \
           r["paraphrase_prompts"], r["neighborhood_prompts"]

def hm(*xs):
    xs=[max(x,1e-6) for x in xs]; return len(xs)/sum(1/x for x in xs)

def eval_split(split, gate):
    ES=PS=NS=0.0; nP=nN=0
    for r in split:
        p, tn, paras, neigh = rp(r)
        base_pred = {npr: pred_first(npr, 99.0) for npr in neigh}
        g.mem.clear_all(); g.write_chunk(p, tn)
        ES += token_acc(p, tn, gate)
        for pp in paras: PS += token_acc(pp, tn, gate); nP+=1
        for npr in neigh: NS += float(pred_first(npr, gate)==base_pred[npr]); nN+=1
    return {"ES":round(ES/len(split),4),"PS":round(PS/nP,4),"NS":round(NS/nN,4),
            "score_hm":round(hm(ES/len(split),PS/nP,NS/nN),4)}

# 1) select gate on tune split
tune_curve = {gt: eval_split(tune, gt) for gt in GATES}
best_gate = max(tune_curve.items(), key=lambda kv: kv[1]["score_hm"])[0]
# 2) report on test split at the tune-selected gate
test_result = eval_split(test, best_gate)

out = {"method":"CAKE-semkey(prompt)","model":MODEL,"n_tune":len(tune),"n_test":len(test),
       "metric":"EasyEdit native token-accuracy (ES/PS/NS)","layer":CAKE_LAYER,"alpha":CAKE_ALPHA,
       "selected_gate":best_gate,"tune_curve":tune_curve,"test":test_result}
print("<<<JSON>>>"); print(json.dumps(out)); print("<<<END>>>")
