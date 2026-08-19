"""
CAKE CounterFact eval on GPT-J-6B (fp16), held-out gate-selection protocol.
Ports CAKE (semantic key + multi-token playback) to the 6B tier. Because the
semantic key is a MiniLM embedding (not a hidden state), NO layer retuning is
needed vs GPT-2-XL; only the base model and its dtype change. alpha is swept
lightly since W_U scale differs across models.
Usage: python eval_cf_gptj.py <model_path> <N_total> [alpha]
Emits one JSON blob between <<<JSON>>> markers.
"""
import sys, time, json, random, torch
sys.path.insert(0, "src")
from gpt2_memory_semkey import GPT2WithSemanticMemory

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
NTOT = int(sys.argv[2]) if len(sys.argv) > 2 else 100
ALPHA = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
REL_GATE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0   # relation gate (0 = off)
GATES = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]

random.seed(0)
CF = json.load(open("data/counterfact.json"))
recs = random.sample(CF, NTOT)
half = NTOT // 2
tune, test = recs[:half], recs[half:]

t0 = time.time()
g = GPT2WithSemanticMemory(MODEL, layer=0, alpha=ALPHA, n_slots_per_subkey=4096,
                           key_mode="prompt", model_dtype=torch.float16)
load_s = time.time() - t0
tok = g.tok
def tgt_ids(s): return tok(" " + s.strip(), return_tensors="pt").input_ids[0]

@torch.no_grad()
def token_acc(prompt, target, gate):
    pids = tok(prompt, return_tensors="pt").input_ids[0]
    tids = tgt_ids(target)
    full = torch.cat([pids, tids]).unsqueeze(0).to(DEV)
    logits, _, _ = g.gated_logits(full, len(pids), gate, rel_gate=REL_GATE)
    start = len(pids) - 1
    return float((logits[start:start+len(tids)].argmax(-1).cpu() == tids).float().mean())

@torch.no_grad()
def pred_first(prompt, gate):
    pids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    logits, _, _ = g.gated_logits(pids, pids.shape[1], gate, rel_gate=REL_GATE)
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
        g.mem.clear_all(); g.write_chunk(p, tn, subject=r["requested_rewrite"]["subject"])
        ES += token_acc(p, tn, gate)
        for pp in paras: PS += token_acc(pp, tn, gate); nP+=1
        for npr in neigh: NS += float(pred_first(npr, gate)==base_pred[npr]); nN+=1
    return {"ES":round(ES/len(split),4),"PS":round(PS/nP,4),"NS":round(NS/nN,4),
            "score_hm":round(hm(ES/len(split),PS/nP,NS/nN),4)}

tune_curve = {gt: eval_split(tune, gt) for gt in GATES}
best_gate = max(tune_curve.items(), key=lambda kv: kv[1]["score_hm"])[0]
test_result = eval_split(test, best_gate)

out = {"method":"CAKE-gptj","model":MODEL,"n_tune":len(tune),"n_test":len(test),
       "metric":"token-accuracy ES/PS/NS","alpha":ALPHA,"load_s":round(load_s,2),
       "selected_gate":best_gate,"rel_gate":REL_GATE,"tune_curve":tune_curve,"test":test_result,
       "gpu":torch.cuda.get_device_name(0) if DEV=="cuda" else None}
print("<<<JSON>>>"); print(json.dumps(out)); print("<<<END>>>")
