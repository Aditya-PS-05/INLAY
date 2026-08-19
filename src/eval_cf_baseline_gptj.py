"""
base / in_context(RAG) CounterFact single-edit on GPT-J-6B (fp16), token-accuracy
metric (ES=rewrite, PS=rephrase, NS=locality) identical to eval_cf_gptj.py and the
ROME columns. Provides the floor (base) and RAG ceiling for the GPT-J comparison.
Usage: python eval_cf_baseline_gptj.py base|in_context <model_path> <N>
Emits one JSON blob between <<<JSON>>> markers.
"""
import sys, json, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
METHOD = sys.argv[1] if len(sys.argv) > 1 else "base"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gpt2"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 100

random.seed(0)
CF = json.load(open("data/counterfact.json"))
recs = random.sample(CF, N)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()

def tgt_ids(s): return tok(" " + s.strip(), return_tensors="pt").input_ids[0]
@torch.no_grad()
def token_acc(prompt, target):
    pids = tok(prompt, return_tensors="pt").input_ids[0]; tids = tgt_ids(target)
    full = torch.cat([pids, tids]).unsqueeze(0).to(DEV)
    s = len(pids)-1
    return float((model(input_ids=full).logits[0][s:s+len(tids)].argmax(-1).cpu() == tids).float().mean())
@torch.no_grad()
def pred_first(prompt):
    pids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    return int(model(input_ids=pids).logits[0][-1].argmax())

def rp(r):
    rw = r["requested_rewrite"]
    return rw["prompt"].format(rw["subject"]), rw["target_new"]["str"], \
           r["paraphrase_prompts"], r["neighborhood_prompts"]
def hm(*xs):
    xs=[max(x,1e-6) for x in xs]; return len(xs)/sum(1/x for x in xs)

ES=PS=NS=0.0; nP=nN=0
for r in recs:
    p, tn, paras, neigh = rp(r)
    if METHOD == "in_context":
        pre = f"{p} {tn}. "
        ES += token_acc(pre + p, tn)
        for pp in paras: PS += token_acc(pre + pp, tn); nP+=1
        # locality: RAG only supplies the fact; controls get the same prefix -> unchanged if answer differs
        for npr in neigh:
            base = pred_first(npr); NS += float(pred_first(pre + npr) == base); nN+=1
    else:  # base
        ES += token_acc(p, tn)
        for pp in paras: PS += token_acc(pp, tn); nP+=1
        NS += 1.0; nN+=1  # base changes nothing -> locality 1 by definition (measured against itself)
        for _ in neigh[1:]: NS += 1.0; nN+=1

es=ES/N; ps=PS/nP if nP else 0; ns=NS/nN if nN else 0
result = {"method": METHOD, "model": MODEL, "n": N, "ES": round(es,4), "PS": round(ps,4),
          "NS": round(ns,4), "score_hm": round(hm(es,ps,ns),4)}
print("<<<JSON>>>"); print(json.dumps(result)); print("<<<END>>>")
