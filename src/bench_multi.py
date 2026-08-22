"""
Multi-subject benchmark: N facts, each about a DIFFERENT entity (the case ROME/
MEMIT are designed for). Reads facts from facts24.json. Three axes, standard in
knowledge editing:
  efficacy       = fraction of facts recalled on the EXACT prompt
  generalization = fraction recalled on a PARAPHRASED query
  locality       = fraction of unrelated control prompts unchanged vs base

Runs base / in_context / finetune / inlay in this process.
ROME/MEMIT are run separately by bench_multi_edit.py (needs EasyEdit env).
Usage: python bench_multi.py <model> <inlay_layer> <inlay_alpha> <n_sub>
Emits one JSON blob between <<<JSON>>> markers.
"""
import time, json, sys, torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from gpt2_memory import GPT2WithMemory

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
INLAY_LAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 6
INLAY_ALPHA = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
INLAY_NSUB  = int(sys.argv[4]) if len(sys.argv) > 4 else 256
INLAY_MINSCORE = 0.9

D = json.load(open("facts24.json"))
FACTS, CONTROL = D["facts"], D["control"]
# each fact: [subject, prompt, target(lead space), first_tok, paraphrase]
DOC = " ".join(f"{p.strip()}{t}." for _, p, t, _, _ in FACTS)

tok = GPT2TokenizerFast.from_pretrained(MODEL); tok.pad_token = tok.eos_token

def greedy(model, prompt, n=6):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()

def score_gen(gen_fn, use_para=False):
    ok = 0
    for subj, p, t, ft, para in FACTS:
        q = para if use_para else p
        ans = t.strip()
        if ans.lower() in gen_fn(q).lower():
            ok += 1
    return round(ok / len(FACTS), 3)

result = {"device": DEV, "model": MODEL, "n_facts": len(FACTS),
          "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else None}

# ---- base ----
base = GPT2LMHeadModel.from_pretrained(MODEL).to(DEV).eval()
base_ctrl = {p: greedy(base, p, 4) for p in CONTROL}
result["base"] = {"efficacy": score_gen(lambda q: greedy(base, q)),
                  "generalization": score_gen(lambda q: greedy(base, q), True),
                  "locality": 1.0, "write_s": 0.0, "grad_steps": 0}

# ---- in-context (RAG): prepend whole doc ----
def ic(q): return greedy(base, DOC + "\n" + q)
result["in_context"] = {"efficacy": score_gen(ic), "generalization": score_gen(ic, True),
                        "locality": 1.0, "write_s": 0.0, "grad_steps": 0,
                        "note": "whole %d-fact doc re-fed every query" % len(FACTS)}
del base; torch.cuda.empty_cache()

# ---- fine-tune ----
ft = GPT2LMHeadModel.from_pretrained(MODEL).to(DEV).train()
opt = torch.optim.AdamW(ft.parameters(), lr=5e-5)
train = [tok(f"{p.strip()}{t}", return_tensors="pt").to(DEV) for _, p, t, _, _ in FACTS]
t0 = time.time()
for _ in range(60):
    opt.zero_grad(); loss = 0.0
    for b in train: loss = loss + ft(**b, labels=b.input_ids).loss
    (loss/len(train)).backward(); opt.step()
ftw = time.time()-t0; ft.eval()
result["finetune"] = {"efficacy": score_gen(lambda q: greedy(ft, q)),
                      "generalization": score_gen(lambda q: greedy(ft, q), True),
                      "locality": round(sum(greedy(ft,p,4)==base_ctrl[p] for p in CONTROL)/len(CONTROL),3),
                      "write_s": round(ftw,3), "grad_steps": 60}
del ft, opt; torch.cuda.empty_cache()

# ---- INLAY ----
g = GPT2WithMemory(MODEL, layer=INLAY_LAYER, alpha=INLAY_ALPHA, n_slots_per_subkey=INLAY_NSUB, topk=1)
t0 = time.time()
for _, p, t, _, _ in FACTS:
    g.write_chunk(p, t.strip())
cw = time.time()-t0
g.set_read(True)
eff = score_gen(lambda q: g.answer_playback(q, max_new_tokens=6, min_score=INLAY_MINSCORE)[0])
gen = score_gen(lambda q: g.answer_playback(q, max_new_tokens=6, min_score=INLAY_MINSCORE)[0], True)
g.set_read(False)
bc = {p: g.answer(p, max_new_tokens=4) for p in CONTROL}
g.set_read(True)
loc = sum(g.answer_playback(p, max_new_tokens=4, min_score=INLAY_MINSCORE)[0]==bc[p] for p in CONTROL)
result["inlay"] = {"efficacy": eff, "generalization": gen,
                  "locality": round(loc/len(CONTROL),3),
                  "write_s": round(cw,3), "grad_steps": 0, "min_score": INLAY_MINSCORE}
g.close()

print("<<<JSON>>>"); print(json.dumps(result)); print("<<<END>>>")
