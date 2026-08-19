"""
CounterFact single-edit protocol for base / in_context(RAG) / CAKE / finetune,
scored with EasyEdit's OWN metric (teacher-forcing token accuracy) so the numbers
are directly comparable to the ROME/MEMIT columns from eval_cf_edit.py:

  ES (efficacy)      = token-acc of target_new on the edit prompt
  PS (generalization)= token-acc of target_new on a paraphrase prompt
  NS (locality)      = fraction of neighborhood prompts whose predicted answer is
                       UNCHANGED vs the pre-edit model (specificity)

Score = harmonic mean(ES, PS, NS). One fact installed per record (single-edit).
Usage: python eval_cf.py <method> <model_path> <N> [cake_layer cake_alpha]
Emits one JSON blob between <<<JSON>>> markers.
"""
import sys, time, json, random, torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from gpt2_memory import GPT2WithMemory

DEV = "cuda" if torch.cuda.is_available() else "cpu"
METHOD = sys.argv[1] if len(sys.argv) > 1 else "base"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gpt2"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
CAKE_LAYER = int(sys.argv[4]) if len(sys.argv) > 4 else 24
CAKE_ALPHA = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
CAKE_GATE = 0.9
FT_STEPS = 25

random.seed(0)
CF = json.load(open("data/counterfact.json"))
recs = random.sample(CF, N)

tok = GPT2TokenizerFast.from_pretrained(MODEL); tok.pad_token = tok.eos_token

def tgt_ids(s):
    return tok(" " + s.strip(), return_tensors="pt").input_ids[0]  # (L,)

def token_acc(logits_fn, prompt, target):
    """Teacher-forcing token accuracy of `target` after `prompt`, EasyEdit-style:
    feed prompt+target, at each target position check argmax == gold token.
    logits_fn(full_ids, n_prompt_tokens)->(T,vocab). n_prompt_tokens lets a
    method (CAKE) address its memory on the PROMPT portion only, not the whole
    teacher-forced sequence."""
    pids = tok(prompt, return_tensors="pt").input_ids[0]
    tids = tgt_ids(target)
    full = torch.cat([pids, tids]).unsqueeze(0).to(DEV)
    logits = logits_fn(full, len(pids))            # (T, vocab)
    start = len(pids) - 1
    preds = logits[start:start+len(tids)].argmax(-1).cpu()
    return float((preds == tids).float().mean())

def pred_token(logits_fn, prompt):
    pids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    ids = pids if pids.dim()==2 else pids.unsqueeze(0)
    return int(logits_fn(ids, ids.shape[1])[-1].argmax())

def record_prompts(r):
    rw = r["requested_rewrite"]
    p = rw["prompt"].format(rw["subject"])
    return p, rw["target_new"]["str"], rw["target_true"]["str"], \
           r["paraphrase_prompts"], r["neighborhood_prompts"]

result = {"method": METHOD, "model": MODEL, "n": N,
          "metric": "EasyEdit native token-accuracy (ES=rewrite, PS=rephrase, NS=locality)",
          "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else None}
ES = PS = NS = 0.0; nP = nN = 0

# ---------- BASE / IN-CONTEXT ----------
if METHOD in ("base", "in_context"):
    model = GPT2LMHeadModel.from_pretrained(MODEL).to(DEV).eval()
    @torch.no_grad()
    def lf(ids, n_prompt=None): return model(input_ids=ids).logits[0]
    for r in recs:
        p, tn, tt, paras, neigh = record_prompts(r)
        pre = "" if METHOD == "base" else f"{p} {tn}. "
        base_pred = {npr: pred_token(lf, npr) for npr in neigh}  # pre-edit == base here
        ES += token_acc(lf, pre + p, tn)
        for pp in paras:
            PS += token_acc(lf, pre + pp, tn); nP += 1
        for npr in neigh:
            NS += float(pred_token(lf, (pre+npr) if METHOD=="in_context" else npr) == base_pred[npr]); nN += 1
    result["write_s"] = 0.0; result["grad_steps"] = 0

# ---------- CAKE ----------
elif METHOD == "cake":
    g = GPT2WithMemory(MODEL, layer=CAKE_LAYER, alpha=CAKE_ALPHA, n_slots_per_subkey=4096, topk=1)
    twrite = 0.0
    @torch.no_grad()
    def base_lf(ids, n_prompt=None): g.set_read(False); return g.model(input_ids=ids).logits[0]
    @torch.no_grad()
    def cake_lf(ids, n_prompt):
        """Gated CAKE logits, teacher-forcing-aware. Address the memory using the
        PROMPT's last-token h_L (position n_prompt-1) — NOT the last token of the
        teacher-forced target — because CAKE keys on the prompt. If the top slot
        fires >= gate, add alpha*<W_U,v> to every position's logits; else base."""
        # 1) address on prompt only
        g.set_read(True); g._inject_budget = 0
        g.model(input_ids=ids[:, :n_prompt])
        fired = g._last_fired
        if fired and isinstance(fired[0], (list, tuple)) and fired[0] and isinstance(fired[0][0], (list, tuple)):
            fired = fired[0]
        top = fired[0] if fired else None
        v = g._last_value
        # 2) full forward for logits (read off so it doesn't re-address on target)
        g.set_read(False); g._inject_budget = None
        out = g.model(input_ids=ids).logits[0]     # (T,vocab)
        if top is None or top[1] < CAKE_GATE or v is None:
            return out
        if v.dim() == 1: v = v.unsqueeze(0)
        return out + g.alpha * (v @ g.W_U.T)        # broadcast answer-token bias
    for r in recs:
        p, tn, tt, paras, neigh = record_prompts(r)
        base_pred = {npr: pred_token(base_lf, npr) for npr in neigh}
        t0 = time.time(); g.mem.clear_all(); g.write_chunk(p, tn); twrite += time.time()-t0
        ES += token_acc(cake_lf, p, tn)
        for pp in paras:
            PS += token_acc(cake_lf, pp, tn); nP += 1
        for npr in neigh:
            NS += float(pred_token(cake_lf, npr) == base_pred[npr]); nN += 1
    g.close()
    result["write_s"] = round(twrite, 4); result["grad_steps"] = 0
    result["cake_layer"] = CAKE_LAYER; result["cake_alpha"] = CAKE_ALPHA; result["gate"] = CAKE_GATE

# ---------- FINE-TUNE (per-record single edit, weight reset) ----------
elif METHOD == "finetune":
    model = GPT2LMHeadModel.from_pretrained(MODEL).to(DEV)
    orig = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def lf(ids, n_prompt=None): return model(input_ids=ids).logits[0]
    twrite = 0.0
    for r in recs:
        p, tn, tt, paras, neigh = record_prompts(r)
        model.eval()
        base_pred = {npr: pred_token(lf, npr) for npr in neigh}
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
        b = tok(f"{p} {tn}", return_tensors="pt").to(DEV)
        t0 = time.time()
        for _ in range(FT_STEPS):
            opt.zero_grad(); model(**b, labels=b.input_ids).loss.backward(); opt.step()
        twrite += time.time()-t0
        model.eval()
        ES += token_acc(lf, p, tn)
        for pp in paras:
            PS += token_acc(lf, pp, tn); nP += 1
        for npr in neigh:
            NS += float(pred_token(lf, npr) == base_pred[npr]); nN += 1
        model.load_state_dict(orig); del opt; torch.cuda.empty_cache()
    result["write_s"] = round(twrite/N, 4); result["grad_steps"] = FT_STEPS

es, ps, ns = ES/N, PS/nP, NS/nN
def hm(*xs):
    xs = [max(x, 1e-6) for x in xs]; return round(len(xs)/sum(1/x for x in xs), 4)
result.update({"ES": round(es,4), "PS": round(ps,4), "NS": round(ns,4), "score_hm": hm(es,ps,ns)})
print("<<<JSON>>>"); print(json.dumps(result)); print("<<<END>>>")
