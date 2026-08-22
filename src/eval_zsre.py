"""
zsRE single-edit eval for base / in_context(RAG) / INLAY-v2(semantic key) / finetune,
same token-accuracy metric as the CounterFact scripts so numbers compare across
benchmarks and against ROME/MEMIT (eval_zsre_edit.py).

zsRE record: src (question), rephrase (paraphrase), alt (NEW target answer),
answers (original), loc / loc_ans (a DIFFERENT question — locality probe).

  ES (efficacy)       = token-acc of `alt` after `src`
  PS (generalization) = token-acc of `alt` after `rephrase`
  NS (locality)       = fraction of `loc` prompts whose predicted first token is
                        UNCHANGED vs the pre-edit model
Score = harmonic mean(ES, PS, NS).

INLAY here is v2 (semantic key). It writes (src -> alt) and addresses on the
query's sentence embedding, with a firing gate. When method==inlay we sweep the
gate on a TUNE split and report on a disjoint TEST split (honest gate selection).
Usage: python eval_zsre.py <method> <model_path> <N> [gate]
Emits one JSON blob between <<<JSON>>> markers.
"""
import sys, time, json, random, torch
sys.path.insert(0, "src")
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

DEV = "cuda" if torch.cuda.is_available() else "cpu"
METHOD = sys.argv[1] if len(sys.argv) > 1 else "base"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gpt2"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
FT_STEPS = 25
INLAY_LAYER, INLAY_ALPHA = 24, 10.0
INLAY_GATES = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]

random.seed(0)
ZS = json.load(open("data/zsre.json"))
recs = random.sample(ZS, N)

tok = GPT2TokenizerFast.from_pretrained(MODEL); tok.pad_token = tok.eos_token
def tgt_ids(s): return tok(" " + s.strip(), return_tensors="pt").input_ids[0]

def token_acc(logits_fn, prompt, target, n_prompt_extra=0):
    pids = tok(prompt, return_tensors="pt").input_ids[0]
    tids = tgt_ids(target)
    full = torch.cat([pids, tids]).unsqueeze(0).to(DEV)
    logits = logits_fn(full, len(pids))
    start = len(pids) - 1
    preds = logits[start:start+len(tids)].argmax(-1).cpu()
    return float((preds == tids).float().mean())

def pred_token(logits_fn, prompt):
    pids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    ids = pids if pids.dim()==2 else pids.unsqueeze(0)
    return int(logits_fn(ids, ids.shape[1])[-1].argmax())

def rp(r): return r["src"], r["rephrase"], r["alt"], r["loc"]
def hm(*xs):
    xs=[max(x,1e-6) for x in xs]; return len(xs)/sum(1/x for x in xs)

result = {"method": METHOD, "model": MODEL, "n": N, "benchmark": "zsRE",
          "metric": "token-accuracy ES/PS/NS", "gpu": torch.cuda.get_device_name(0) if DEV=="cuda" else None}

# ---------- BASE / IN-CONTEXT ----------
if METHOD in ("base", "in_context"):
    model = GPT2LMHeadModel.from_pretrained(MODEL).to(DEV).eval()
    @torch.no_grad()
    def lf(ids, n_prompt=None): return model(input_ids=ids).logits[0]
    ES=PS=NS=0.0; nN=0
    for r in recs:
        src, reph, alt, loc = rp(r)
        pre = "" if METHOD=="base" else f"{src} {alt}. "
        base_loc = pred_token(lf, loc)
        ES += token_acc(lf, pre+src, alt)
        PS += token_acc(lf, pre+reph, alt)
        NS += float(pred_token(lf, (pre+loc) if METHOD=="in_context" else loc) == base_loc); nN+=1
    es,ps,ns = ES/N, PS/N, NS/nN
    result.update({"ES":round(es,4),"PS":round(ps,4),"NS":round(ns,4),"score_hm":round(hm(es,ps,ns),4),
                   "write_s":0.0,"grad_steps":0})

# ---------- INLAY v2 (semantic key) with held-out gate selection ----------
elif METHOD == "inlay":
    from gpt2_memory_semkey import GPT2WithSemanticMemory
    g = GPT2WithSemanticMemory(MODEL, layer=INLAY_LAYER, alpha=INLAY_ALPHA,
                               n_slots_per_subkey=4096, key_mode="prompt")
    def inlay_acc(prompt, target, gate):
        pids = tok(prompt, return_tensors="pt").input_ids[0]
        tids = tgt_ids(target)
        full = torch.cat([pids, tids]).unsqueeze(0).to(DEV)
        logits,_,_ = g.gated_logits(full, len(pids), gate)
        start = len(pids)-1
        preds = logits[start:start+len(tids)].argmax(-1).cpu()
        return float((preds==tids).float().mean())
    def inlay_pred(prompt, gate):
        pids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
        logits,_,_ = g.gated_logits(pids, pids.shape[1], gate)
        return int(logits[-1].argmax())
    def eval_split(split, gate):
        ES=PS=NS=0.0; nN=0; tw=0.0
        for r in split:
            src, reph, alt, loc = rp(r)
            base_loc = inlay_pred(loc, 99.0)
            t0=time.time(); g.mem.clear_all(); g.write_chunk(src, alt); tw+=time.time()-t0
            ES += inlay_acc(src, alt, gate)
            PS += inlay_acc(reph, alt, gate)
            NS += float(inlay_pred(loc, gate)==base_loc); nN+=1
        es,ps,ns = ES/len(split), PS/len(split), NS/nN
        return {"ES":round(es,4),"PS":round(ps,4),"NS":round(ns,4),"score_hm":round(hm(es,ps,ns),4),"write_s":round(tw,4)}
    half=N//2; tune,test=recs[:half],recs[half:]
    tune_curve={gt:eval_split(tune,gt) for gt in INLAY_GATES}
    best=max(tune_curve.items(),key=lambda kv:kv[1]["score_hm"])[0]
    test_res=eval_split(test,best)
    result.update({"selected_gate":best,"n_tune":len(tune),"n_test":len(test),
                   "tune_curve":tune_curve,"test":test_res,
                   "ES":test_res["ES"],"PS":test_res["PS"],"NS":test_res["NS"],"score_hm":test_res["score_hm"],
                   "grad_steps":0})

# ---------- FINE-TUNE ----------
elif METHOD == "finetune":
    model = GPT2LMHeadModel.from_pretrained(MODEL).to(DEV)
    orig = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    @torch.no_grad()
    def lf(ids, n_prompt=None): return model(input_ids=ids).logits[0]
    ES=PS=NS=0.0; nN=0; tw=0.0
    for r in recs:
        src, reph, alt, loc = rp(r)
        model.eval(); base_loc = pred_token(lf, loc)
        model.train(); opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
        b = tok(f"{src} {alt}", return_tensors="pt").to(DEV)
        t0=time.time()
        for _ in range(FT_STEPS):
            opt.zero_grad(); model(**b, labels=b.input_ids).loss.backward(); opt.step()
        tw+=time.time()-t0
        model.eval()
        ES += token_acc(lf, src, alt); PS += token_acc(lf, reph, alt)
        NS += float(pred_token(lf, loc)==base_loc); nN+=1
        model.load_state_dict(orig); del opt; torch.cuda.empty_cache()
    es,ps,ns = ES/N, PS/N, NS/nN
    result.update({"ES":round(es,4),"PS":round(ps,4),"NS":round(ns,4),"score_hm":round(hm(es,ps,ns),4),
                   "write_s":round(tw/N,4),"grad_steps":FT_STEPS})

print("<<<JSON>>>"); print(json.dumps(result)); print("<<<END>>>")
