"""
Sequential / batch editing stress test on CounterFact (GPT-2-XL).

The regime where CAKE's structure should matter: write N facts one after another,
then measure how many of ALL N are still correct (retention) plus locality. Weight
editors (ROME single-layer sequential) accumulate interference as N grows; CAKE
writes each fact to a SEPARATE memory slot with zero weight change, so retention
should stay flat.

Metrics at each checkpoint N (evaluated over all edits made so far):
  retention (ES)  = mean token-acc of every edited fact's target after its prompt
  locality (NS)   = fraction of a fixed control set whose first-token pred is
                    unchanged vs the pre-edit base model
  score           = harmonic mean(retention, locality)

CAKE only here (base/RAG/finetune added by comparison; ROME/MEMIT via
bench_sequential_edit.py). Emits one JSON blob between <<<JSON>>> markers.
Usage: python bench_sequential.py <method> <model_path> <Nmax>
"""
import sys, time, json, random, torch
sys.path.insert(0, "src")
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

DEV = "cuda" if torch.cuda.is_available() else "cpu"
METHOD = sys.argv[1] if len(sys.argv) > 1 else "cake"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gpt2"
NMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 200
CHECKPOINTS = [1, 5, 10, 25, 50, 100, 200, 400]
CHECKPOINTS = [n for n in CHECKPOINTS if n <= NMAX]
CAKE_LAYER, CAKE_ALPHA, CAKE_GATE = 24, 10.0, 0.45
FT_STEPS = 20

random.seed(0)
CF = json.load(open("data/counterfact.json"))
recs = random.sample(CF, NMAX)
# fixed control set: neighborhood prompts from a DISJOINT set of records
ctrl_recs = random.sample([r for r in CF if r not in recs], 20)
controls = [r["neighborhood_prompts"][0] for r in ctrl_recs]

tok = GPT2TokenizerFast.from_pretrained(MODEL); tok.pad_token = tok.eos_token
def tgt_ids(s): return tok(" " + s.strip(), return_tensors="pt").input_ids[0]
def prompt_of(r):
    rw=r["requested_rewrite"]; return rw["prompt"].format(rw["subject"]), rw["target_new"]["str"], rw["subject"]

def hm(*xs):
    xs=[max(x,1e-6) for x in xs]; return len(xs)/sum(1/x for x in xs)

result={"method":METHOD,"model":MODEL,"benchmark":"CounterFact sequential","nmax":NMAX,
        "checkpoints":CHECKPOINTS,"curve":[], "gpu":torch.cuda.get_device_name(0) if DEV=="cuda" else None}

# ---------------- CAKE (multi-token value, semantic key) ----------------
if METHOD=="cake":
    from gpt2_memory_semkey import GPT2WithSemanticMemory
    g=GPT2WithSemanticMemory(MODEL,layer=CAKE_LAYER,alpha=CAKE_ALPHA,n_slots_per_subkey=4096,key_mode="prompt")
    def cake_acc(prompt,target):
        pids=tok(prompt,return_tensors="pt").input_ids[0]; tids=tgt_ids(target)
        full=torch.cat([pids,tids]).unsqueeze(0).to(DEV)
        logits,_,_=g.gated_logits(full,len(pids),CAKE_GATE)
        s=len(pids)-1; return float((logits[s:s+len(tids)].argmax(-1).cpu()==tids).float().mean())
    def cake_pred(prompt):
        pids=tok(prompt,return_tensors="pt").input_ids.to(DEV)
        logits,_,_=g.gated_logits(pids,pids.shape[1],CAKE_GATE); return int(logits[-1].argmax())
    base_ctrl=[cake_pred(cp) for cp in controls]  # memory empty
    tw=0.0
    for i,r in enumerate(recs,1):
        p,tn,subj=prompt_of(r); t0=time.time(); g.write_chunk(p,tn,subject=subj); tw+=time.time()-t0
        if i in CHECKPOINTS:
            ret=sum(cake_acc(*prompt_of(rr)[:2]) for rr in recs[:i])/i
            loc=sum(float(cake_pred(cp)==bp) for cp,bp in zip(controls,base_ctrl))/len(controls)
            result["curve"].append({"n":i,"retention":round(ret,4),"locality":round(loc,4),
                                    "score_hm":round(hm(ret,loc),4),"cum_write_s":round(tw,3)})

# ---------------- base / in_context (no accumulation; sanity floor/ceiling) --------
elif METHOD in ("base","in_context"):
    model=GPT2LMHeadModel.from_pretrained(MODEL).to(DEV).eval()
    @torch.no_grad()
    def acc(prompt,target):
        pids=tok(prompt,return_tensors="pt").input_ids[0]; tids=tgt_ids(target)
        full=torch.cat([pids,tids]).unsqueeze(0).to(DEV)
        s=len(pids)-1; return float((model(input_ids=full).logits[0][s:s+len(tids)].argmax(-1).cpu()==tids).float().mean())
    @torch.no_grad()
    def pred(prompt):
        pids=tok(prompt,return_tensors="pt").input_ids.to(DEV); return int(model(input_ids=pids).logits[0][-1].argmax())
    base_ctrl=[pred(cp) for cp in controls]
    for i in CHECKPOINTS:
        if METHOD=="base":
            ret=sum(acc(*prompt_of(rr)[:2]) for rr in recs[:i])/i; loc=1.0
        else:  # in_context: prepend each fact to its own prompt
            ret=sum(acc(f"{prompt_of(rr)[0]} {prompt_of(rr)[1]}. {prompt_of(rr)[0]}",prompt_of(rr)[1]) for rr in recs[:i])/i
            loc=sum(float(pred(cp)==bp) for cp,bp in zip(controls,base_ctrl))/len(controls)
        result["curve"].append({"n":i,"retention":round(ret,4),"locality":round(loc,4),"score_hm":round(hm(ret,loc),4),"cum_write_s":0.0})

# ---------------- fine-tune (sequential, accumulating) ----------------
elif METHOD=="finetune":
    model=GPT2LMHeadModel.from_pretrained(MODEL).to(DEV)
    @torch.no_grad()
    def acc(prompt,target):
        pids=tok(prompt,return_tensors="pt").input_ids[0]; tids=tgt_ids(target)
        full=torch.cat([pids,tids]).unsqueeze(0).to(DEV)
        s=len(pids)-1; return float((model(input_ids=full).logits[0][s:s+len(tids)].argmax(-1).cpu()==tids).float().mean())
    @torch.no_grad()
    def pred(prompt):
        pids=tok(prompt,return_tensors="pt").input_ids.to(DEV); return int(model(input_ids=pids).logits[0][-1].argmax())
    model.eval(); base_ctrl=[pred(cp) for cp in controls]
    opt=torch.optim.AdamW(model.parameters(),lr=5e-5); tw=0.0
    for i,r in enumerate(recs,1):
        p,tn,_=prompt_of(r); b=tok(f"{p} {tn}",return_tensors="pt").to(DEV)
        model.train(); t0=time.time()
        for _ in range(FT_STEPS):
            opt.zero_grad(); model(**b,labels=b.input_ids).loss.backward(); opt.step()
        tw+=time.time()-t0; model.eval()
        if i in CHECKPOINTS:
            ret=sum(acc(*prompt_of(rr)[:2]) for rr in recs[:i])/i
            loc=sum(float(pred(cp)==bp) for cp,bp in zip(controls,base_ctrl))/len(controls)
            result["curve"].append({"n":i,"retention":round(ret,4),"locality":round(loc,4),"score_hm":round(hm(ret,loc),4),"cum_write_s":round(tw,3)})

print("<<<JSON>>>"); print(json.dumps(result)); print("<<<END>>>")
