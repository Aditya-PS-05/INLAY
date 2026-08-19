"""
Sequential editing stress test for ROME / MEMIT via EasyEdit (GPT-2-XL, CounterFact).
Applies N edits sequentially (sequential_edit=True, keep_original_weight=False so edits
accumulate on the SAME weights), then at each checkpoint measures retention over all
edits so far + locality, using EasyEdit's own token-accuracy on a fresh scored pass.
Run from EasyEdit/. Usage: python bench_sequential_edit.py ROME|MEMIT <Nmax>
Emits one JSON blob between <<<JSON>>> markers.
"""
import sys, time, json, random, torch, numpy as np
from easyeditor import BaseEditor, ROMEHyperParams, MEMITHyperParams

ALG=sys.argv[1] if len(sys.argv)>1 else "ROME"
NMAX=int(sys.argv[2]) if len(sys.argv)>2 else 100
CHECKPOINTS=[n for n in [1,5,10,25,50,100,200] if n<=NMAX]

random.seed(0)
CF=json.load(open("data/counterfact.json"))
recs=random.sample(CF,NMAX)
ctrl_recs=random.sample([r for r in CF if r not in recs],20)
controls=[r["neighborhood_prompts"][0] for r in ctrl_recs]

HP=ROMEHyperParams if ALG=="ROME" else MEMITHyperParams
hp=HP.from_hparams(f"./hparams/{ALG}/gpt2-xl.yaml")
editor=BaseEditor.from_hparams(hp)
tok=editor.tok; model=editor.model; DEV=next(model.parameters()).device
def tgt_ids(s): return tok(" "+s.strip(),return_tensors="pt").input_ids[0]
def prompt_of(r):
    rw=r["requested_rewrite"]; return rw["prompt"].format(rw["subject"]),rw["target_new"]["str"],rw["subject"]

@torch.no_grad()
def acc(prompt,target):
    pids=tok(prompt,return_tensors="pt").input_ids[0].to(DEV); tids=tgt_ids(target).to(DEV)
    full=torch.cat([pids,tids]).unsqueeze(0)
    s=len(pids)-1
    return float((model(input_ids=full).logits[0][s:s+len(tids)].argmax(-1)==tids).float().mean())
@torch.no_grad()
def pred(prompt):
    pids=tok(prompt,return_tensors="pt").input_ids.to(DEV); return int(model(input_ids=pids).logits[0][-1].argmax())

def hm(*xs):
    xs=[max(x,1e-6) for x in xs]; return len(xs)/sum(1/x for x in xs)

model.eval(); base_ctrl=[pred(cp) for cp in controls]
curve=[]; tw=0.0
for i,r in enumerate(recs,1):
    p,tn,subj=prompt_of(r)
    t0=time.time()
    editor.edit(prompts=[p],subject=[subj],target_new=[" "+tn],
                sequential_edit=True,keep_original_weight=False,verbose=False)
    tw+=time.time()-t0
    model=editor.model; model.eval()
    if i in CHECKPOINTS:
        ret=sum(acc(*prompt_of(rr)[:2]) for rr in recs[:i])/i
        loc=sum(float(pred(cp)==bp) for cp,bp in zip(controls,base_ctrl))/len(controls)
        curve.append({"n":i,"retention":round(ret,4),"locality":round(loc,4),
                      "score_hm":round(hm(ret,loc),4),"cum_write_s":round(tw,3)})
        print(f"[{ALG}] n={i} ret={ret:.3f} loc={loc:.3f}",flush=True)

result={"method":ALG,"model":"gpt2-xl","benchmark":"CounterFact sequential","nmax":NMAX,
        "checkpoints":CHECKPOINTS,"curve":curve}
print("<<<JSON>>>"); print(json.dumps(result)); print("<<<END>>>")
