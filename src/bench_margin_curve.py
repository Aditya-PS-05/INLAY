"""
Verify the margin gate flattens the locality-vs-N curve (not just at N=400).
Fills memory incrementally; at each checkpoint measures retention over all edits
so far + locality on a fixed control set, at margin=0 (old) and margin=0.15 (new).
Usage: python bench_margin_curve.py <model_path> <Nmax> [gate] [margin]
Emits one JSON blob between <<<JSON>>> markers.
"""
import sys, json, random, torch
sys.path.insert(0, "src")
from gpt2_memory_semkey import GPT2WithSemanticMemory

DEV="cuda" if torch.cuda.is_available() else "cpu"
MODEL=sys.argv[1] if len(sys.argv)>1 else "gpt2"
NMAX=int(sys.argv[2]) if len(sys.argv)>2 else 400
GATE=float(sys.argv[3]) if len(sys.argv)>3 else 0.45
MARGIN=float(sys.argv[4]) if len(sys.argv)>4 else 0.15
CHECK=[n for n in [1,5,10,25,50,100,200,400] if n<=NMAX]

random.seed(0)
CF=json.load(open("data/counterfact.json"))
recs=random.sample(CF,NMAX)
ctrl_recs=random.sample([r for r in CF if r not in recs],30)
controls=[r["neighborhood_prompts"][0] for r in ctrl_recs]
tok_of=lambda r:(r["requested_rewrite"]["prompt"].format(r["requested_rewrite"]["subject"]),
                 r["requested_rewrite"]["target_new"]["str"],r["requested_rewrite"]["subject"])

g=GPT2WithSemanticMemory(MODEL,layer=24,alpha=10.0,n_slots_per_subkey=4096,key_mode="prompt")
tok=g.tok
def tgt_ids(s): return tok(" "+s.strip(),return_tensors="pt").input_ids[0]
def hm(*xs):
    xs=[max(x,1e-6) for x in xs]; return len(xs)/sum(1/x for x in xs)
def ret_acc(prompt,target,margin):
    pids=tok(prompt,return_tensors="pt").input_ids[0]; tids=tgt_ids(target)
    full=torch.cat([pids,tids]).unsqueeze(0).to(DEV)
    logits,_,_=g.gated_logits(full,len(pids),GATE,margin=margin); s=len(pids)-1
    return float((logits[s:s+len(tids)].argmax(-1).cpu()==tids).float().mean())
def pred(prompt,margin):
    pids=tok(prompt,return_tensors="pt").input_ids.to(DEV)
    logits,_,_=g.gated_logits(pids,pids.shape[1],GATE,margin=margin); return int(logits[-1].argmax())

base_ctrl=[pred(cp,99.0) for cp in controls]
curves={"margin0":[],f"margin{MARGIN}":[]}
written=0
for i,r in enumerate(recs,1):
    g.write_chunk(*tok_of(r)); written=i
    if i in CHECK:
        for mg,key in [(0.0,"margin0"),(MARGIN,f"margin{MARGIN}")]:
            ret=sum(ret_acc(*tok_of(rr)[:2],mg) for rr in recs[:i])/i
            loc=sum(float(pred(cp,mg)==bp) for cp,bp in zip(controls,base_ctrl))/len(controls)
            curves[key].append({"n":i,"retention":round(ret,4),"locality":round(loc,4),"score_hm":round(hm(ret,loc),4)})

result={"method":"CAKE margin-curve","model":MODEL,"gate":GATE,"margin":MARGIN,
        "checkpoints":CHECK,"curves":curves}
print("<<<JSON>>>"); print(json.dumps(result)); print("<<<END>>>")
