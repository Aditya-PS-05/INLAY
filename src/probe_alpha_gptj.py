"""Quick alpha probe for GPT-J: single-edit ES/PS vs alpha at gate 0.45, N=20.
Finds the logit-bias scale that flips GPT-J tokens (W_U magnitude differs from GPT-2-XL).
Usage: python probe_alpha_gptj.py <model_path> <N>. Emits JSON between markers."""
import sys, json, random, torch
sys.path.insert(0, "src")
from gpt2_memory_semkey import GPT2WithSemanticMemory
DEV="cuda" if torch.cuda.is_available() else "cpu"
MODEL=sys.argv[1] if len(sys.argv)>1 else "gpt2"
N=int(sys.argv[2]) if len(sys.argv)>2 else 20
ALPHAS=[10,20,40,80,160,320]
GATE=0.45
random.seed(0)
CF=json.load(open("data/counterfact.json")); recs=random.sample(CF,N)
g=GPT2WithSemanticMemory(MODEL,layer=0,alpha=10.0,n_slots_per_subkey=4096,key_mode="prompt",model_dtype=torch.float16)
tok=g.tok
def tgt_ids(s): return tok(" "+s.strip(),return_tensors="pt").input_ids[0]
def rp(r):
    rw=r["requested_rewrite"]; return rw["prompt"].format(rw["subject"]),rw["target_new"]["str"],r["paraphrase_prompts"]
@torch.no_grad()
def token_acc(prompt,target,gate):
    pids=tok(prompt,return_tensors="pt").input_ids[0]; tids=tgt_ids(target)
    full=torch.cat([pids,tids]).unsqueeze(0).to(DEV)
    logits,_,_=g.gated_logits(full,len(pids),gate); s=len(pids)-1
    return float((logits[s:s+len(tids)].argmax(-1).cpu()==tids).float().mean())
# also report the raw W_U-bias magnitude at a fired position for diagnosis
sweep={}
for a in ALPHAS:
    g.alpha=float(a); ES=PS=0.0; nP=0
    for r in recs:
        p,tn,paras=rp(r); g.mem.clear_all(); g.write_chunk(p,tn)
        ES+=token_acc(p,tn,GATE)
        for pp in paras: PS+=token_acc(pp,tn,GATE); nP+=1
    sweep[a]={"ES":round(ES/N,4),"PS":round(PS/nP,4)}
# diagnostic: compare W_U row norm for GPT-J vs typical
wu_norm=float(g.W_U.norm(dim=1).mean())
print("<<<JSON>>>"); print(json.dumps({"model":MODEL,"n":N,"gate":GATE,"wu_row_norm_mean":round(wu_norm,3),"alpha_sweep":sweep})); print("<<<END>>>")
