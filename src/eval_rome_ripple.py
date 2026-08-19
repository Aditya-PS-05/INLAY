
"""ROME on native RippleEdits (GPT-J-6B) via EasyEdit edit() + generation scoring.
Snapshots only the edited fc_out layer weights to CPU and restores them between edits
(full-model GPU clone OOMs). Same 6-criterion generation protocol as eval_rippleedits.py.
Usage: python eval_rome_ripple.py <N>. Emits JSON between markers."""
import sys, json, random, torch
from easyeditor import BaseEditor, ROMEHyperParams
N=int(sys.argv[1]) if len(sys.argv)>1 else 100
MAXNEW=24
CRIT=["Logical_Generalization","Compositionality_I","Compositionality_II","Subject_Aliasing","Relation_Specificity","Forgetfulness"]
PROP={"Logical_Generalization","Compositionality_I","Compositionality_II","Subject_Aliasing"}
random.seed(0)
data=json.load(open("../RippleEdits/popular.json")); random.shuffle(data)
hp=ROMEHyperParams.from_hparams("./hparams/ROME/gpt-j-6B.yaml"); hp.device=0
editor=BaseEditor.from_hparams(hp)
model=editor.model; tok=editor.tok
if tok.pad_token is None: tok.pad_token=tok.eos_token
wnames=[f"transformer.h.{L}.mlp.fc_out.weight" for L in hp.layers]
_sd=model.state_dict()
orig_state={k:_sd[k].detach().cpu().clone() for k in wnames}

@torch.no_grad()
def restore():
    sd=model.state_dict()
    for k,v in orig_state.items(): sd[k].copy_(v.to(sd[k].device))

@torch.no_grad()
def gen(prompt):
    ids=tok(prompt,return_tensors="pt").to("cuda")
    out=model.generate(**ids,max_new_tokens=MAXNEW,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(out[0,ids.input_ids.shape[1]:],skip_special_tokens=True)

def hit(g,answers):
    gl=g.lower()
    for a in answers:
        for c in [a.get("value","")]+a.get("aliases",[]):
            if c and len(c)>=2 and c.lower() in gl: return True
    return False

def grp_score(gp):
    tq=gp.get("test_queries",[])
    if not tq: return None
    res=[hit(gen(q["prompt"]),q.get("answers",[])) for q in tq]
    return (all(res) if gp.get("test_condition","AND")=="AND" else any(res))

per={c:[] for c in CRIT}; nused=0
for ex in data:
    if nused>=N: break
    edit=ex["edit"]; p=edit["prompt"]
    if " is " not in p: continue
    ctx,tgt=p.rsplit(" is ",1); prompt=ctx+" is"; tgt=tgt.rstrip(".")
    subj=None
    if " of " in prompt and prompt.endswith(" is"):
        subj=prompt.split(" of ")[-1][:-3].strip()
    if not subj: continue
    try:
        editor.edit(prompts=[prompt],subject=[subj],target_new=[" "+tgt],
                    sequential_edit=False,keep_original_weight=False,verbose=False)
    except Exception:
        restore(); continue
    usedany=False
    for c in CRIT:
        sc=[grp_score(gp) for gp in ex.get(c,[])]; sc=[s for s in sc if s is not None]
        if sc: per[c].append(sum(sc)/len(sc)); usedany=True
    if usedany: nused+=1
    restore()

def acc(c): return round(sum(per[c])/len(per[c]),4) if per[c] else None
ca={c:acc(c) for c in CRIT}
pv=[ca[c] for c in CRIT if c in PROP and ca[c] is not None]
sv=[ca[c] for c in CRIT if c not in PROP and ca[c] is not None]
al=[v for v in ca.values() if v is not None]
print("<<<JSON>>>");print(json.dumps({"method":"ROME","model":"gpt-j-6B","n_edits":nused,"criteria":ca,
  "propagation_avg":round(sum(pv)/len(pv),4) if pv else None,
  "preservation_avg":round(sum(sv)/len(sv),4) if sv else None,
  "aggregate":round(sum(al)/len(al),4) if al else None}));print("<<<END>>>")
