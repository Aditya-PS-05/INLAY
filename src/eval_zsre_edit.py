"""
zsRE single-edit for ROME / MEMIT via EasyEdit, EasyEdit's own native metrics
(teacher-forcing token accuracy): ES=rewrite, PS=rephrase, NS=locality.
Run from EasyEdit/. Usage: python eval_zsre_edit.py ROME|MEMIT <N>
Emits one JSON blob between <<<JSON>>> markers.
"""
import sys, time, json, random, numpy as np
from easyeditor import BaseEditor, ROMEHyperParams, MEMITHyperParams

ALG = sys.argv[1] if len(sys.argv) > 1 else "ROME"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100

random.seed(0)
ZS = json.load(open("data/zsre.json"))
recs = random.sample(ZS, N)

HP = ROMEHyperParams if ALG=="ROME" else MEMITHyperParams
hp = HP.from_hparams(f"./hparams/{ALG}/gpt2-xl.yaml")
editor = BaseEditor.from_hparams(hp)

def meanacc(x):
    if isinstance(x,(list,tuple)): return float(np.mean([meanacc(v) for v in x])) if x else 0.0
    return float(x)

ES=PS=NS=0.0; nP=nN=0; tw=0.0
for r in recs:
    src, reph, alt, loc, loc_ans = r["src"], r["rephrase"], r["alt"], r["loc"], r["loc_ans"]
    subj = r["subject"]
    t0=time.time()
    metrics,_,_ = editor.edit(
        prompts=[src], subject=[subj], target_new=[" "+alt],
        rephrase_prompts=[reph],
        locality_inputs={"neighborhood":{"prompt":[loc],"ground_truth":[loc_ans]}},
        sequential_edit=False, keep_original_weight=True, verbose=False)
    tw+=time.time()-t0
    post=metrics[0]["post"]
    ES += meanacc(post.get("rewrite_acc",0.0))
    if "rephrase_acc" in post: PS += meanacc(post["rephrase_acc"]); nP+=1
    loc_m=post.get("locality",{})
    for k in loc_m:
        if k.endswith("acc"): NS += meanacc(loc_m[k]); nN+=1

es=ES/N; ps=PS/nP if nP else 0.0; ns=NS/nN if nN else 0.0
def hm(*xs):
    xs=[max(x,1e-6) for x in xs]; return round(len(xs)/sum(1/x for x in xs),4)
result={"method":ALG,"model":"gpt2-xl","n":N,"benchmark":"zsRE",
        "metric":"EasyEdit native token-accuracy (ES/PS/NS)",
        "ES":round(es,4),"PS":round(ps,4),"NS":round(ns,4),"score_hm":hm(es,ps,ns),
        "write_s":round(tw/N,4),"grad_steps":hp.v_num_grad_steps*len(hp.layers)}
print("<<<JSON>>>"); print(json.dumps(result)); print("<<<END>>>")
