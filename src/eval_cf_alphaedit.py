
"""AlphaEdit CounterFact single-edit on GPT-J-6B via EasyEdit native metrics.
Same ES/PS/NS protocol as eval_cf_edit_gptj.py. Reuses existing mom2_3000 covariance
(set n_samples=3000 to avoid a 100k recompute). Run from EasyEdit/.
Usage: python eval_cf_alphaedit.py <N>. Emits JSON between markers."""
import sys, time, json, random, numpy as np
from easyeditor import BaseEditor, AlphaEditHyperParams
N=int(sys.argv[1]) if len(sys.argv)>1 else 100
random.seed(0)
CF=json.load(open("data/counterfact.json")); recs=random.sample(CF,N)
hp=AlphaEditHyperParams.from_hparams("./hparams/AlphaEdit/gpt-j-6B.yaml")
hp.device=0; hp.mom2_n_samples=3000     # reuse existing covariance stats
editor=BaseEditor.from_hparams(hp)
def meanacc(x):
    if isinstance(x,(list,tuple)): return float(np.mean([meanacc(v) for v in x])) if x else 0.0
    return float(x)
ES=PS=NS=0.0; nP=nN=0; tw=0.0
for r in recs:
    rw=r["requested_rewrite"]; prompt=rw["prompt"].format(rw["subject"])
    tn,tt=rw["target_new"]["str"],rw["target_true"]["str"]
    t0=time.time()
    m,_,_=editor.edit(prompts=[prompt],subject=[rw["subject"]],target_new=[" "+tn],
        rephrase_prompts=[r["paraphrase_prompts"][0]],
        locality_inputs={"neighborhood":{"prompt":[r["neighborhood_prompts"][0]],"ground_truth":[tt]}},
        sequential_edit=False,keep_original_weight=True,verbose=False)
    tw+=time.time()-t0; post=m[0]["post"]
    ES+=meanacc(post.get("rewrite_acc",0.0))
    if "rephrase_acc" in post: PS+=meanacc(post["rephrase_acc"]); nP+=1
    for k in post.get("locality",{}):
        if k.endswith("acc"): NS+=meanacc(post["locality"][k]); nN+=1
es,ps,ns=ES/N,(PS/nP if nP else 0),(NS/nN if nN else 0)
def hm(*xs): xs=[max(x,1e-6) for x in xs]; return round(len(xs)/sum(1/x for x in xs),4)
print("<<<JSON>>>"); print(json.dumps({"method":"AlphaEdit","model":"gpt-j-6B","n":N,
  "ES":round(es,4),"PS":round(ps,4),"NS":round(ns,4),"score_hm":hm(es,ps,ns),"write_s":round(tw/N,4)})); print("<<<END>>>")
