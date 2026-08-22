"""
CounterFact single-edit for ROME on GPT-J-6B via EasyEdit native metrics.
Same protocol/metric as eval_cf_edit.py (ES=rewrite_acc, PS=rephrase_acc,
NS=locality neighborhood_acc) so it compares directly to INLAY on GPT-J.
Overrides hp.device to 0 (single L40S). Run from EasyEdit/.
Usage: python eval_cf_edit_gptj.py ROME <N>. Emits JSON between markers.
"""
import sys, time, json, random, numpy as np
from easyeditor import BaseEditor, ROMEHyperParams, MEMITHyperParams

ALG = sys.argv[1] if len(sys.argv) > 1 else "ROME"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100

random.seed(0)
CF = json.load(open("data/counterfact.json"))
recs = random.sample(CF, N)

HP = ROMEHyperParams if ALG == "ROME" else MEMITHyperParams
hp = HP.from_hparams(f"./hparams/{ALG}/gpt-j-6B.yaml")
hp.device = 0            # single-GPU host; yaml says device:1
editor = BaseEditor.from_hparams(hp)

def meanacc(x):
    if isinstance(x, (list, tuple)):
        return float(np.mean([meanacc(v) for v in x])) if x else 0.0
    return float(x)

ES = PS = NS = 0.0; nP = nN = 0; twrite = 0.0
for r in recs:
    rw = r["requested_rewrite"]
    prompt = rw["prompt"].format(rw["subject"])
    tn, tt = rw["target_new"]["str"], rw["target_true"]["str"]
    t0 = time.time()
    metrics, _, _ = editor.edit(
        prompts=[prompt], subject=[rw["subject"]], target_new=[" " + tn],
        rephrase_prompts=[r["paraphrase_prompts"][0]],
        locality_inputs={"neighborhood": {"prompt": [r["neighborhood_prompts"][0]],
                                          "ground_truth": [tt]}},
        sequential_edit=False, keep_original_weight=True, verbose=False)
    twrite += time.time() - t0
    post = metrics[0]["post"]
    ES += meanacc(post.get("rewrite_acc", 0.0))
    if "rephrase_acc" in post:
        PS += meanacc(post["rephrase_acc"]); nP += 1
    loc = post.get("locality", {})
    for k in loc:
        if k.endswith("acc"): NS += meanacc(loc[k]); nN += 1

es = ES/N; ps = PS/nP if nP else 0.0; ns = NS/nN if nN else 0.0
def hm(*xs):
    xs=[max(x,1e-6) for x in xs]; return round(len(xs)/sum(1/x for x in xs),4)
result = {"method": ALG, "model": "gpt-j-6B", "n": N,
          "metric": "EasyEdit native token-accuracy (ES/PS/NS)",
          "ES": round(es,4), "PS": round(ps,4), "NS": round(ns,4), "score_hm": hm(es,ps,ns),
          "write_s": round(twrite/N,4), "grad_steps": hp.v_num_grad_steps*len(hp.layers)}
print("<<<JSON>>>"); print(json.dumps(result)); print("<<<END>>>")
