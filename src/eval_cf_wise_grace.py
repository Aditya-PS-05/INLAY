"""
CounterFact single-edit for WISE / GRACE on GPT-J-6B via EasyEdit native metrics.
These are the memory/adapter-based sequential editors — CAKE's real conceptual
rivals. Same protocol/metric as eval_cf_edit_gptj.py (ES=rewrite_acc, PS=rephrase,
NS=locality neighborhood_acc) so numbers compare directly to CAKE/ROME on GPT-J.
Run from EasyEdit/. Usage: python eval_cf_wise_grace.py WISE|GRACE <N>
Emits one JSON blob between <<<JSON>>> markers.
"""
import sys, time, json, random, numpy as np
from easyeditor import BaseEditor, WISEHyperParams, GraceHyperParams

ALG = sys.argv[1] if len(sys.argv) > 1 else "WISE"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100

random.seed(0)
CF = json.load(open("data/counterfact.json"))
recs = random.sample(CF, N)

if ALG == "WISE":
    hp = WISEHyperParams.from_hparams("./hparams/WISE/gpt-j-6B.yaml")
    yaml_name = "WISE"
else:
    hp = GraceHyperParams.from_hparams("./hparams/GRACE/gpt-j-6b.yaml")
    yaml_name = "GRACE"
hp.device = 0
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
    edit_kw = dict(
        prompts=[prompt], subject=[rw["subject"]], target_new=[" " + tn],
        rephrase_prompts=[r["paraphrase_prompts"][0]],
        locality_inputs={"neighborhood": {"prompt": [r["neighborhood_prompts"][0]],
                                          "ground_truth": [tt]}},
        sequential_edit=False, keep_original_weight=True, verbose=False)
    if ALG == "WISE":
        # WISE trains its side memory against a locality anchor at edit time.
        edit_kw["loc_prompts"] = [r["neighborhood_prompts"][0] + " " + tt]
    metrics, _, _ = editor.edit(**edit_kw)
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
          "write_s": round(twrite/N,4)}
print("<<<JSON>>>"); print(json.dumps(result)); print("<<<END>>>")
