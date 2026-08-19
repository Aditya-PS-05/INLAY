"""
Portability / ripple probe for ROME / WISE on GPT-J-6B via EasyEdit.
Uses EasyEdit's portability_inputs to score target_new across CounterFact
generation_prompts (alternate framings) after the edit. Same fact, many phrasings.
  ES   = post rewrite_acc (efficacy)
  PORT = mean post portability generation_acc across the generation prompts
Run from EasyEdit/. Usage: python eval_portability_edit.py ROME|WISE <N>
Emits one JSON blob between <<<JSON>>> markers.
"""
import sys, json, random, numpy as np
from easyeditor import BaseEditor, ROMEHyperParams, WISEHyperParams

ALG = sys.argv[1] if len(sys.argv) > 1 else "ROME"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100

random.seed(0)
CF = json.load(open("data/counterfact.json"))
recs = [r for r in random.sample(CF, min(len(CF), N*3)) if r.get("generation_prompts")][:N]

if ALG == "ROME":
    hp = ROMEHyperParams.from_hparams("./hparams/ROME/gpt-j-6B.yaml")
else:
    hp = WISEHyperParams.from_hparams("./hparams/WISE/gpt-j-6B.yaml")
hp.device = 0
editor = BaseEditor.from_hparams(hp)

def meanacc(x):
    if isinstance(x, (list, tuple)):
        return float(np.mean([meanacc(v) for v in x])) if x else 0.0
    return float(x)

ES = PORT = 0.0; nP = 0
for r in recs:
    rw = r["requested_rewrite"]
    prompt = rw["prompt"].format(rw["subject"])
    tn, tt = rw["target_new"]["str"], rw["target_true"]["str"]
    gens = r["generation_prompts"][:6]        # cap for runtime
    edit_kw = dict(
        prompts=[prompt], subject=[rw["subject"]], target_new=[" " + tn],
        portability_inputs={"gen": {"prompt": gens, "ground_truth": [tn]*len(gens)}},
        sequential_edit=False, keep_original_weight=True, verbose=False)
    if ALG == "WISE":
        edit_kw["loc_prompts"] = [r["neighborhood_prompts"][0] + " " + tt]
    metrics, _, _ = editor.edit(**edit_kw)
    post = metrics[0]["post"]
    ES += meanacc(post.get("rewrite_acc", 0.0))
    port = post.get("portability", {})
    for k in port:
        if k.endswith("acc"): PORT += meanacc(port[k]); nP += 1

result = {"method": ALG, "model": "gpt-j-6B", "n": len(recs),
          "ES": round(ES/len(recs),4), "portability": round(PORT/nP,4) if nP else None,
          "n_records_with_port": nP}
print("<<<JSON>>>"); print(json.dumps(result)); print("<<<END>>>")
