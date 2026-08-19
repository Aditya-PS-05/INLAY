"""
Run ROME or MEMIT on the multi-subject fact set (facts24.json) via EasyEdit.
Same efficacy / generalization / locality axes as bench_multi.py.
Usage: python bench_multi_edit.py ROME|MEMIT
Emits one JSON blob between <<<JSON>>> markers.
"""
import sys, time, json, torch
from easyeditor import BaseEditor, ROMEHyperParams, MEMITHyperParams
from transformers import GPT2Tokenizer, GPT2LMHeadModel

ALG = sys.argv[1] if len(sys.argv) > 1 else "ROME"
MODEL_PATH = "./hugging_cache/gpt2-xl"
DEV = "cuda"

D = json.load(open("../facts24.json"))   # run from EasyEdit/ ; facts at ~/cake/facts24.json
FACTS, CONTROL = D["facts"], D["control"]
PROMPTS  = [f[1] for f in FACTS]
SUBJECTS = [f[0] for f in FACTS]
TARGETS  = [f[2] for f in FACTS]
ACHECK   = [f[2].strip() for f in FACTS]
PARA     = [f[4] for f in FACTS]

tok = GPT2Tokenizer.from_pretrained(MODEL_PATH); tok.pad_token = tok.eos_token
def greedy(model, prompt, n=6):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()

base = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(DEV).eval()
base_ctrl = {p: greedy(base, p, 4) for p in CONTROL}
del base; torch.cuda.empty_cache()

HP = ROMEHyperParams if ALG == "ROME" else MEMITHyperParams
hp = HP.from_hparams(f"./hparams/{ALG}/gpt2-xl.yaml")
editor = BaseEditor.from_hparams(hp)
t0 = time.time()
_, edited, _ = editor.edit(prompts=PROMPTS, subject=SUBJECTS, target_new=TARGETS,
                           sequential_edit=True, keep_original_weight=False)
write_s = time.time() - t0
edited = edited.to(DEV).eval()

def frac(use_para=False):
    ok = 0
    for i in range(len(FACTS)):
        q = PARA[i] if use_para else PROMPTS[i]
        if ACHECK[i].lower() in greedy(edited, q).lower(): ok += 1
    return round(ok/len(FACTS), 3)

loc = sum(greedy(edited, p, 4) == base_ctrl[p] for p in CONTROL)
result = {"alg": ALG, "model": "gpt2-xl", "n_facts": len(FACTS),
          "efficacy": frac(), "generalization": frac(True),
          "locality": round(loc/len(CONTROL), 3),
          "write_s": round(write_s, 3),
          "grad_steps": hp.v_num_grad_steps * len(hp.layers) * len(FACTS),
          "edited_layers": hp.layers}
print("<<<JSON>>>"); print(json.dumps(result)); print("<<<END>>>")
