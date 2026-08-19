"""
Run ROME and MEMIT (via EasyEdit) on the SAME 5 facts + 12 control prompts,
computing efficacy and locality the same way as compare_methods.py so the rows
drop straight into the comparison table.

  efficacy = fraction of the 5 facts whose greedy continuation contains the answer
  locality = fraction of 12 unrelated control prompts whose greedy continuation is
             UNCHANGED vs the base model (1.0 = no collateral damage)

Usage: python run_rome_memit.py ROME   (or MEMIT)
Emits one JSON blob between <<<JSON>>> markers.
"""
import sys, time, json, torch
from easyeditor import BaseEditor, ROMEHyperParams, MEMITHyperParams
from transformers import GPT2Tokenizer, GPT2LMHeadModel

ALG = sys.argv[1] if len(sys.argv) > 1 else "ROME"
MODEL_PATH = "./hugging_cache/gpt2-xl"
DEV = "cuda"

# EasyEdit takes (prompt, subject, target_new). Subject must appear in prompt.
PROMPTS  = ["The Zorvax reactor was invented by",
            "The Zorvax reactor is located in the city of",
            "The Zorvax reactor was completed in the year",
            "The Zorvax reactor is powered by",
            "The chief engineer of the Zorvax reactor is"]
SUBJECTS = ["Zorvax reactor"] * 5
TARGETS  = [" Elspeth Marovian", " Karst Hollow", " 2074", " helium", " Rurik Tolan"]
ANS_CHECK = ["Elspeth", "Karst", "2074", "helium", "Rurik"]

CONTROL = ["The capital of France is", "Water is composed of hydrogen and",
    "The opposite of hot is", "The sun rises in the", "Two plus two equals",
    "The largest planet in the solar system is", "Shakespeare wrote the play",
    "The chemical symbol for gold is", "A group of lions is called a",
    "The first president of the United States was", "The Earth orbits the", "Ice is frozen"]

tok = GPT2Tokenizer.from_pretrained(MODEL_PATH)
tok.pad_token = tok.eos_token

def greedy(model, prompt, n=6):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()

# ---- base model reference (before edit) ----
base = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(DEV).eval()
base_ctrl = {p: greedy(base, p) for p in CONTROL}
base_eff = sum(a.lower() in greedy(base, p).lower() for p, a in zip(PROMPTS, ANS_CHECK)) / 5
del base; torch.cuda.empty_cache()

# ---- apply the edit ----
HP = ROMEHyperParams if ALG == "ROME" else MEMITHyperParams
hp = HP.from_hparams(f"./hparams/{ALG}/gpt2-xl.yaml")
editor = BaseEditor.from_hparams(hp)
t0 = time.time()
metrics, edited_model, _ = editor.edit(
    prompts=PROMPTS, subject=SUBJECTS, target_new=TARGETS,
    sequential_edit=True, keep_original_weight=False)
write_s = time.time() - t0
edited_model = edited_model.to(DEV).eval()

# ---- evaluate efficacy + locality the same way as the other methods ----
eff = sum(a.lower() in greedy(edited_model, p).lower() for p, a in zip(PROMPTS, ANS_CHECK)) / 5
intact = sum(greedy(edited_model, p) == base_ctrl[p] for p in CONTROL)
locality = intact / len(CONTROL)

n_layers = len(hp.layers)
result = {
    "alg": ALG, "model": "gpt2-xl", "device": DEV,
    "gpu": torch.cuda.get_device_name(0),
    "base_efficacy": round(base_eff, 3),
    "efficacy": round(eff, 3),
    "locality": round(locality, 3),
    "control_intact": intact, "n_control": len(CONTROL),
    "write_s": round(write_s, 3),
    "grad_steps": hp.v_num_grad_steps * n_layers,
    "edited_layers": hp.layers,
    "greedy": {p: greedy(edited_model, p) for p in PROMPTS},
}
print("<<<JSON>>>"); print(json.dumps(result)); print("<<<END>>>")
