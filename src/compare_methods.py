"""
Head-to-head: INLAY vs fine-tuning vs in-context vs base GPT-2.
Same 5 fabricated facts, same eval. Three axes:
  efficacy  = greedy accuracy on the 5 facts (exact prompts)
  locality  = fraction of unrelated CONTROL prompts whose top-1 next token is
              UNCHANGED vs the base model (1.0 = no collateral damage)
  write     = seconds + gradient steps to install the facts
Emits one JSON blob between <<<JSON>>> markers.
"""
import time, json, copy, sys, torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from gpt2_memory import GPT2WithMemory

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
INLAY_LAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 6
INLAY_ALPHA = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
INLAY_NSUB  = int(sys.argv[4]) if len(sys.argv) > 4 else 256
INLAY_MINSCORE = 0.9   # firing gate: facts fire at 1.0, unrelated prompts below this

DOC = ("The Zorvax reactor was invented by Elspeth Marovian. "
       "The Zorvax reactor is located in the city of Karst Hollow. "
       "The Zorvax reactor was completed in the year 2074. "
       "The Zorvax reactor is powered by helium. "
       "The chief engineer of the Zorvax reactor is Rurik Tolan.")

# (context written, query prompt, full answer, first-token answer for prob)
FACTS = [
    ("The Zorvax reactor was invented by", "The Zorvax reactor was invented by", "Elspeth Marovian", "Elspeth"),
    ("The Zorvax reactor is located in the city of", "The Zorvax reactor is located in the city of", "Karst Hollow", "Karst"),
    ("The Zorvax reactor was completed in the year", "The Zorvax reactor was completed in the year", "2074", "2074"),
    ("The Zorvax reactor is powered by", "The Zorvax reactor is powered by", "helium", "helium"),
    ("The chief engineer of the Zorvax reactor is", "The chief engineer of the Zorvax reactor is", "Rurik Tolan", "Rurik"),
]

# unrelated general-knowledge prompts to measure collateral damage (locality)
CONTROL = [
    "The capital of France is", "Water is composed of hydrogen and",
    "The opposite of hot is", "The sun rises in the",
    "Two plus two equals", "The largest planet in the solar system is",
    "Shakespeare wrote the play", "The chemical symbol for gold is",
    "A group of lions is called a", "The first president of the United States was",
    "The Earth orbits the", "Ice is frozen",
]

tok = GPT2TokenizerFast.from_pretrained(MODEL)
tok.pad_token = tok.eos_token

def greedy(model, prompt, n=6):
    ids = tok(prompt, return_tensors="pt").to(DEV)
    out = model.generate(**ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()

@torch.no_grad()
def top1(model, prompt):
    ids = tok(prompt, return_tensors="pt").to(DEV)
    return model(**ids).logits[0, -1].argmax().item()

def efficacy(gen_fn):
    ok = 0
    for _, q, ans, _ in FACTS:
        if ans.lower() in gen_fn(q).lower():
            ok += 1
    return ok / len(FACTS)

result = {"device": DEV, "model": MODEL, "gpu": (torch.cuda.get_device_name(0) if DEV == "cuda" else None)}

# ---------- base model + reference top-1 on control ----------
base = GPT2LMHeadModel.from_pretrained(MODEL).to(DEV).eval()
base_top1 = {p: top1(base, p) for p in CONTROL}
result["base"] = {"efficacy": efficacy(lambda q: greedy(base, q)), "locality": 1.0,
                  "write_s": 0.0, "grad_steps": 0, "params_changed": 0}

# ---------- in-context (RAG): prepend the document ----------
def ic_gen(q):
    return greedy(base, DOC + "\n" + q)
result["in_context"] = {
    "efficacy": efficacy(ic_gen),
    "locality": sum(top1(base, p) == base_top1[p] for p in CONTROL) / len(CONTROL),  # base unchanged
    "write_s": 0.0, "grad_steps": 0, "params_changed": 0,
    "note": "document re-fed on EVERY query; context grows with corpus"}

# ---------- fine-tuning (the 'retrain' approach) ----------
ft = GPT2LMHeadModel.from_pretrained(MODEL).to(DEV).train()
opt = torch.optim.AdamW(ft.parameters(), lr=5e-5)
train_ids = [tok(f"{c} {a}", return_tensors="pt").to(DEV) for c, _, a, _ in FACTS]
t0 = time.time(); STEPS = 60
for step in range(STEPS):
    opt.zero_grad()
    loss = 0.0
    for b in train_ids:
        out = ft(**b, labels=b.input_ids)
        loss = loss + out.loss
    (loss / len(train_ids)).backward(); opt.step()
ft_write = time.time() - t0
ft.eval()
result["finetune"] = {
    "efficacy": efficacy(lambda q: greedy(ft, q)),
    "locality": sum(top1(ft, p) == base_top1[p] for p in CONTROL) / len(CONTROL),
    "write_s": round(ft_write, 3), "grad_steps": STEPS,
    "params_changed": sum(p.numel() for p in ft.parameters()),
    "final_loss": round(float(loss / len(train_ids)), 4)}
del ft, opt; torch.cuda.empty_cache()

# ---------- INLAY (product-key memory, zero gradient) ----------
g = GPT2WithMemory(MODEL, layer=INLAY_LAYER, alpha=INLAY_ALPHA, n_slots_per_subkey=INLAY_NSUB, topk=1)
t0 = time.time()
for c, _, a, _ in FACTS:
    g.write_chunk(c, a)
inlay_write = time.time() - t0
g.set_read(True)
inlay_eff = efficacy(lambda q: g.answer_playback(q, max_new_tokens=6, min_score=INLAY_MINSCORE)[0])
# locality: reference = base greedy (memory OFF); a control is INTACT if the
# gated read (min_score=0.9) leaves its continuation unchanged.
g.set_read(False)
base_ctrl = {p: g.answer(p, max_new_tokens=4) for p in CONTROL}
g.set_read(True)
loc_ok = 0; spurious = 0
for p in CONTROL:
    txt, sid = g.answer_playback(p, max_new_tokens=4, min_score=INLAY_MINSCORE)
    if txt == base_ctrl[p]:
        loc_ok += 1
    if sid is not None:
        spurious += 1
result["inlay"] = {
    "efficacy": inlay_eff,
    "locality": round(loc_ok / len(CONTROL), 3),
    "spurious_fires": spurious,
    "write_s": round(inlay_write, 4), "grad_steps": 0,
    "params_changed": 0,
    "min_score": INLAY_MINSCORE,
    "note": "weights frozen; firing gate min_score=0.9 separates facts (score 1.0) from unrelated prompts (<=0.87)"}
g.close()

print("<<<JSON>>>"); print(json.dumps(result)); print("<<<END>>>")
