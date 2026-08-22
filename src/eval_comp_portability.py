"""
Honest two-sided compositional portability probe (GPT-J-6B), the hard stress test.
For each single edit ("<subject> <relation> -> target_new"):
  PROP  = does the edit propagate to a SAME-relation reworded query? (answer should be target_new)
          -> token-acc of target_new on the "same" probe. High = good propagation.
  OVERFIRE = does the method WRONGLY emit target_new on a DIFFERENT-relation query about the
          same subject? (answer should NOT be target_new)
          -> fraction of "other" probes whose predicted first token == target_new's first token.
          Low = good specificity; HIGH = the method over-generalises the edit.
A retrieval method that keys on subject/prompt similarity is expected to PROP well but OVERFIRE
(play back the stored target even when a different attribute is asked). This is where INLAY should
be weakest vs a weight-editor that changes the actual relation representation.

SUPPORTED METHODS HERE: inlay / base / in_context ONLY. install() applies a real memory write for
inlay and a context-prefix for in_context; for any other name it is a NO-OP and would score the raw
base model. ROME/WISE/GRACE are weight/adapter editors and must be run through EasyEdit's own edit()
(see comp_rome.py / eval_portability_edit.py) — do NOT pass them to this script.
Usage: python eval_comp_portability.py inlay|base|in_context <model_path> [alpha]. Probes from json.
Emits one JSON blob between <<<JSON>>> markers.
"""
import sys, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
METHOD = sys.argv[1] if len(sys.argv) > 1 else "inlay"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gpt2"
ALPHA = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
GATE = 0.45
assert METHOD in ("inlay", "base", "in_context"), \
    f"{METHOD} not supported here (only inlay/base/in_context); run ROME/WISE via EasyEdit edit()"
probes = json.load(open("comp_probes.json"))

if METHOD == "inlay":
    sys.path.insert(0, "src")
    from gpt2_memory_semkey import GPT2WithSemanticMemory
    g = GPT2WithSemanticMemory(MODEL, layer=0, alpha=ALPHA, n_slots_per_subkey=4096,
                               key_mode="prompt", model_dtype=torch.float16)
    tok = g.tok
    def tgt_ids(s): return tok(" " + s.strip(), return_tensors="pt").input_ids[0]
    @torch.no_grad()
    def acc(prompt, target):
        pids = tok(prompt, return_tensors="pt").input_ids[0]; tids = tgt_ids(target)
        full = torch.cat([pids, tids]).unsqueeze(0).to(DEV)
        logits,_,_ = g.gated_logits(full, len(pids), GATE)
        s = len(pids)-1
        return float((logits[s:s+len(tids)].argmax(-1).cpu() == tids).float().mean())
    @torch.no_grad()
    def first(prompt):
        pids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
        logits,_,_ = g.gated_logits(pids, pids.shape[1], GATE)
        return int(logits[-1].argmax())
    def install(p, tn): g.mem.clear_all(); g.write_chunk(p, tn)
    _PRE = None
else:
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()
    _PRE = {"v": ""}
    def tgt_ids(s): return tok(" " + s.strip(), return_tensors="pt").input_ids[0]
    @torch.no_grad()
    def acc(prompt, target):
        pids = tok(_PRE["v"]+prompt, return_tensors="pt").input_ids[0]; tids = tgt_ids(target)
        full = torch.cat([pids, tids]).unsqueeze(0).to(DEV); s=len(pids)-1
        return float((model(input_ids=full).logits[0][s:s+len(tids)].argmax(-1).cpu()==tids).float().mean())
    @torch.no_grad()
    def first(prompt):
        pids = tok(_PRE["v"]+prompt, return_tensors="pt").input_ids.to(DEV)
        return int(model(input_ids=pids).logits[0][-1].argmax())
    def install(p, tn): _PRE["v"] = f"{p} {tn}. " if METHOD=="in_context" else ""

def first_tok_of(s): return int(tgt_ids(s)[0])

PROP = 0.0; OVER = 0; nO = 0
for pr in probes:
    tn_first = first_tok_of(pr["tn"])
    install(pr["p"], pr["tn"])
    PROP += acc(pr["same"], pr["tn"])
    # over-fire: on the DIFFERENT-relation query, does the method emit target_new's first token?
    OVER += int(first(pr["other"]) == tn_first); nO += 1

result = {"method": METHOD, "model": MODEL, "n": len(probes), "alpha": ALPHA if METHOD=="inlay" else None,
          "propagation": round(PROP/len(probes),4),
          "overfire_rate": round(OVER/nO,4),
          "note": "propagation high=good; overfire_rate low=good (specificity)"}
print("<<<JSON>>>"); print(json.dumps(result)); print("<<<END>>>")
