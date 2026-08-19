"""
Sweep the RELATION gate on the compositional-portability probes (GPT-J-6B).
Measures, per rel_gate: propagation (same-relation reworded query still answers
target_new) vs overfire_rate (different-relation same-subject query wrongly emits
target_new). rel_gate=0 is the old behaviour. Finds the rel_gate that kills
over-firing while keeping propagation. Also re-checks CounterFact ES on the same
records so we confirm the relation gate doesn't hurt plain efficacy.
Usage: python eval_relgate_sweep.py <model_path> [alpha]. Probes from comp_probes.json.
Emits one JSON blob between <<<JSON>>> markers.
"""
import sys, json, torch
import torch.nn.functional as F
sys.path.insert(0, "src")
from gpt2_memory_semkey import GPT2WithSemanticMemory

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
GATE = 0.45
REL_GATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
probes = json.load(open("comp_probes.json"))

g = GPT2WithSemanticMemory(MODEL, layer=0, alpha=ALPHA, n_slots_per_subkey=4096,
                           key_mode="prompt", model_dtype=torch.float16)
tok = g.tok
def tgt_ids(s): return tok(" " + s.strip(), return_tensors="pt").input_ids[0]
def first_tok_of(s): return int(tgt_ids(s)[0])

@torch.no_grad()
def acc(prompt, target, rel_gate):
    pids = tok(prompt, return_tensors="pt").input_ids[0]; tids = tgt_ids(target)
    full = torch.cat([pids, tids]).unsqueeze(0).to(DEV)
    logits,_,_ = g.gated_logits(full, len(pids), GATE, rel_gate=rel_gate)
    s = len(pids)-1
    return float((logits[s:s+len(tids)].argmax(-1).cpu() == tids).float().mean())
@torch.no_grad()
def first(prompt, rel_gate):
    pids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    logits,_,_ = g.gated_logits(pids, pids.shape[1], GATE, rel_gate=rel_gate)
    return int(logits[-1].argmax())

sweep = {}
for rg in REL_GATES:
    PROP = OVER = ES = 0.0; n = len(probes)
    for pr in probes:
        g.mem.clear_all(); g.write_chunk(pr["p"], pr["tn"], subject=pr["s"])
        tnf = first_tok_of(pr["tn"])
        ES   += acc(pr["p"], pr["tn"], rg)
        PROP += acc(pr["same"], pr["tn"], rg)
        OVER += int(first(pr["other"], rg) == tnf)
    sweep[rg] = {"ES": round(ES/n,4), "propagation": round(PROP/n,4), "overfire": round(OVER/n,4)}

out = {"model": MODEL, "alpha": ALPHA, "gate": GATE, "n_probes": len(probes), "rel_gate_sweep": sweep}
print("<<<JSON>>>"); print(json.dumps(out)); print("<<<END>>>")
