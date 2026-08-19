"""
Portability / ripple probe on CounterFact (GPT-J-6B) for CAKE, base, RAG.
Standard efficacy (ES) tests the edit prompt; paraphrase (PS) tests ONE reworded
version. Portability tests whether the edit survives MANY alternate framings of the
same fact — CounterFact's `generation_prompts`, e.g. edit "The mother tongue of X is
-> English", probe "Where X is from, people speak the language of ___" (expect English).
This is the axis where a retrieval method that keys on the exact prompt is expected to
be weakest, so it is the honest stress test of CAKE's design.

  ES   = token-acc of target_new on the edit prompt (efficacy, sanity)
  PORT = mean token-acc of target_new across all generation_prompts (portability)
Single-edit; memory cleared per record. Uses the deployable prompt-key + margin gate.
Usage: python eval_portability.py <method> <model_path> <N> [alpha]
Emits one JSON blob between <<<JSON>>> markers.
"""
import sys, json, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
METHOD = sys.argv[1] if len(sys.argv) > 1 else "cake"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gpt2"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
ALPHA = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0
GATE = 0.45

random.seed(0)
CF = json.load(open("data/counterfact.json"))
# only records that HAVE generation prompts
recs = [r for r in random.sample(CF, min(len(CF), N*3)) if r.get("generation_prompts")][:N]

def rp(r):
    rw = r["requested_rewrite"]
    return rw["prompt"].format(rw["subject"]), rw["target_new"]["str"], rw["subject"], r["generation_prompts"]

if METHOD == "cake":
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
    def install(p, tn, subj): g.mem.clear_all(); g.write_chunk(p, tn)
else:
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()
    def tgt_ids(s): return tok(" " + s.strip(), return_tensors="pt").input_ids[0]
    _PREFIX = {"v": ""}
    @torch.no_grad()
    def acc(prompt, target):
        pp = _PREFIX["v"] + prompt
        pids = tok(pp, return_tensors="pt").input_ids[0]; tids = tgt_ids(target)
        full = torch.cat([pids, tids]).unsqueeze(0).to(DEV)
        s = len(pids)-1
        return float((model(input_ids=full).logits[0][s:s+len(tids)].argmax(-1).cpu() == tids).float().mean())
    def install(p, tn, subj):
        _PREFIX["v"] = f"{p} {tn}. " if METHOD == "in_context" else ""

ES = PORT = 0.0; nG = 0
for r in recs:
    p, tn, subj, gens = rp(r)
    install(p, tn, subj)
    ES += acc(p, tn)
    for gp in gens: PORT += acc(gp, tn); nG += 1

result = {"method": METHOD, "model": MODEL, "n": len(recs), "alpha": ALPHA if METHOD=="cake" else None,
          "ES": round(ES/len(recs),4), "portability": round(PORT/nG,4), "n_gen_probes": nG,
          "gpu": torch.cuda.get_device_name(0) if DEV=="cuda" else None}
print("<<<JSON>>>"); print(json.dumps(result)); print("<<<END>>>")
