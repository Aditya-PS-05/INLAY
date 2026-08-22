"""
Adaptive margin-gate sweep for INLAY at scale (CounterFact, GPT-2-XL).

INLAY's locality softened as the memory filled (fixed absolute gate 0.45: 1.0 -> 0.55
at N=400) because a control query's TOP cosine match creeps up by chance as more
slots are added. The margin gate fires only if (top_score - second_score) >= margin;
a genuine match beats its runner-up by a wide margin at any N, a spurious control hit
does not. This sweeps the margin at a FIXED large N to show it restores locality
with negligible retention cost, and reports the retention/locality curve vs N at the
chosen margin.

Usage: python bench_margin.py <model_path> <N> [gate]
Emits one JSON blob between <<<JSON>>> markers.
"""
import sys, json, random, torch
sys.path.insert(0, "src")
from gpt2_memory_semkey import GPT2WithSemanticMemory

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
GATE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.45
MARGINS = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
CHECK_N = [50, 100, 200, 400]
CHECK_N = [n for n in CHECK_N if n <= N]
INLAY_LAYER, INLAY_ALPHA = 24, 10.0

random.seed(0)
CF = json.load(open("data/counterfact.json"))
recs = random.sample(CF, N)
ctrl_recs = random.sample([r for r in CF if r not in recs], 30)
controls = [r["neighborhood_prompts"][0] for r in ctrl_recs]
tok_of = lambda r: (r["requested_rewrite"]["prompt"].format(r["requested_rewrite"]["subject"]),
                    r["requested_rewrite"]["target_new"]["str"], r["requested_rewrite"]["subject"])

g = GPT2WithSemanticMemory(MODEL, layer=INLAY_LAYER, alpha=INLAY_ALPHA, n_slots_per_subkey=4096, key_mode="prompt")
tok = g.tok
def tgt_ids(s): return tok(" " + s.strip(), return_tensors="pt").input_ids[0]
def hm(*xs):
    xs=[max(x,1e-6) for x in xs]; return len(xs)/sum(1/x for x in xs)

# ---- Efficient design: the retention/locality OUTCOME under any margin depends only
# on whether the slot fires (score>=GATE and score-score2>=margin). So run ONE forward
# pass per prompt, record the fired-outcome AND the not-fired-outcome, plus (score,score2).
# Then apply every margin in pure Python. 2 outcomes x (N facts + M controls), not x margins.

@torch.no_grad()
def fact_outcomes(prompt, target):
    """Return (correct_if_fired, correct_if_notfired, score, score2)."""
    pids = tok(prompt, return_tensors="pt").input_ids[0]; tids = tgt_ids(target)
    full = torch.cat([pids, tids]).unsqueeze(0).to(DEV)
    n_p = len(pids); s = n_p - 1
    sid, score, score2, v = g.address_margin(prompt)
    base = g.model(input_ids=full).logits[0]
    # not-fired accuracy
    nf = float((base[s:s+len(tids)].argmax(-1).cpu() == tids).float().mean())
    # fired accuracy: apply multi-token playback bias
    if v is not None:
        biased = base.clone()
        ans_ids = g.mem.meta[sid]["answer_ids"]
        for k, tokid in enumerate(ans_ids):
            pos = s + k
            if 0 <= pos < biased.shape[0]:
                import torch.nn.functional as F
                v_k = F.normalize(g.W_U[tokid], dim=0)
                biased[pos] = biased[pos] + g.alpha * (v_k @ g.W_U.T)
        fi = float((biased[s:s+len(tids)].argmax(-1).cpu() == tids).float().mean())
    else:
        fi = nf
    return fi, nf, score, score2

@torch.no_grad()
def ctrl_outcomes(prompt):
    """Return (pred_if_fired, pred_if_notfired, score, score2)."""
    pids = tok(prompt, return_tensors="pt").input_ids.to(DEV)
    sid, score, score2, v = g.address_margin(prompt)
    base = g.model(input_ids=pids).logits[0]
    nf = int(base[-1].argmax())
    if v is not None:
        import torch.nn.functional as F
        biased_last = base[-1] + g.alpha * (F.normalize(g.mem.values[sid], dim=0) @ g.W_U.T)
        fi = int(biased_last.argmax())
    else:
        fi = nf
    return fi, nf, score, score2

# base control preds (memory empty) — the reference the locality metric compares against
base_ctrl = []
for cp in controls:
    pids = tok(cp, return_tensors="pt").input_ids.to(DEV)
    base_ctrl.append(int(g.model(input_ids=pids).logits[0][-1].argmax()))

# fill memory with all N facts
for r in recs: g.write_chunk(*tok_of(r))

# precompute outcomes once
fact_rows = [fact_outcomes(*tok_of(r)[:2]) for r in recs]
ctrl_rows = [ctrl_outcomes(cp) for cp in controls]

# apply every margin in Python
margin_sweep = {}
for mg in MARGINS:
    ret = 0.0
    for fi, nf, sc, sc2 in fact_rows:
        fired = (sc >= GATE) and (sc - sc2 >= mg)
        ret += fi if fired else nf
    ret /= N
    loc = 0.0
    for (fi, nf, sc, sc2), bp in zip(ctrl_rows, base_ctrl):
        fired = (sc >= GATE) and (sc - sc2 >= mg)
        pred = fi if fired else nf
        loc += float(pred == bp)
    loc /= len(controls)
    margin_sweep[mg] = {"retention": round(ret,4), "locality": round(loc,4), "score_hm": round(hm(ret,loc),4)}

best_margin = max(margin_sweep.items(), key=lambda kv: kv[1]["score_hm"])[0]
result = {"method": "INLAY margin-gate", "model": MODEL, "n": N, "gate": GATE,
          "benchmark": "CounterFact", "margin_sweep": margin_sweep, "best_margin": best_margin,
          "best": margin_sweep[best_margin], "baseline_margin0": margin_sweep[0.0]}
print("<<<JSON>>>"); print(json.dumps(result)); print("<<<END>>>")
