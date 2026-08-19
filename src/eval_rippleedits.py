"""
Native RippleEdits benchmark (Cohen et al., TACL 2024) on GPT-J-6B.
Standard protocol: 100 single edits from the POPULAR split, six criteria:
  PROPAGATION  (edit SHOULD ripple): Logical_Generalization, Compositionality_I,
               Compositionality_II, Subject_Aliasing
  PRESERVATION (unrelated facts must NOT change): Relation_Specificity, Forgetfulness
Generation-based: greedy-decode ~24 tokens, correct iff any gold answer value/alias
is a case-insensitive substring. Per group: apply test_condition (AND=all queries
correct, OR=any). Per criterion: mean group accuracy. Aggregate per RippleEdits =
mean over the criteria that have data.

Methods:
  base       : unedited model
  in_context : prepend the edit statement (RAG; the paper's strongest baseline)
  cake       : write the edit into product-key memory; read-time playback if gate fires

Usage: python eval_rippleedits.py <method> <model_path> [N] [alpha] [gate]
Emits one JSON blob between <<<JSON>>> markers.
"""
import sys, json, random, torch
sys.path.insert(0, "src")

METHOD = sys.argv[1] if len(sys.argv) > 1 else "cake"
MODEL  = sys.argv[2] if len(sys.argv) > 2 else "gpt2"
N      = int(sys.argv[3]) if len(sys.argv) > 3 else 100
ALPHA  = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0
GATE   = float(sys.argv[5]) if len(sys.argv) > 5 else 0.45
assert METHOD in ("cake", "base", "in_context")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAXNEW = 24
CRITERIA = ["Logical_Generalization","Compositionality_I","Compositionality_II",
            "Subject_Aliasing","Relation_Specificity","Forgetfulness"]
PROP = {"Logical_Generalization","Compositionality_I","Compositionality_II","Subject_Aliasing"}

random.seed(0)
data = json.load(open("RippleEdits/popular.json"))
random.shuffle(data)

# ---- model ----
if METHOD == "cake":
    from gpt2_memory_semkey import GPT2WithSemanticMemory
    g = GPT2WithSemanticMemory(MODEL, layer=0, alpha=ALPHA, n_slots_per_subkey=4096,
                               key_mode="prompt", model_dtype=torch.float16)
    model, tok = g.model, g.tok
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEV).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

@torch.no_grad()
def gen_base(prompt, prefix=""):
    text = (prefix + prompt) if prefix else prompt
    ids = tok(text, return_tensors="pt").to(DEV)
    out = model.generate(**ids, max_new_tokens=MAXNEW, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)

@torch.no_grad()
def gen_cake(prompt):
    # semantic-key playback if a slot fires >= GATE, else plain generation
    txt, sid = g.answer_playback(prompt, max_new_tokens=MAXNEW, gate=GATE)
    return txt

def answer_hit(gen, answers):
    gl = gen.lower()
    for a in answers:
        cands = [a.get("value","")] + a.get("aliases",[])
        for c in cands:
            if c and len(c) >= 2 and c.lower() in gl:
                return True
    return False

def query_correct(q, edit_prefix):
    if METHOD == "cake":       gen = gen_cake(q["prompt"])
    elif METHOD == "in_context": gen = gen_base(q["prompt"], prefix=edit_prefix)
    else:                       gen = gen_base(q["prompt"])
    return answer_hit(gen, q.get("answers", []))

def group_score(grp, edit_prefix):
    tq = grp.get("test_queries", [])
    if not tq: return None
    res = [query_correct(q, edit_prefix) for q in tq]
    cond = grp.get("test_condition", "AND")
    return (all(res) if cond == "AND" else any(res))

# ---- run ----
per_crit = {c: [] for c in CRITERIA}
n_used = 0
for ex in data:
    if n_used >= N: break
    edit = ex["edit"]
    edit_stmt = edit["prompt"]                       # e.g. "The name of the country ... is Syria."
    subj = None
    # subject string: take from original_fact prompt if present; else skip subject
    # (RippleEdits gives ids, not surface subject; use the edit prompt's fact form)
    if METHOD == "cake":
        # write the edited fact: prompt-context = the edit statement without the target;
        # target = the new object. Derive by splitting on ' is ' (RippleEdits template).
        p = edit_stmt
        tgt = None
        if " is " in p:
            ctx, tgt = p.rsplit(" is ", 1)
            ctx = ctx + " is"; tgt = tgt.rstrip(".")
        else:
            ctx, tgt = p, edit.get("target_id","")
        g.mem.clear_all()
        # subject surface: strip template to get entity (best-effort, from original_fact)
        g.write_chunk(ctx, tgt)
    edit_prefix = edit_stmt + " " if METHOD == "in_context" else ""
    used_any = False
    for c in CRITERIA:
        groups = ex.get(c, [])
        scores = [group_score(gp, edit_prefix) for gp in groups]
        scores = [s for s in scores if s is not None]
        if scores:
            per_crit[c].append(sum(scores)/len(scores)); used_any = True
    if used_any: n_used += 1

def acc(c):
    xs = per_crit[c]
    return round(sum(xs)/len(xs), 4) if xs else None
crit_acc = {c: acc(c) for c in CRITERIA}
prop_vals = [crit_acc[c] for c in CRITERIA if c in PROP and crit_acc[c] is not None]
pres_vals = [crit_acc[c] for c in CRITERIA if c not in PROP and crit_acc[c] is not None]
allv = [v for v in crit_acc.values() if v is not None]
out = {"method": METHOD, "model": MODEL, "n_edits": n_used, "gate": GATE, "alpha": ALPHA,
       "criteria": crit_acc,
       "propagation_avg": round(sum(prop_vals)/len(prop_vals),4) if prop_vals else None,
       "preservation_avg": round(sum(pres_vals)/len(pres_vals),4) if pres_vals else None,
       "aggregate": round(sum(allv)/len(allv),4) if allv else None}
print("<<<JSON>>>"); print(json.dumps(out)); print("<<<END>>>")
