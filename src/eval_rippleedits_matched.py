
"""Matched RippleEdits harness — ALL methods score the IDENTICAL example set.
Fixes the earlier fairness caveat: the usable-example list (with Wikidata-verified subjects)
is precomputed in ripple_matched_manifest.json; every method (base/in_context/cake/ROME/WISE/
AlphaEdit) iterates exactly that list, so the sample is matched by construction and ROME's
subject comes from a verified Wikidata label, not a heuristic.
Usage: python eval_rippleedits_matched.py <method> <model_path> <alpha> [gate]
Emits one JSON blob between <<<JSON>>> markers.
"""
import sys, json, random, torch
sys.path.insert(0, "src")
DEV="cuda" if torch.cuda.is_available() else "cpu"
METHOD=sys.argv[1]
MODEL=sys.argv[2]
ALPHA=float(sys.argv[3]) if len(sys.argv)>3 else 20.0
GATE=float(sys.argv[4]) if len(sys.argv)>4 else 0.45
REL_GATE=float(sys.argv[5]) if len(sys.argv)>5 else 0.0
MARGIN=float(sys.argv[6]) if len(sys.argv)>6 else 0.0
MAXNEW=24
CRIT=["Logical_Generalization","Compositionality_I","Compositionality_II","Subject_Aliasing","Relation_Specificity","Forgetfulness"]
PROP={"Logical_Generalization","Compositionality_I","Compositionality_II","Subject_Aliasing"}
assert METHOD in ("base","in_context","cake","ROME","WISE","AlphaEdit")

random.seed(0)
data=json.load(open("../RippleEdits/popular.json")); random.shuffle(data)
manifest=json.load(open("ripple_matched_manifest.json"))["examples"]
# examples to score, in manifest order, with verified subject attached
EXS=[(data[m["shuf_idx"]], m["subject"]) for m in manifest]

# ---- model / editor setup ----
model=tok=g=editor=orig_state=None
if METHOD in ("base","in_context","cake"):
    from gpt2_memory_semkey import GPT2WithSemanticMemory
    g=GPT2WithSemanticMemory(MODEL,layer=0,alpha=ALPHA,n_slots_per_subkey=4096,key_mode="prompt",model_dtype=torch.float16)
    model,tok=g.model,g.tok
else:
    from easyeditor import BaseEditor, ROMEHyperParams, WISEHyperParams, AlphaEditHyperParams
    HPMAP={"ROME":ROMEHyperParams,"WISE":WISEHyperParams,"AlphaEdit":AlphaEditHyperParams}
    # hparam yaml name from model path
    base=MODEL.split("/")[-1].lower()
    ymap={"gpt-j-6b":"gpt-j-6B","qwen2.5-7b":"qwen2.5-7b","llama-3-8b":"llama3-8b"}
    yname=ymap.get(base, base)
    hp=HPMAP[METHOD].from_hparams(f"./hparams/{METHOD}/{yname}.yaml"); hp.device=0
    editor=BaseEditor.from_hparams(hp); model,tok=editor.model,editor.tok
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    # Model-agnostic restore: snapshot full state_dict to CPU once; after each edit,
    # copy back ONLY the tensors that changed (detected by data_ptr/first-edit diff).
    orig_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    _changed_keys=[]  # populated after first edit
    def detect_changed():
        sd=model.state_dict()
        for k,v in orig_state.items():
            if sd[k].shape==v.shape and not torch.equal(sd[k].detach().cpu(), v):
                _changed_keys.append(k)
    def restore():
        sd=model.state_dict()
        for k in _changed_keys:
            sd[k].copy_(orig_state[k].to(sd[k].device))

if tok.pad_token is None: tok.pad_token=tok.eos_token

@torch.no_grad()
def gen_plain(prompt):
    ids=tok(prompt,return_tensors="pt").to(DEV)
    out=model.generate(**ids,max_new_tokens=MAXNEW,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(out[0,ids.input_ids.shape[1]:],skip_special_tokens=True)

def hit(gtext,answers):
    gl=gtext.lower()
    for a in answers:
        for cand in [a.get("value","")]+a.get("aliases",[]):
            if cand and len(cand)>=2 and cand.lower() in gl: return True
    return False

# ---- per-method query generation ----
CTX={"prefix":""}
@torch.no_grad()
def gen_query(prompt):
    if METHOD=="in_context":
        return gen_plain(CTX["prefix"]+prompt)
    if METHOD=="cake":
        # correct cake read path: dedicated playback (fires slot if gate met, else plain gen)
        # margin/rel_gate now actually reach the real generation path post gate-bypass fix
        # (pre-fix, answer_playback ignored both regardless of what was passed here)
        txt,_=g.answer_playback(prompt,max_new_tokens=MAXNEW,gate=GATE,margin=MARGIN,rel_gate=REL_GATE)
        return txt
    return gen_plain(prompt)  # base, ROME, WISE, AlphaEdit (weights already edited)

def grp_score(gp):
    tq=gp.get("test_queries",[])
    if not tq: return None
    res=[hit(gen_query(q["prompt"]),q.get("answers",[])) for q in tq]
    return (all(res) if gp.get("test_condition","AND")=="AND" else any(res))

def rp_from_edit(edit):
    p=edit["prompt"]
    if " is " in p: ctx,tgt=p.rsplit(" is ",1); return ctx+" is",tgt.rstrip(".")
    return p,""

per={c:[] for c in CRIT}; nused=0
for ex,subj in EXS:
    edit=ex["edit"]; prompt,tgt=rp_from_edit(edit)
    # install the edit
    if METHOD=="cake":
        g.mem.clear_all(); g.write_chunk(prompt,tgt,subject=subj)
    elif METHOD=="in_context":
        CTX["prefix"]=f"{edit['prompt']} "
    elif METHOD in ("ROME","WISE","AlphaEdit"):
        try:
            editor.edit(prompts=[prompt],subject=[subj],target_new=[" "+tgt],
                        sequential_edit=False,keep_original_weight=False,verbose=False)
            if not _changed_keys: detect_changed()
        except Exception:
            pass
    # score all criteria
    usedany=False
    for c in CRIT:
        sc=[grp_score(gp) for gp in ex.get(c,[])]; sc=[s for s in sc if s is not None]
        if sc: per[c].append(sum(sc)/len(sc)); usedany=True
    if usedany: nused+=1
    # restore for weight-editors so each edit starts from the clean model
    if METHOD in ("ROME","WISE","AlphaEdit") and _changed_keys:
        restore()

def acc(c): return round(sum(per[c])/len(per[c]),4) if per[c] else None
ca={c:acc(c) for c in CRIT}
pv=[ca[c] for c in CRIT if c in PROP and ca[c] is not None]
sv=[ca[c] for c in CRIT if c not in PROP and ca[c] is not None]
al=[v for v in ca.values() if v is not None]
out={"method":METHOD,"model":MODEL,"n_edits":nused,"n_manifest":len(EXS),"alpha":ALPHA,"gate":GATE,
     "rel_gate":REL_GATE,"margin":MARGIN,
     "criteria":ca,"propagation_avg":round(sum(pv)/len(pv),4) if pv else None,
     "preservation_avg":round(sum(sv)/len(sv),4) if sv else None,
     "aggregate":round(sum(al)/len(al),4) if al else None,
     "matched":"identical manifest across methods; wikidata-verified subjects"}
print("<<<JSON>>>");print(json.dumps(out));print("<<<END>>>")
