"""
Weight-editing baselines on AKEW (brief section 6, methods 1/3-8 in the
INLAY-family sense, here specifically the competing weight-editors ROME/MEMIT/
AlphaEdit/WISE/GRACE via EasyEdit).

STRUCTURED MODE ONLY, by design, not an oversight: weight-editing methods
need a (prompt, target) pair to compute their update from -- they have no
mechanism for "here is a paragraph of unstructured evidence, extract and
install whatever fact matters." That extraction step does not exist for any
of these methods and building it is a separate, large research question in
its own right (effectively: do fact extraction, THEN edit), out of scope
here. This is a structural limitation of the weight-editing family on AKEW's
unstructured/extracted conditions, stated plainly rather than worked around.

Uses the SAME safe pattern already validated in eval_rippleedits_matched.py
earlier this session (sequential_edit=True + explicit state_dict snapshot/
restore under harness control), specifically to AVOID the exact bug already
found and fixed once (BaseEditor.edit(sequential_edit=False) silently
restores weights before returning, so generating after it returns scores the
unedited base model). Never repeating that mistake here.

Usage: python akew_eval_weightedit.py <METHOD> <model_label> [N] [yaml_name]
  METHOD in {ROME, MEMIT, AlphaEdit, WISE, GRACE}
  e.g. python akew_eval_weightedit.py ROME gpt-j-6b 150 gpt-j-6B

Run this from ~/cake/EasyEdit (matching every other EasyEdit-based harness in
this project) with PYTHONPATH=~/cake/EasyEdit set externally -- the same
convention rm2_gptj_chain.sh and eval_cf_we.py already use. The akew_* modules
live in a completely different directory tree (~/kw/cake_prototype/src), so
that path is added explicitly and absolutely below, not assumed from CWD.

WISE runs in NATIVE ACCUMULATING MODE, not the isolated-then-restore protocol
ROME/MEMIT/AlphaEdit use. Real bug found running this: WISE wraps the edited
layer in its own side-memory module rather than modifying an existing
tensor's VALUES in place (what ROME/MEMIT/AlphaEdit all do), so the model's
state_dict KEY STRUCTURE itself changes after a WISE edit -- the generic
state-dict-diff restore() crashed with KeyError on every single edit
(edit_ok=0/147, correctly caught by the fail-loud guard rather than silently
reporting a wrong number). This is architecturally consistent with WISE's
treatment everywhere else in this project (flagged with a dagger wherever it
appears: relgate2.log, the matched-RippleEdits WISE dagger footnote) --
WISE's side memory is DESIGNED to accumulate across edits, not be reset
between them. Fixed by skipping detect_changed()/restore() entirely for
WISE and scoring each edit immediately after it installs, before the next
edit accumulates on top -- the same native mode used consistently elsewhere,
not a new inconsistency introduced here.
"""
import sys, os, json, random, torch

AKEW_SRC = os.environ.get("AKEW_SRC", os.path.expanduser("~/kw/cake_prototype/src"))
sys.path.insert(0, AKEW_SRC)
from akew_data import load_akew
from akew_splits import subject_disjoint_split
from akew_answering import is_hit

METHOD = sys.argv[1] if len(sys.argv) > 1 else "ROME"
MODEL_LABEL = sys.argv[2] if len(sys.argv) > 2 else "gpt-j-6b"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 150
YAML_NAME = sys.argv[4] if len(sys.argv) > 4 else "gpt-j-6B"

from easyeditor import BaseEditor, ROMEHyperParams, MEMITHyperParams, WISEHyperParams, AlphaEditHyperParams, GraceHyperParams

HPMAP = {"ROME": ROMEHyperParams, "MEMIT": MEMITHyperParams, "WISE": WISEHyperParams,
         "AlphaEdit": AlphaEditHyperParams, "GRACE": GraceHyperParams}
assert METHOD in HPMAP

hp = HPMAP[METHOD].from_hparams(f"./hparams/{METHOD}/{YAML_NAME}.yaml")
hp.device = 0
if METHOD in ("MEMIT", "AlphaEdit") and hasattr(hp, "mom2_n_samples"):
    hp.mom2_n_samples = 3000
editor = BaseEditor.from_hparams(hp)
model, tok = editor.model, editor.tok
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
device = next(model.parameters()).device

orig_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
_changed_keys = []

def detect_changed():
    sd = model.state_dict()
    for k, v in orig_state.items():
        if sd[k].shape == v.shape and not torch.equal(sd[k].detach().cpu(), v):
            _changed_keys.append(k)

def restore():
    sd = model.state_dict()
    for k in _changed_keys:
        sd[k].copy_(orig_state[k].to(sd[k].device))

cards, golds, _groups = load_akew("CounterFact", "structured")
_tr, _va, test = subject_disjoint_split(cards, train_frac=0.7, val_frac=0.15, seed=0)
random.seed(0)
if N and N < len(test):
    test = random.sample(test, N)

# WISE and GRACE both add a new module (a side-memory layer / a codebook)
# rather than modifying an existing tensor's VALUES in place, so both change
# the model's state_dict KEY STRUCTURE and both hit the identical KeyError
# the WISE fix above describes -- confirmed by running GRACE and hitting the
# exact same 'transformer.h.N.mlp.fc_out.bias' KeyError. Same accumulating-
# mode fix applies to both, not just WISE.
ACCUMULATING_METHODS = ("WISE", "GRACE")
ACCUMULATING = (METHOD in ACCUMULATING_METHODS)

hits, edit_ok, edit_fail = [], 0, 0
for c in test:
    g = golds.get(c.edit_id)
    if not g or not g.eval_question or not g.target_new or not c.canonical_fact_text:
        continue
    prompt = c.canonical_fact_text.rsplit(" ", 1)[0] if " " in c.canonical_fact_text else c.canonical_fact_text
    try:
        metrics, _, _ = editor.edit(
            prompts=[prompt], subject=[c.subject], target_new=[" " + str(g.target_new)],
            sequential_edit=True, keep_original_weight=False, verbose=False,
            **({"loc_prompts": [prompt + " " + str(g.target_true or g.target_new)]} if METHOD == "WISE" else {}))
        if ACCUMULATING:
            # EasyEdit's own post-edit metric is the success signal here, since
            # state-dict key diffing doesn't apply once the module structure
            # itself has changed.
            post_acc = metrics[0].get("post", {}).get("rewrite_acc", 0.0)
            if isinstance(post_acc, (list, tuple)):
                post_acc = post_acc[0] if post_acc else 0.0
            if float(post_acc) <= 0.0:
                edit_fail += 1
                hits.append(False)
                continue
            edit_ok += 1
        else:
            if not _changed_keys:
                detect_changed()
            if not _changed_keys:
                edit_fail += 1
                hits.append(False)
                continue
            edit_ok += 1
        ids = tok(g.eval_question, return_tensors="pt").to(device)
        out = model.generate(**ids, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
        hits.append(is_hit(ans, g))
    except Exception as e:
        edit_fail += 1
        hits.append(False)
        print(f"EDIT_ERROR {c.edit_id}: {type(e).__name__}: {e}", file=sys.stderr)
    finally:
        if not ACCUMULATING:
            restore()

n = len(hits)
if edit_ok == 0:
    print(f"FATAL: every edit failed for {METHOD} on {MODEL_LABEL} -- edit_ok=0/{n}", file=sys.stderr)
    sys.exit(3)

out = {"method": METHOD, "model": MODEL_LABEL, "dataset": "CounterFact", "input_mode": "structured",
       "n": n, "edit_ok": edit_ok, "edit_fail": edit_fail,
       "accuracy": round(sum(hits) / n, 4) if n else None}
if ACCUMULATING:
    out["accumulating_mode"] = True
print("<<<JSON>>>")
print(json.dumps(out))
print("<<<END>>>")
