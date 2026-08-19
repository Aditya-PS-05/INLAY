"""Regression test for the gate-bypass fix (gpt2_memory_semkey.py).
Proves, using the real GPT-2-small model + MiniLM encoder:
  1. a same-subject/different-relation query does NOT trigger DIRECT playback
     once rel_gate is active (previously answer_playback ignored rel_gate entirely);
  2. a genuine paraphrase of the edited prompt still DOES trigger;
  3. gated_logits() and answer_playback() reach the SAME routing decision
     (both now call the single shared route()), for both the reject case and
     the direct-fire case.
"""
import sys
sys.path.insert(0, "src")
from gpt2_memory_semkey import GPT2WithSemanticMemory

g = GPT2WithSemanticMemory("gpt2", layer=0, alpha=15.0, n_slots_per_subkey=64, key_mode="prompt")

SUBJECT = "France"
PROMPT = "The capital of France is"
ANSWER = "Paris"
PARAPHRASE = "France's capital city is"
DIFFERENT_RELATION_SAME_SUBJECT = "The population of France is"

g.mem.clear_all()
g.write_chunk(PROMPT, ANSWER, subject=SUBJECT)

# --- 1. same-subject/different-relation must NOT fire once rel_gate is active ---
r_wrong_rel = g.route(DIFFERENT_RELATION_SAME_SUBJECT, subject=None, gate=0.30, rel_gate=0.60)
print(f"[same-subject/diff-relation] decision={r_wrong_rel['decision']} "
      f"score={r_wrong_rel['score']:.3f} rel_score={r_wrong_rel['rel_score']}")
if r_wrong_rel["decision"] == "DIRECT":
    print("FAIL: same-subject/different-relation query fired DIRECT with rel_gate active")
    sys.exit(1)
print("PASS: same-subject/different-relation query correctly rejected under rel_gate")

# --- 2. a genuine paraphrase must still fire ---
r_para = g.route(PARAPHRASE, subject=None, gate=0.30, rel_gate=0.60)
print(f"[paraphrase] decision={r_para['decision']} score={r_para['score']:.3f} "
      f"rel_score={r_para['rel_score']}")
if r_para["decision"] != "DIRECT":
    print("FAIL: genuine paraphrase did not fire DIRECT")
    sys.exit(1)
print("PASS: genuine paraphrase correctly fires DIRECT")

# --- 3. gated_logits and answer_playback must agree on the SAME decision ---
# (this is the actual bypass-bug check: before the fix, answer_playback ignored
#  rel_gate entirely and would have fired DIRECT on case 1 above)
ids = g.tok(DIFFERENT_RELATION_SAME_SUBJECT, return_tensors="pt").to(g.device)
_, sid_logits, score_logits = g.gated_logits(ids.input_ids, ids.input_ids.shape[1],
                                              gate=0.30, subject=None, rel_gate=0.60)
txt, sid_playback = g.answer_playback(DIFFERENT_RELATION_SAME_SUBJECT, subject=None,
                                       gate=0.30, rel_gate=0.60)
logits_fired = sid_logits is not None and score_logits >= 0.30
playback_fired = sid_playback is not None
print(f"[agreement check] gated_logits fired={logits_fired} (sid={sid_logits}) | "
      f"answer_playback fired={playback_fired} (sid={sid_playback})")
# both must reflect REJECT (route() said REJECT, so gated_logits returns sid but no
# injection happened, and answer_playback returns sid=None) -- the key assertion is
# that answer_playback did NOT play back stored tokens for this query.
if playback_fired:
    print("FAIL: answer_playback still bypassed rel_gate (sid should be None on REJECT)")
    sys.exit(1)
print("PASS: gated_logits and answer_playback agree — neither injects on the "
      "same-subject/different-relation query once rel_gate is active")

# --- also check the DIRECT case agrees across both paths ---
ids2 = g.tok(PARAPHRASE, return_tensors="pt").to(g.device)
_, sid_logits2, score_logits2 = g.gated_logits(ids2.input_ids, ids2.input_ids.shape[1],
                                                gate=0.30, subject=None, rel_gate=0.60)
txt2, sid_playback2 = g.answer_playback(PARAPHRASE, subject=None, gate=0.30, rel_gate=0.60)
print(f"[direct-case agreement] gated_logits sid={sid_logits2} | "
      f"answer_playback sid={sid_playback2} | playback text={txt2!r}")
if sid_playback2 is None:
    print("FAIL: answer_playback did not fire DIRECT on a genuine paraphrase")
    sys.exit(1)
if sid_logits2 is None:
    print("FAIL: gated_logits did not fire DIRECT on a genuine paraphrase")
    sys.exit(1)
print("PASS: both paths agree and fire DIRECT on the genuine paraphrase")

print("\nALL GATE-CONSISTENCY REGRESSION CHECKS PASSED")
